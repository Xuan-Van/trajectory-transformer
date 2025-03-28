import numpy as np
import math
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

from .ein import EinLinear

class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制模块
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 键、查询、值的投影层，用于所有注意力头
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # 正则化
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # 输出投影层
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # 因果掩码，确保注意力只应用于输入序列的左侧
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        ## 掩码之前的值估计
        joined_dim = config.observation_dim + config.action_dim + 2
        self.mask.squeeze()[:,joined_dim-1::joined_dim] = 0
        ##
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # 计算所有头的查询、键、值，并将头维度移到批处理维度
        ## [ B x n_heads x T x head_dim ]
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力；自注意力：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        ## [ B x n_heads x T x T ]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        ## [ B x n_heads x T x head_size ]
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        ## [ B x T x embedding_dim ]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 重新组合所有头的输出

        # 输出投影
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """
    基本块，包含自注意力和前馈网络
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """
    完整的GPT语言模型，具有block_size的上下文大小
    """
    def __init__(self, config):
        super().__init__()

        # 输入嵌入层（+1用于停止标记）
        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # 变压器
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # 解码器头
        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head = EinLinear(config.transition_dim, config.n_embd, config.vocab_size + 1, bias=False)

        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim

        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight

        self.embedding_dim = config.n_embd
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)): # 线性层和嵌入层的权重初始化
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm): # 层归一化的权重初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        配置优化器，将模型的所有参数分为两个桶：
        1. 需要正则化权重衰减的参数
        2. 不需要正则化权重衰减的参数（偏置和层归一化/嵌入权重）
        """
        # 分离所有参数到两个桶
        decay = set() # 存储需要正则化权重衰减的参数
        no_decay = set() # 存储不需要正则化权重衰减的参数
        whitelist_weight_modules = (torch.nn.Linear, EinLinear) # 包含需要权重衰减的模块类型
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding) # 包含不需要权重衰减的模块类型
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # 完整参数名称

                if pn.endswith('bias'): # 参数名称以bias结尾，则将其添加到no_decay集合中
                    # 所有偏置不会被衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules): # 参数名称以weight结尾且模块类型在白名单中，则将其添加到decay集合中
                    # 白名单模块的权重将被权重衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules): # 参数名称以weight结尾且模块类型在黑名单中，则将其添加到no_decay集合中
                    # 黑名单模块的权重不会被权重衰减
                    no_decay.add(fpn)

        # 特殊处理位置嵌入参数，不进行衰减
        no_decay.add('pos_emb')

        # 验证我们考虑了每个参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "参数 %s 同时出现在衰减/非衰减集合中!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "参数 %s 没有被分离到衰减/非衰减集合中!" \
                                                    % (str(param_dict.keys() - union_params), )

        # 创建PyTorch优化器对象
        optim_groups = [ # 创建两个优化器组，分别包含需要权重衰减和不需要权重衰减的参数
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def offset_tokens(self, idx):
        _, t = idx.shape # 获取输入索引的形状
        n_states = int(np.ceil(t / self.transition_dim)) # 计算状态的数量
        offsets = torch.arange(self.transition_dim) * self.vocab_size # 计算偏移量
        offsets = offsets.repeat(n_states).to(idx.device) # offsets被重复n_states次，并移动到与idx相同的设备上
        offset_idx = idx + offsets[:t] # 应用偏移量
        offset_idx[idx == self.vocab_size] = self.stop_token # 处理停止标记
        return offset_idx

    def pad_to_full_observation(self, x, verify=False):
        b, t, _ = x.shape
        n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim # 计算需要填充的长度
        padding = torch.zeros(b, n_pad, self.embedding_dim, device=x.device) # 创建填充张量
        ## [ B x T' x embedding_dim ]
        x_pad = torch.cat([x, padding], dim=1) # 填充输入张量
        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim) # 重塑填充后的张量
        if verify: # 验证填充结果
            self.verify(x, x_pad)
        return x_pad, n_pad

    def verify(self, x, x_pad):
        b, t, embedding_dim = x.shape # 获取输入张量的形状
        n_states = int(np.ceil(t / self.transition_dim)) # 计算状态的数量
        inds = torch.arange(0, self.transition_dim).repeat(n_states)[:t] # 创建索引张量
        for i in range(self.transition_dim):
            x_ = x[:,inds == i] # 对于每个transition_dim维度i，从原始张量x中提取相应的子序列x_
            t_ = x_.shape[1]
            x_pad_ = x_pad[:,i].view(b, n_states, embedding_dim)[:,:t_] # 从填充后的张量x_pad中提取相应的子序列x_pad_
            print(i, x_.shape, x_pad_.shape)
            try:
                assert (x_ == x_pad_).all() # 使用断言检查x_和x_pad_是否完全相等
            except:
                pdb.set_trace() # 如果断言失败，则调用pdb.set_trace()进入调试模式，以便进一步检查问题

    def forward(self, idx, targets=None, mask=None):
        """
            idx : [ B x T ]
            values : [ B x 1 x 1 ]
        """
        b, t = idx.size()
        assert t <= self.block_size, "无法前向传播，模型块大小已耗尽。" # 确保序列长度不超过模型的块大小self.block_size

        offset_idx = self.offset_tokens(idx) # 调用offset_tokens方法将输入索引进行偏移，以便在嵌入层中正确映射到相应的嵌入向量
        ## [ B x T x embedding_dim ]
        # 前向传播GPT模型
        token_embeddings = self.tok_emb(offset_idx) # 每个索引映射到一个可学习的向量
        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # 每个位置映射到一个可学习的向量
        ## [ B x T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings) # 将token_embeddings和position_embeddings相加，并通过self.drop进行dropout处理
        x = self.blocks(x) # self.blocks是变压器块的序列，包含多个Block模块
        ## [ B x T x embedding_dim ]
        x = self.ln_f(x) # self.ln_f是最终的层归一化

        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad, n_pad = self.pad_to_full_observation(x) # 调用pad_to_full_observation方法将嵌入向量填充到完整的观察维度
        ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        logits = self.head(x_pad) # self.head是最终的线性层，将嵌入向量映射到词汇表大小加一的维度
        ## [ B x T' x (vocab_size + 1) ]
        logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        ## [ B x T x (vocab_size + 1) ]
        logits = logits[:,:t] # 将logits重塑为(batch_size,sequence_length+n_pad,vocab_size+1)的形状，并截取前t个时间步

        # 如果有目标，计算损失
        if targets is not None: # 如果提供了目标标签targets，则计算交叉熵损失
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
            if self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1: # 如果action_weight、reward_weight或value_weight不等于1，则创建权重张量并应用于损失
                #### 创建权重
                n_states = int(np.ceil(t / self.transition_dim))
                weights = torch.cat([
                    torch.ones(self.observation_dim, device=idx.device),
                    torch.ones(self.action_dim, device=idx.device) * self.action_weight,
                    torch.ones(1, device=idx.device) * self.reward_weight,
                    torch.ones(1, device=idx.device) * self.value_weight,
                ])
                ## [ t + 1]
                weights = weights.repeat(n_states)
                ## [ b x t ]
                weights = weights[1:].repeat(b, 1)
                ####
                loss = loss * weights.view(-1)
            loss = (loss * mask.view(-1)).mean() # 最终损失通过mask进行掩码，并计算平均值
        else:
            loss = None

        return logits, loss

class ConditionalGPT(GPT):
    """
    条件GPT模型，增加了目标观察
    """
    def __init__(self, config):
        ## 增加块大小，因为我们在序列前添加了目标观察
        config.block_size += config.observation_dim
        super().__init__(config) # 调用父类GPT的初始化方法
        self.goal_emb = nn.Embedding(config.vocab_size * config.observation_dim, config.n_embd) # 创建目标嵌入层self.goal_emb，用于嵌入目标观察

    def get_block_size(self): # 返回块大小，减去目标观察的维度
        return self.block_size - self.observation_dim

    def forward(self, idx, goal, targets=None, mask=None):
        # 获取输入索引 idx 的形状，并确保序列长度不超过块大小
        b, t = idx.size()
        assert t <= self.block_size, "无法前向传播，模型块大小已耗尽。"

        #### 目标
        offset_goal = self.offset_tokens(goal) # 调用offset_tokens方法将目标观察goal进行偏移
        goal_embeddings = self.goal_emb(offset_goal) # 使用目标嵌入层self.goal_emb将偏移后的目标观察嵌入到嵌入向量中
        #### /目标

        offset_idx = self.offset_tokens(idx) # 调用offset_tokens方法将输入索引idx进行偏移
        ## [ B x T x embedding_dim ]
        # 前向传播GPT模型
        token_embeddings = self.tok_emb(offset_idx) # 每个索引映射到一个可学习的向量
        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # 每个位置映射到一个可学习的向量
        ## [ B x T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings) # 计算输入索引的嵌入向量和位置嵌入向量，并通过self.drop进行dropout处理

        #### 目标
        ## [ B + (obs_dim + T) x embedding_dim ]
        gx = torch.cat([goal_embeddings, x], dim=1) # 将目标嵌入向量goal_embeddings和输入序列嵌入向量x拼接在一起
        gx = self.blocks(gx) # 通过变压器块self.blocks进行前向传播
        x = gx[:, self.observation_dim:] # 从拼接后的张量中提取输入序列的部分
        #### /目标

        ## [ B x T x embedding_dim ]
        x = self.ln_f(x) # 通过最终的层归一化self.ln_f进行处理

        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad, n_pad = self.pad_to_full_observation(x) # 调用pad_to_full_observation方法将嵌入向量填充到完整的观察维度
        ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        logits = self.head(x_pad) # 通过最终的线性层self.head计算预测结果
        ## [ B x T' x (vocab_size + 1) ]
        logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        ## [ B x T x (vocab_size + 1) ]
        logits = logits[:,:t] # 重塑和截取预测结果

        # 如果有目标，计算损失
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
            if self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
                #### 创建权重
                n_states = int(np.ceil(t / self.transition_dim))
                weights = torch.cat([
                    torch.ones(self.observation_dim, device=idx.device),
                    torch.ones(self.action_dim, device=idx.device) * self.action_weight,
                    torch.ones(1, device=idx.device) * self.reward_weight,
                    torch.ones(1, device=idx.device) * self.value_weight,
                ])
                ## [ t + 1]
                weights = weights.repeat(n_states)
                ## [ b x t ]
                weights = weights[1:].repeat(b, 1)
                ####
                loss = loss * weights.view(-1)
            loss = (loss * mask.view(-1)).mean()
        else:
            loss = None

        return logits, loss