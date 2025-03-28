import os
import numpy as np
import torch
import pdb

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.models.transformers import GPT


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'  # 数据集名称
    config: str = 'config.offline'  # 配置文件路径

#######################
######## 设置 ########
#######################

args = Parser().parse_args('train')  # 解析命令行参数

#######################
####### 数据集 #######
#######################

env = datasets.load_environment(args.dataset)  # 加载环境

sequence_length = args.subsampled_sequence_length * args.step  # 计算序列长度

dataset_config = utils.Config(
    datasets.DiscretizedDataset,  # 数据集类
    savepath=(args.savepath, 'data_config.pkl'),  # 配置保存路径
    env=args.dataset,  # 环境名称
    N=args.N,  # 离散化级别
    penalty=args.termination_penalty,  # 终止惩罚
    sequence_length=sequence_length,  # 序列长度
    step=args.step,  # 步长
    discount=args.discount,  # 折扣因子
    discretizer=args.discretizer,  # 离散化方法
)

dataset = dataset_config()  # 创建数据集实例
obs_dim = dataset.observation_dim  # 观察维度
act_dim = dataset.action_dim  # 动作维度
transition_dim = dataset.joined_dim  # 联合维度（观察+动作）

#######################
######## 模型 ########
#######################

block_size = args.subsampled_sequence_length * transition_dim - 1  # 计算块大小
print(
    f'Dataset size: {len(dataset)} | '
    f'Joined dim: {transition_dim} '
    f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
)

model_config = utils.Config(
    GPT,  # 模型类
    savepath=(args.savepath, 'model_config.pkl'),  # 配置保存路径
    ## 离散化
    vocab_size=args.N, block_size=block_size,
    ## 架构
    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd*args.n_head,
    ## 维度
    observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
    ## 损失权重
    action_weight=args.action_weight, reward_weight=args.reward_weight, value_weight=args.value_weight,
    ## 丢弃概率
    embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
)

model = model_config()  # 创建模型实例
model.to(args.device)  # 将模型移动到指定设备

#######################
####### 训练器 #######
#######################

warmup_tokens = len(dataset) * block_size  # 每个epoch看到的token数量
final_tokens = 20 * warmup_tokens  # 最终token数量

trainer_config = utils.Config(
    utils.Trainer,  # 训练器类
    savepath=(args.savepath, 'trainer_config.pkl'),  # 配置保存路径
    # 优化参数
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1,  # 仅应用于矩阵乘法的权重
    # 学习率衰减：线性预热后余弦衰减到原始值的10%
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    ## 数据加载器
    num_workers=0,
    device=args.device,
)

trainer = trainer_config()  # 创建训练器实例

#######################
###### 主循环 ######
#######################

## 根据更新次数保持恒定来缩放epoch数量
n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
save_freq = int(n_epochs // args.n_saves)

for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} | {args.exp_name}')

    trainer.train(model, dataset)  # 训练模型

    ## 获取小于或等于`save_epoch`的最大`save_freq`倍数
    save_epoch = (epoch + 1) // save_freq * save_freq
    statepath = os.path.join(args.savepath, f'state_{save_epoch}.pt')
    print(f'Saving model to {statepath}')

    ## 保存状态到磁盘
    state = model.state_dict()
    torch.save(state, statepath)