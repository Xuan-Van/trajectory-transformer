import numpy as np
import torch
import torch.nn as nn
import pdb

def make_weights(N, weights):
    # 确保权重数量是奇数
    assert len(weights) % 2 == 1, f'Expected odd number of weights, got: {weights}'
    # 计算权重的中心位置
    center = int((len(weights) - 1) / 2)

    # 初始化一个N x N的零矩阵
    tokens = np.zeros((N, N))
    # 遍历每一行
    for i in range(N):
        # 初始化一个N维零向量
        token = np.zeros(N)
        # 遍历权重列表
        for j, w in enumerate(weights):
            # 计算当前权重对应的索引
            ind = i + j - center
            # 将索引限制在0到N-1之间
            ind = np.clip(ind, 0, N-1)
            # 将权重加到对应的索引位置
            token[ind] += w
        # 将生成的token赋值给tokens矩阵的第i行
        tokens[i] = token
    # 确保每一行的权重和为1
    assert np.allclose(tokens.sum(axis=-1), 1)
    return tokens

def add_stop_token(tokens):
    # 获取tokens的行数
    N = len(tokens)
    # 为每个token添加一个概率为0的停止token
    pad = np.zeros((N, 1))
    tokens = np.concatenate([tokens, pad], axis=1)
    # 创建一个停止token，其概率为1
    stop_weight = np.zeros((1, N+1))
    stop_weight[0,-1] = 1
    # 将停止token添加到tokens矩阵中
    tokens = np.concatenate([tokens, stop_weight], axis=0)

    # 确保矩阵是方阵且每行的权重和为1
    assert tokens.shape[0] == tokens.shape[1]
    assert np.allclose(tokens.sum(axis=-1), 1)
    return tokens

class SmoothEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, weights, stop_token=False):
        super().__init__()
        # 生成权重矩阵
        self.weights = make_weights(num_embeddings, weights)
        # 如果需要停止token，则添加停止token
        if stop_token:
            self.weights = add_stop_token(self.weights)
            num_embeddings += 1
        # 将权重矩阵转换为torch张量并移动到GPU
        self.weights = torch.tensor(self.weights, dtype=torch.float, device='cuda:0')
        # 生成索引张量
        self.inds = torch.arange(0, num_embeddings, device='cuda:0')
        # 初始化嵌入层
        self._embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        '''
            x : [ batch_size x context ]
        '''
        # 获取所有嵌入向量
        embed = self._embeddings(self.inds)
        # 获取输入x对应的权重矩阵
        weights = self.weights[x]
        # 确保每行的权重和为1
        assert torch.allclose(weights.sum(-1), torch.ones(1, device=weights.device))

        # 计算加权嵌入向量
        weighted_embed = torch.einsum('btn,nd->btd', weights, embed)
        return weighted_embed

if __name__ == '__main__':

    # 生成随机输入
    x = torch.randint(0, 100, size=(5, 10,)).cuda()

    # 测试带有权重的SmoothEmbedding
    embed = SmoothEmbedding(100, 32, weights=[0.15, 0.2, 0.3, 0.2, 0.15], stop_token=True)
    embed.cuda()
    out = embed(x)

    # 测试极限情况下的SmoothEmbedding
    embed_1 = SmoothEmbedding(100, 32, weights=[1.0], stop_token=True)
    embed_1.cuda()
    out_1 = embed_1(x)

    # 参考输出
    out_0 = embed_1._embeddings(x)

    # 检查输出是否相同
    print(f'Same: {(out_0 == out_1).all().item()}')