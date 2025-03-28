import math  # 导入数学库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import pdb  # 导入Python调试器

class EinLinear(nn.Module):
    """
    自定义的线性层，支持多个模型的并行计算。
    """

    def __init__(self, n_models, in_features, out_features, bias):
        """
        初始化函数。
        参数:
        n_models (int): 模型的数量。
        in_features (int): 输入特征的数量。
        out_features (int): 输出特征的数量。
        bias (bool): 是否使用偏置项。
        """
        super().__init__()  # 调用父类nn.Module的初始化函数
        self.n_models = n_models  # 保存模型的数量
        self.out_features = out_features  # 保存输出特征的数量
        self.in_features = in_features  # 保存输入特征的数量
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))  # 定义权重参数
        if bias:  # 如果使用偏置项
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))  # 定义偏置参数
        else:
            self.register_parameter('bias', None)  # 不使用偏置项
        self.reset_parameters()  # 初始化权重和偏置

    def reset_parameters(self):
        """
        初始化权重和偏置。
        """
        for i in range(self.n_models):  # 遍历每个模型
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))  # 使用Kaiming初始化权重
            if self.bias is not None:  # 如果使用偏置项
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])  # 计算输入特征的数量
                bound = 1 / math.sqrt(fan_in)  # 计算偏置的初始化范围
                nn.init.uniform_(self.bias[i], -bound, bound)  # 均匀分布初始化偏置

    def forward(self, input):
        """
        前向传播函数。
        参数:
        input (Tensor): 输入张量，形状为 [B x n_models x input_dim]。
        返回:
        output (Tensor): 输出张量，形状为 [B x n_models x output_dim]。
        """
        ## [ B x n_models x output_dim ]
        output = torch.einsum('eoi,bei->beo', self.weight, input)  # 使用爱因斯坦求和约定计算输出
        if self.bias is not None:  # 如果使用偏置项
            raise RuntimeError()  # 抛出运行时错误（当前代码未实现偏置项的加法）
        return output  # 返回输出张量

    def extra_repr(self):
        """
        返回模块的额外信息。
        """
        return 'n_models={}, in_features={}, out_features={}, bias={}'.format(
            self.n_models, self.in_features, self.out_features, self.bias is not None
        )