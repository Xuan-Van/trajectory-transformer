import torch
import torch.nn as nn
import pdb

def get_activation(params):
    # 如果params是字典，则提取激活函数的名称和参数
    if type(params) == dict:
        name = params['type']
        kwargs = params['kwargs']
    else:
        # 否则，直接将params转换为字符串作为激活函数的名称
        name = str(params)
        kwargs = {}
    # 返回一个lambda函数，用于创建激活函数实例
    return lambda: getattr(nn, name)(**kwargs)

def flatten(condition_dict):
    # 对字典的键进行排序
    keys = sorted(condition_dict)
    # 根据排序后的键获取对应的值
    vals = [condition_dict[key] for key in keys]
    # 将所有值在最后一个维度上拼接
    condition = torch.cat(vals, dim=-1)
    return condition

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, activation='GELU', output_activation='Identity', name='mlp',
                 model_class=None):
        """
            @TODO: 清理从配置中实例化模型的代码，以便我们不需要将`model_class`传递给模型本身
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.name = name
        # 获取激活函数实例
        activation = get_activation(activation)
        output_activation = get_activation(output_activation)

        layers = []
        current = input_dim
        # 遍历隐藏层维度，逐层添加线性层和激活函数
        for dim in hidden_dims:
            linear = nn.Linear(current, dim)
            layers.append(linear)
            layers.append(activation())
            current = dim

        # 添加输出层和输出激活函数
        layers.append(nn.Linear(current, output_dim))
        layers.append(output_activation())

        # 将所有层组合成一个Sequential模块
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        return self._layers(x)

    @property
    def num_parameters(self):
        # 计算需要梯度更新的参数总数
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([p.numel() for p in parameters])

    def __repr__(self):
        # 自定义模型的字符串表示
        return '[ {} : {} parameters ] {}'.format(
            self.name, self.num_parameters,
            super().__repr__())

class FlattenMLP(MLP):

    def forward(self, *args):
        # 将所有输入在最后一个维度上拼接
        x = torch.cat(args, dim=-1)
        # 调用父类的前向传播方法
        return super().forward(x)