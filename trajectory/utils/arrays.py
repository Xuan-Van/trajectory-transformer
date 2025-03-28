import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习

DTYPE = torch.float  # 定义默认的数据类型为浮点型
DEVICE = 'cuda:0'  # 定义默认的设备为第一个GPU

def to_np(x):
    """
    将PyTorch张量转换为NumPy数组
    :param x: PyTorch张量
    :return: NumPy数组
    """
    if torch.is_tensor(x):  # 检查输入是否为PyTorch张量
        x = x.detach().cpu().numpy()  # 将张量从GPU移到CPU并转换为NumPy数组
    return x

def to_torch(x, dtype=None, device=None):
    """
    将NumPy数组或其他数据类型转换为PyTorch张量
    :param x: 输入数据
    :param dtype: 数据类型，默认为DTYPE
    :param device: 设备类型，默认为DEVICE
    :return: PyTorch张量
    """
    dtype = dtype or DTYPE  # 如果没有指定数据类型，则使用默认的DTYPE
    device = device or DEVICE  # 如果没有指定设备，则使用默认的DEVICE
    return torch.tensor(x, dtype=dtype, device=device)  # 创建并返回PyTorch张量

def to_device(*xs, device=DEVICE):
    """
    将多个PyTorch张量移动到指定设备
    :param xs: 多个PyTorch张量
    :param device: 目标设备，默认为DEVICE
    :return: 移动到指定设备后的张量列表
    """
    return [x.to(device) for x in xs]  # 将每个张量移动到指定设备并返回列表

def normalize(x):
    """
    将输入数据缩放到[0, 1]范围内
    :param x: 输入数据
    :return: 归一化后的数据
    """
    x = x - x.min()  # 将数据平移到最小值为0
    x = x / x.max()  # 将数据缩放到最大值为1
    return x

def to_img(x):
    """
    将输入数据转换为图像格式（0-255，uint8）
    :param x: 输入数据
    :return: 转换后的图像数据
    """
    normalized = normalize(x)  # 归一化输入数据
    array = to_np(normalized)  # 将归一化后的数据转换为NumPy数组
    array = np.transpose(array, (1, 2, 0))  # 调整数组的维度顺序，使其适合图像显示 $ 转置操作，如(C,W,H)->(H,W,C)
    return (array * 255).astype(np.uint8)  # 将数据缩放到0-255范围，并转换为uint8类型

def set_device(device):
    """
    设置默认的设备类型
    :param device: 目标设备
    """
    DEVICE = device  # 更新默认设备
    if 'cuda' in device:  # 如果设备是GPU
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # 设置默认的张量类型为GPU上的浮点型