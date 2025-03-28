import time  # 导入时间模块，用于时间相关的操作
import sys  # 导入系统模块，用于与Python运行时环境交互
import os  # 导入操作系统接口模块，用于文件和目录操作
import glob  # 导入文件模式匹配模块，用于查找符合特定规则的文件路径名
import pickle  # 导入序列化模块，用于对象的序列化和反序列化
import json  # 导入JSON模块，用于处理JSON数据
import torch  # 导入PyTorch模块，用于深度学习相关的操作
import pdb  # 导入Python调试器模块，用于调试代码

def mkdir(savepath, prune_fname=False):
    """
    创建目录。
    参数:
    savepath (str): 要创建的目录路径。
    prune_fname (bool): 如果为True，则去除路径中的文件名部分。
    返回:
    bool: 如果目录创建成功，返回True；否则返回False。
    """
    if prune_fname:  # 如果需要去除文件名部分
        savepath = os.path.dirname(savepath)  # 获取目录路径
    if not os.path.exists(savepath):  # 如果目录不存在
        try:
            os.makedirs(savepath)  # 创建目录
        except:
            print(f'[ utils/serialization ] Warning: did not make directory: {savepath}')  # 打印警告信息
            return False  # 返回False表示目录未创建成功
        return True  # 返回True表示目录创建成功
    else:
        return False  # 返回False表示目录已存在

def get_latest_epoch(loadpath):
    """
    获取指定路径下最新的模型状态文件对应的epoch。
    参数:
    loadpath (str): 模型状态文件所在的目录路径。
    返回:
    int: 最新的epoch值，如果没有找到任何状态文件则返回-1。
    """
    states = glob.glob1(loadpath, 'state_*')  # 获取所有以'state_'开头的文件名
    latest_epoch = -1  # 初始化最新的epoch为-1
    for state in states:  # 遍历所有状态文件
        epoch = int(state.replace('state_', '').replace('.pt', ''))  # 提取epoch值
        latest_epoch = max(epoch, latest_epoch)  # 更新最新的epoch值
    return latest_epoch  # 返回最新的epoch值

def load_model(*loadpath, epoch=None, device='cuda:0'):
    """
    加载指定epoch的模型。
    参数:
    *loadpath (str): 模型文件的路径。
    epoch (int or str): 要加载的模型对应的epoch值，如果为'latest'则加载最新的epoch。
    device (str): 模型加载到的设备，默认为'cuda:0'。
    返回:
    tuple: 包含加载的模型和对应的epoch值。
    """
    loadpath = os.path.join(*loadpath)  # 拼接路径
    config_path = os.path.join(loadpath, 'model_config.pkl')  # 模型配置文件路径

    if epoch == 'latest':  # 如果epoch为'latest'
        epoch = get_latest_epoch(loadpath)  # 获取最新的epoch值

    print(f'[ utils/serialization ] Loading model epoch: {epoch}')  # 打印加载的epoch信息
    state_path = os.path.join(loadpath, f'state_{epoch}.pt')  # 模型状态文件路径

    config = pickle.load(open(config_path, 'rb'))  # 加载模型配置
    state = torch.load(state_path)  # 加载模型状态

    model = config()  # 根据配置创建模型
    model.to(device)  # 将模型移动到指定设备
    model.load_state_dict(state, strict=True)  # 加载模型状态字典

    print(f'\n[ utils/serialization ] Loaded config from {config_path}\n')  # 打印加载的配置信息
    print(config)  # 打印配置内容

    return model, epoch  # 返回加载的模型和对应的epoch值

def load_config(*loadpath):
    """
    加载模型配置。
    参数:
    *loadpath (str): 配置文件的路径。
    返回:
    object: 加载的配置对象。
    """
    loadpath = os.path.join(*loadpath)  # 拼接路径
    config = pickle.load(open(loadpath, 'rb'))  # 加载配置
    print(f'[ utils/serialization ] Loaded config from {loadpath}')  # 打印加载的配置信息
    print(config)  # 打印配置内容
    return config  # 返回加载的配置对象

def load_from_config(*loadpath):
    """
    从配置文件中创建并返回模型。
    参数:
    *loadpath (str): 配置文件的路径。
    返回:
    object: 根据配置创建的模型对象。
    """
    config = load_config(*loadpath)  # 加载配置
    return config.make()  # 根据配置创建并返回模型

def load_args(*loadpath):
    """
    加载命令行参数。
    参数:
    *loadpath (str): 参数文件的路径。
    返回:
    object: 加载的参数对象。
    """
    from .setup import Parser  # 导入解析器类
    loadpath = os.path.join(*loadpath)  # 拼接路径
    args_path = os.path.join(loadpath, 'args.json')  # 参数文件路径
    args = Parser()  # 创建解析器对象
    args.load(args_path)  # 加载参数
    return args  # 返回加载的参数对象