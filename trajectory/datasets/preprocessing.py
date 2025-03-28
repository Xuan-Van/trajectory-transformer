import numpy as np

def kitchen_preprocess_fn(observations):
    """
    预处理厨房环境的观测数据。
    参数:
    - observations (numpy.ndarray): 一个二维数组，每一行代表一个观测数据。
    返回:
    - numpy.ndarray: 一个二维数组，包含原始观测数据的前30个维度。
    """
    ## 保留60维观测数据的前30维
    keep = observations[:, :30] # 保留前30维数据
    remove = observations[:, 30:] # 移除后30维数据
    assert (remove.max(0) == remove.min(0)).all(), '移除了重要的状态信息' # 断言移除的部分不包含重要的状态信息
    return keep

def ant_preprocess_fn(observations):
    """
    预处理蚂蚁环境的观测数据。
    参数:
    - observations (numpy.ndarray): 一个二维数组，每一行代表一个观测数据。
    返回:
    - numpy.ndarray: 一个二维数组，包含原始观测数据的位置和速度信息。
    """
    qpos_dim = 13 # 移除了root_x和root_y，定义位置信息的维度为13
    qvel_dim = 14 # 定义速度信息的维度为14
    cfrc_dim = 84 # 定义接触力矩信息的维度为84
    assert observations.shape[1] == qpos_dim + qvel_dim + cfrc_dim # 断言观测数据的维度是否正确
    keep = observations[:, :qpos_dim + qvel_dim] # 保留位置和速度信息
    return keep

def vmap(fn):
    """
    将函数应用于输入数据的每一行。
    参数:
    - fn (function): 要应用的函数。
    返回:
    - function: 一个新的函数，可以处理一维或二维输入数据。
    """
    def _fn(inputs):
        if inputs.ndim == 1: # 处理一维输入数据
            inputs = inputs[None] # 如果输入数据是一维的，将其转换为二维数组
            return_1d = True # 标记输入数据是一维的，以便在后续处理中恢复为一维
        else:
            return_1d = False

        outputs = fn(inputs) # 将传入的函数fn应用于输入数据inputs，并将结果存储在outputs变量中

        if return_1d: # 检查是否需要将输出数据恢复为一维
            return outputs.squeeze(0) # 通过移除第一个维度，将二维数组转换为一维数组 $ squeeze：从数组的形状中移除单维度条目，即形状中为1的维度
        else:
            return outputs

    return _fn

def preprocess_dataset(preprocess_fn):
    """
    预处理数据集中的观测数据和下一个观测数据。
    参数:
    - preprocess_fn (function): 用于预处理观测数据的函数。
    返回:
    - function: 一个新的函数，可以处理整个数据集。
    """
    def _fn(dataset):
        for key in ['observations', 'next_observations']: # 这两个键通常分别对应当前观测数据和下一个观测数据
            dataset[key] = preprocess_fn(dataset[key]) # 对每个键对应的值应用传入的预处理函数preprocess_fn，并将结果重新赋值给数据集中的相应键
        return dataset

    return _fn

preprocess_functions = { # 存储不同厨房环境和蚂蚁环境对应的预处理函数
    'kitchen-complete-v0': vmap(kitchen_preprocess_fn),
    'kitchen-mixed-v0': vmap(kitchen_preprocess_fn),
    'kitchen-partial-v0': vmap(kitchen_preprocess_fn),
    'ant-expert-v2': vmap(ant_preprocess_fn),
    'ant-medium-expert-v2': vmap(ant_preprocess_fn),
    'ant-medium-replay-v2': vmap(ant_preprocess_fn),
    'ant-medium-v2': vmap(ant_preprocess_fn),
    'ant-random-v2': vmap(ant_preprocess_fn),
}

dataset_preprocess_functions = { # 存储不同环境对应的预处理数据集的函数
    k: preprocess_dataset(fn) for k, fn in preprocess_functions.items()
}

def get_preprocess_fn(env):
    """
    根据环境名称获取预处理函数。
    参数:
    - env (str): 环境名称。
    返回:
    - function: 对应的预处理函数，如果没有找到则返回恒等函数。
    """
    return preprocess_functions.get(env, lambda x: x)