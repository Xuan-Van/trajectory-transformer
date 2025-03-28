import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入torch库，用于深度学习
import pdb  # 导入pdb库，用于调试

from ..utils.arrays import to_torch  # 从上级目录的utils模块中导入to_torch函数

VALUE_PLACEHOLDER = 1e6  # 定义一个占位符值，用于表示未知的值

def make_prefix(discretizer, context, obs, prefix_context=True):
    """
    生成前缀序列，用于模型输入
    :param discretizer: 离散化器对象
    :param context: 上下文列表，包含之前的离散化观测值
    :param obs: 当前观测值
    :param prefix_context: 是否将上下文包含在前缀中
    :return: 前缀序列，包含上下文和当前观测值的离散化结果
    """
    observation_dim = obs.size  # 获取观测值的维度
    obs_discrete = discretizer.discretize(obs, subslice=[0, observation_dim])  # 将观测值离散化
    obs_discrete = to_torch(obs_discrete, dtype=torch.long)  # 将离散化后的观测值转换为torch张量

    if prefix_context:  # 如果需要包含上下文
        prefix = torch.cat(context + [obs_discrete], dim=-1)  # 将上下文和当前观测值拼接成前缀
    else:
        prefix = obs_discrete  # 否则，前缀仅包含当前观测值

    return prefix  # 返回前缀序列

def extract_actions(x, observation_dim, action_dim, t=None):
    """
    从输入数据中提取动作序列
    :param x: 输入数据，包含观测值、动作值、奖励值和占位符值
    :param observation_dim: 观测值的维度
    :param action_dim: 动作值的维度
    :param t: 时间步，如果指定则返回该时间步的动作值
    :return: 动作序列或指定时间步的动作值
    """
    assert x.shape[1] == observation_dim + action_dim + 2  # 确保输入数据的维度正确
    actions = x[:, observation_dim:observation_dim+action_dim]  # 提取动作值
    if t is not None:  # 如果指定了时间步
        return actions[t]  # 返回该时间步的动作值
    else:
        return actions  # 否则返回整个动作序列

def update_context(context, discretizer, observation, action, reward, max_context_transitions):
    """
    更新上下文列表
    :param context: 当前上下文列表
    :param discretizer: 离散化器对象
    :param observation: 当前观测值
    :param action: 当前动作值
    :param reward: 当前奖励值
    :param max_context_transitions: 上下文列表的最大长度
    :return: 更新后的上下文列表
    """
    rew_val = np.array([reward, VALUE_PLACEHOLDER])  # 创建包含奖励值和占位符值的数组
    transition = np.concatenate([observation, action, rew_val])  # 将观测值、动作值和奖励值拼接成一个过渡

    transition_discrete = discretizer.discretize(transition)  # 将过渡离散化
    transition_discrete = to_torch(transition_discrete, dtype=torch.long)  # 将离散化后的过渡转换为torch张量

    context.append(transition_discrete)  # 将离散化后的过渡添加到上下文列表中

    context = context[-max_context_transitions:]  # 如果上下文列表超过最大长度，则截取最近的过渡

    return context  # 返回更新后的上下文列表