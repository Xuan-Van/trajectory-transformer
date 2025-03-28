import os  # 导入os模块，用于文件和目录操作
import numpy as np  # 导入numpy库，用于数值计算
import gym  # 导入gym库，用于强化学习环境
import pdb  # 导入pdb库，用于调试

from contextlib import (
    contextmanager,  # 导入contextmanager装饰器，用于创建上下文管理器
    redirect_stderr,  # 导入redirect_stderr函数，用于重定向标准错误输出
    redirect_stdout,  # 导入redirect_stdout函数，用于重定向标准输出
)

@contextmanager # 用于将一个生成器函数转换为上下文管理器 $ 上下文管理器是一种在进入和退出代码块时自动执行特定操作的机制，通常用于资源管理（如文件操作、数据库连接等）和异常处理
def suppress_output():
    """
    一个上下文管理器，将stdout和stderr重定向到devnull
    参考：https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:  # 打开devnull文件，用于丢弃输出
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:  # 重定向stderr和stdout到devnull
            yield (err, out)  # 返回上下文管理器

with suppress_output():  # 使用suppress_output上下文管理器
    ## d4rl会打印出各种警告信息
    import d4rl  # 导入d4rl库

# def construct_dataloader(dataset, **kwargs): # 创建一个数据加载器
#     dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, pin_memory=True, **kwargs)
#     return dataloader

def qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    从环境中提取Q-learning数据集，并处理时间终止（timeouts）的情况。
    参数:
    env: 环境对象
    dataset: 可选，预先加载的数据集
    terminate_on_end: 是否在每个episode的最后一步终止
    **kwargs: 传递给env.get_dataset的其他参数
    返回:
    包含处理后的数据集的字典
    """
    if dataset is None:  # 如果未提供数据集
        dataset = env.get_dataset(**kwargs)  # 从环境中获取数据集

    N = dataset['rewards'].shape[0]  # 获取数据集中的样本数量
    obs_ = []  # 初始化观察列表
    next_obs_ = []  # 初始化下一个观察列表
    action_ = []  # 初始化动作列表
    reward_ = []  # 初始化奖励列表
    done_ = []  # 初始化终止标志列表
    realdone_ = []  # 初始化真实终止标志列表

    episode_step = 0  # 初始化episode步数
    for i in range(N-1):  # 遍历数据集中的每个样本
        obs = dataset['observations'][i]  # 获取当前观察
        new_obs = dataset['observations'][i+1]  # 获取下一个观察
        action = dataset['actions'][i]  # 获取当前动作
        reward = dataset['rewards'][i]  # 获取当前奖励
        done_bool = bool(dataset['terminals'][i])  # 获取当前终止标志
        realdone_bool = bool(dataset['terminals'][i])  # 获取当前真实终止标志
        final_timestep = dataset['timeouts'][i]  # 获取当前时间终止标志

        if i < N - 1:  # 如果不是最后一个样本
            done_bool += dataset['timeouts'][i]  # 更新终止标志

        if (not terminate_on_end) and final_timestep:  # 如果不终止在episode的最后一步且当前是时间终止
            # 跳过这个转换，不在episode的最后一步应用终止
            episode_step = 0  # 重置episode步数
            continue  # 跳过当前循环
        if done_bool or final_timestep:  # 如果当前是终止或时间终止
            episode_step = 0  # 重置episode步数

        obs_.append(obs)  # 添加当前观察到列表
        next_obs_.append(new_obs)  # 添加下一个观察到列表
        action_.append(action)  # 添加当前动作到列表
        reward_.append(reward)  # 添加当前奖励到列表
        done_.append(done_bool)  # 添加当前终止标志到列表
        realdone_.append(realdone_bool)  # 添加当前真实终止标志到列表
        episode_step += 1  # 增加episode步数

    return {
        'observations': np.array(obs_),  # 返回观察数组
        'actions': np.array(action_),  # 返回动作数组
        'next_observations': np.array(next_obs_),  # 返回下一个观察数组
        'rewards': np.array(reward_)[:,None],  # 返回奖励数组，增加一个维度
        'terminals': np.array(done_)[:,None],  # 返回终止标志数组，增加一个维度
        'realterminals': np.array(realdone_)[:,None],  # 返回真实终止标志数组，增加一个维度
    }

def load_environment(name):
    """
    加载环境并返回未包装的环境对象。
    参数:
    name: 环境名称
    返回:
    未包装的环境对象
    """
    with suppress_output():  # 使用suppress_output上下文管理器
        wrapped_env = gym.make(name)  # 创建包装的环境对象
    env = wrapped_env.unwrapped  # 获取未包装的环境对象
    env.max_episode_steps = wrapped_env._max_episode_steps  # 设置环境的最大步数
    env.name = name  # 设置环境的名称
    return env  # 返回未包装的环境对象