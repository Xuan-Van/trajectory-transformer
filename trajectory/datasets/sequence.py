import os  # 导入操作系统相关的模块
import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库，用于张量操作和深度学习
import pdb  # 导入Python调试器，用于调试

from trajectory.utils import discretization  # 从trajectory.utils模块导入discretization工具
from trajectory.utils.arrays import to_torch  # 从trajectory.utils.arrays模块导入to_torch函数

from .d4rl import load_environment, qlearning_dataset_with_timeouts  # 从d4rl模块导入load_environment和qlearning_dataset_with_timeouts函数
from .preprocessing import dataset_preprocess_functions  # 从preprocessing模块导入dataset_preprocess_functions函数

def segment(observations, terminals, max_path_length):
    """
        根据`terminals`将`observations`分割成轨迹。
        参数:
            observations (np.ndarray): 观测值数组。
            terminals (np.ndarray): 终端标志数组。
            max_path_length (int): 最大轨迹长度。
        返回:
            trajectories_pad (np.ndarray): 填充后的轨迹数组。
            early_termination (np.ndarray): 提前终止标志数组。
            path_lengths (list): 每个轨迹的长度列表。
    """
    assert len(observations) == len(terminals)  # 确保观测值和终端标志的长度一致
    observation_dim = observations.shape[1]  # 获取观测值的维度

    trajectories = [[]]  # 初始化轨迹列表
    for obs, term in zip(observations, terminals):  # 遍历观测值和终端标志
        trajectories[-1].append(obs)  # 将观测值添加到当前轨迹
        if term.squeeze():  # 如果遇到终端标志
            trajectories.append([])  # 开始新的轨迹

    if len(trajectories[-1]) == 0:  # 如果最后一个轨迹为空
        trajectories = trajectories[:-1]  # 移除最后一个空轨迹

    ## 将轨迹列表转换为数组列表，因为轨迹长度不同
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)  # 获取轨迹数量
    path_lengths = [len(traj) for traj in trajectories]  # 获取每个轨迹的长度

    ## 填充轨迹使其长度相等
    trajectories_pad = np.zeros((n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype)
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=np.bool) # 存储提前终止标志
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i] # 获取当前轨迹的长度
        trajectories_pad[i, :path_length] = traj # 将当前轨迹填充到trajectories_pad中，直到轨迹的实际长度
        early_termination[i, path_length:] = 1 # 将early_termination中从轨迹实际长度之后的部分设置为1，表示提前终止

    return trajectories_pad, early_termination, path_lengths


class SequenceDataset(torch.utils.data.Dataset):
    """
        序列数据集类，用于处理和加载序列数据。
    """

    def __init__(self, env, sequence_length=250, step=10, discount=0.99, max_path_length=1000, penalty=None,
                 device='cuda:0'):
        """
            初始化序列数据集。
            参数:
                env (str or object): 环境名称或环境对象。
                sequence_length (int): 序列长度。
                step (int): 步长。
                discount (float): 折扣因子。
                max_path_length (int): 最大轨迹长度。
                penalty (float): 终端惩罚。
                device (str): 设备名称，默认为'cuda:0'。
        """
        print(f'[ datasets/sequence ] 序列长度: {sequence_length} | 步长: {step} | 最大轨迹长度: {max_path_length}')
        self.env = env = load_environment(env) if type(env) is str else env  # 加载环境
        self.sequence_length = sequence_length  # 设置序列长度
        self.step = step  # 设置步长
        self.max_path_length = max_path_length  # 设置最大轨迹长度
        self.device = device  # 设置设备

        print(f'[ datasets/sequence ] 加载中...', end=' ', flush=True)
        dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=True)  # 加载Q-learning数据集
        print('✓')

        preprocess_fn = dataset_preprocess_functions.get(env.name)  # 获取预处理函数
        if preprocess_fn:
            print(f'[ datasets/sequence ] 修改环境')
            dataset = preprocess_fn(dataset)  # 对数据集进行预处理

        observations = dataset['observations']  # 获取观测值
        actions = dataset['actions']  # 获取动作
        next_observations = dataset['next_observations']  # 获取下一个观测值
        rewards = dataset['rewards']  # 获取奖励
        terminals = dataset['terminals']  # 获取终端标志
        realterminals = dataset['realterminals']  # 获取真实终端标志

        self.observations_raw = observations  # 保存原始观测值
        self.actions_raw = actions  # 保存原始动作
        self.next_observations_raw = next_observations  # 保存原始下一个观测值
        self.joined_raw = np.concatenate([observations, actions], axis=-1)  # 将观测值和动作拼接
        self.rewards_raw = rewards  # 保存原始奖励
        self.terminals_raw = terminals  # 保存原始终端标志

        ## 终端惩罚
        if penalty is not None:
            terminal_mask = realterminals.squeeze() # 获取真实终端标志的掩码
            self.rewards_raw[terminal_mask] = penalty # 将终端惩罚应用到奖励中

        ## 分割轨迹
        print(f'[ datasets/sequence ] 分割中...', end=' ', flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(self.joined_raw, terminals,
                                                                                   max_path_length) # 调用segment函数分割轨迹，并保存结果
        self.rewards_segmented, *_ = segment(self.rewards_raw, terminals, max_path_length) # 调用segment函数分割奖励，并保存结果
        print('✓')

        self.discount = discount  # 设置折扣因子
        self.discounts = (discount ** np.arange(self.max_path_length))[:, None]  # 计算折扣因子序列

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape) # 初始化值数组

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:, t + 1:] * self.discounts[:-t - 1]).sum(axis=1) # 计算值函数
            self.values_segmented[:, t] = V # 将值函数保存到值数组中

        ## 将(r, V)添加到`joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1) # 将值数组展平
        values_mask = ~self.termination_flags.reshape(-1) # 获取值掩码
        self.values_raw = values_raw[values_mask, None] # 保存原始值
        self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1) # 将奖励和值添加到拼接数组中
        self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented],
                                               axis=-1) # 将奖励和值添加到分割后的拼接数组中

        ## 获取有效索引
        indices = []
        for path_ind, length in enumerate(self.path_lengths): # 遍历每个轨迹及其长度
            end = length - 1 # 计算轨迹的结束位置
            for i in range(end):
                indices.append((path_ind, i, i + sequence_length)) # 将索引添加到索引列表中

        self.indices = np.array(indices) # 将索引列表转换为NumPy数组
        self.observation_dim = observations.shape[1] # 获取观测值的维度
        self.action_dim = actions.shape[1] # 获取动作的维度
        self.joined_dim = self.joined_raw.shape[1] # 获取拼接数组的维度

        ## 填充轨迹
        n_trajectories, _, joined_dim = self.joined_segmented.shape # 获取轨迹数量和拼接数组的维度
        self.joined_segmented = np.concatenate([ # 填充轨迹
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length - 1, joined_dim)),
        ], axis=1)
        self.termination_flags = np.concatenate([ # 填充提前终止标志
            self.termination_flags,
            np.ones((n_trajectories, sequence_length - 1), dtype=np.bool),
        ], axis=1)

    def __len__(self):
        """
            返回数据集的长度。
            返回:
                int: 数据集的长度。
        """
        return len(self.indices)


class DiscretizedDataset(SequenceDataset):
    """
        离散化数据集类，继承自SequenceDataset。
    """

    def __init__(self, *args, N=50, discretizer='QuantileDiscretizer', **kwargs):
        """
            初始化离散化数据集。
            参数:
                *args: 传递给父类的参数。
                N (int): 离散化桶的数量。
                discretizer (str): 离散化器的名称。
                **kwargs: 传递给父类的关键字参数。
        """
        super().__init__(*args, **kwargs)
        self.N = N  # 设置离散化桶的数量
        discretizer_class = getattr(discretization, discretizer)  # 获取离散化器类
        self.discretizer = discretizer_class(self.joined_raw, N)  # 初始化离散化器

    def __getitem__(self, idx):
        """
            获取指定索引的样本。
            参数:
                idx (int): 样本索引。
            返回:
                tuple: 包含输入X、目标Y和掩码的元组。
        """
        path_ind, start_ind, end_ind = self.indices[idx] # 获取指定索引的轨迹索引、起始索引和结束索引
        path_length = self.path_lengths[path_ind] # 获取轨迹的长度

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step] # 获取指定轨迹的拼接数据
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step] # 获取指定轨迹的终止标志

        joined_discrete = self.discretizer.discretize(joined) # 对拼接数据进行离散化

        ## 如果序列已结束，替换为终止标记
        assert (joined[terminations] == 0).all(), \
            f'Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}' # 断言，确保终止标志后的数据为零
        joined_discrete[terminations] = self.N # 将终止标志后的数据替换为终止标记

        ## [ (sequence_length / skip) x observation_dim]
        joined_discrete = to_torch(joined_discrete, device='cpu', dtype=torch.long).contiguous() # 将离散化后的数据转换为PyTorch张量

        ## 不要计算超出最大轨迹长度的预测部分的损失
        traj_inds = torch.arange(start_ind, end_ind, self.step) # 生成轨迹索引
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool) # 初始化掩码
        mask[traj_inds > self.max_path_length - self.step] = 0 # 将超出最大轨迹长度的部分掩码设置为零

        ## 展平所有内容
        joined_discrete = joined_discrete.view(-1) # 展平离散化后的数据
        mask = mask.view(-1) # 展平掩码

        X = joined_discrete[:-1] # 获取输入数据
        Y = joined_discrete[1:] # 获取目标数据
        mask = mask[:-1] # 获取掩码

        return X, Y, mask


class GoalDataset(DiscretizedDataset):
    """
        目标数据集类，继承自DiscretizedDataset。
    """

    def __init__(self, *args, **kwargs):
        """
            初始化目标数据集。
            参数:
                *args: 传递给父类的参数。
                **kwargs: 传递给父类的关键字参数。
        """
        super().__init__(*args, **kwargs)
        pdb.set_trace()  # 设置断点，用于调试

    def __getitem__(self, idx):
        """
            获取指定索引的样本。
            参数:
                idx (int): 样本索引。
            返回:
                tuple: 包含输入X、目标Y、掩码和目标离散值的元组。
        """
        X, Y, mask = super().__getitem__(idx) # 调用父类的__getitem__方法，获取输入数据、目标数据和掩码

        ## 获取轨迹长度以查找轨迹中的最后一个转换
        path_ind, start_ind, end_ind = self.indices[idx] # 获取指定索引的轨迹索引、起始索引和结束索引
        path_length = self.path_lengths[path_ind] # 获取轨迹的长度

        ## 目标是最后一个转换的前`observation_dim`维
        goal = self.joined_segmented[path_ind, path_length - 1, :self.observation_dim] # 获取轨迹中最后一个转换的目标
        goal_discrete = self.discretizer.discretize(goal, subslice=(0, self.observation_dim)) # 对目标进行离散化
        goal_discrete = to_torch(goal_discrete, device='cpu', dtype=torch.long).contiguous().view(-1) # 将离散化后的目标转换为PyTorch张量

        return X, goal_discrete, Y, mask # 返回输入数据、目标离散值、目标数据和掩码