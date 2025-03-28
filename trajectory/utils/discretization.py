import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习
import pdb  # 导入pdb库，用于调试 $ 例如插入pdb.set_trace()，程序运行到该行时会自动进入调试模式

from .arrays import to_np, to_torch  # 从当前目录的arrays模块中导入to_np和to_torch函数

class QuantileDiscretizer:
    """
    分位数离散化器类，用于将连续数据离散化为分位数区间
    """

    def __init__(self, data, N):
        """
        初始化分位数离散化器
        :param data: 输入数据，形状为 [数据点数 x 特征维度]
        :param N: 离散化的区间数
        """
        self.data = data  # 存储输入数据
        self.N = N  # 存储离散化的区间数

        n_points_per_bin = int(np.ceil(len(data) / N))  # 计算每个区间内的数据点数 $ np.ceil：向上取整
        obs_sorted = np.sort(data, axis=0)  # 按列排序数据
        thresholds = obs_sorted[::n_points_per_bin, :]  # 计算每个区间的阈值 $ `::n_points_per_bin`表示每隔n_points_per_bin行取一行,`:`表示取所有列
        maxs = data.max(axis=0, keepdims=True)  # 计算数据的最大值 $ keepdims=True：保持输出数组的维度与输入数组相同

        ## [ (N + 1) x 特征维度 ]
        self.thresholds = np.concatenate([thresholds, maxs], axis=0)  # 将阈值和最大值拼接起来

        ## [ N x 特征维度 ]
        self.diffs = self.thresholds[1:] - self.thresholds[:-1]  # 计算相邻阈值之间的差值

        self._test()  # 测试离散化器的正确性

    def __call__(self, x):
        """
        对输入数据进行离散化和重构，并计算重构误差
        :param x: 输入数据，形状为 [B x 特征维度]
        :return: 离散化后的索引、重构数据和重构误差
        """
        indices = self.discretize(x)  # 对输入数据进行离散化
        recon = self.reconstruct(indices)  # 对离散化后的索引进行重构
        error = np.abs(recon - x).max(0)  # 计算重构误差 $ max(0)：max(axis=0)
        return indices, recon, error  # 返回离散化后的索引、重构数据和重构误差

    def _test(self): # $ _表示是私有函数，不应该被外部代码直接访问
        """
        测试离散化器的正确性
        """
        print('[ utils/discretization ] Testing...', end=' ', flush=True)  # 打印测试信息 $ flush：是否在输出后立即刷新输出缓冲区
        inds = np.random.randint(0, len(self.data), size=1000)  # 随机选择1000个数据点
        X = self.data[inds]  # 获取随机选择的1000个数据点
        indices = self.discretize(X)  # 对随机选择的1000个数据点进行离散化
        recon = self.reconstruct(indices)  # 对离散化后的索引进行重构
        ## 确保重构误差小于每个维度的最大允许误差
        error = np.abs(X - recon).max(0)  # 计算重构误差
        assert (error <= self.diffs.max(axis=0)).all()  # 断言重构误差小于每个维度的最大允许误差 $ 如果断言条件为假，程序会抛出一个AssertionError异常，并终止执行
        ## 重新离散化重构数据，并确保其与原始索引相同
        indices_2 = self.discretize(recon)  # 对重构数据进行离散化
        assert (indices == indices_2).all()  # 断言重新离散化后的索引与原始索引相同
        print('✓')  # 打印测试通过信息

    def discretize(self, x, subslice=(None, None)):
        """
        对输入数据进行离散化
        :param x: 输入数据，形状为 [B x 特征维度]
        :param subslice: 子切片，用于选择特征维度
        :return: 离散化后的索引
        """
        if torch.is_tensor(x):  # 如果输入数据是PyTorch张量
            x = to_np(x)  # 将其转换为NumPy数组

        ## 强制批处理模式
        if x.ndim == 1:  # 如果输入数据的维度为1
            x = x[None]  # 将其扩展为二维数组

        ## [ N x B x 特征维度 ]
        start, end = subslice  # 获取子切片的起始和结束位置
        thresholds = self.thresholds[:, start:end]  # 选择子切片对应的阈值

        gt = x[None] >= thresholds[:,None]  # 计算输入数据是否大于等于阈值
        indices = largest_nonzero_index(gt, dim=0)  # 计算最大非零索引

        if indices.min() < 0 or indices.max() >= self.N:  # 如果索引超出范围
            indices = np.clip(indices, 0, self.N - 1)  # 将其裁剪到有效范围内 $ np.clip：如果元素小于最小值，则将其替换为最小值；如果元素大于最大值，则将其替换为最大值

        return indices  # 返回离散化后的索引

    def reconstruct(self, indices, subslice=(None, None)):
        """
        对离散化后的索引进行重构
        :param indices: 离散化后的索引，形状为 [B x 特征维度]
        :param subslice: 子切片，用于选择特征维度
        :return: 重构后的数据
        """
        if torch.is_tensor(indices):  # 如果输入索引是PyTorch张量
            indices = to_np(indices)  # 将其转换为NumPy数组

        ## 强制批处理模式
        if indices.ndim == 1:  # 如果输入索引的维度为1
            indices = indices[None]  # 将其扩展为二维数组

        if indices.min() < 0 or indices.max() >= self.N:  # 如果索引超出范围
            print(f'[ utils/discretization ] indices out of range: ({indices.min()}, {indices.max()}) | N: {self.N}')  # 打印警告信息
            indices = np.clip(indices, 0, self.N - 1)  # 将其裁剪到有效范围内

        start, end = subslice  # 获取子切片的起始和结束位置
        thresholds = self.thresholds[:, start:end]  # 选择子切片对应的阈值

        left = np.take_along_axis(thresholds, indices, axis=0)  # 获取左边界阈值 $ np.take_along_axis：沿指定轴从数组中提取元素
        right = np.take_along_axis(thresholds, indices + 1, axis=0)  # 获取右边界阈值
        recon = (left + right) / 2.  # 计算重构数据
        return recon  # 返回重构后的数据

    #---------------------------- wrappers for planning ----------------------------# $ 用于planning的包装器

    def expectation(self, probs, subslice):
        """
        计算期望值
        :param probs: 概率分布，形状为 [B x N]
        :param subslice: 子切片，用于选择特征维度
        :return: 期望值
        """
        if torch.is_tensor(probs):  # 如果输入概率是PyTorch张量
            probs = to_np(probs)  # 将其转换为NumPy数组

        ## [ N ]
        thresholds = self.thresholds[:, subslice]  # 选择子切片对应的阈值
        ## [ B ] $ 将一个概率数组probs与一个阈值数组thresholds进行矩阵乘法（点积）
        left  = probs @ thresholds[:-1]  # 计算左边界期望值
        right = probs @ thresholds[1:]  # 计算右边界期望值

        avg = (left + right) / 2.  # 计算期望值
        return avg  # 返回期望值

    def percentile(self, probs, percentile, subslice):
        """
        计算百分位数
        :param probs: 概率分布，形状为 [B x N]
        :param percentile: 百分位数
        :param subslice: 子切片，用于选择特征维度
        :return: 百分位数
        """
        ## [ N ]
        thresholds = self.thresholds[:, subslice]  # 选择子切片对应的阈值
        ## [ B x N ]
        cumulative = np.cumsum(probs, axis=-1)  # 计算累积概率分布 $ np.cumsum：计算数组的累积和
        valid = cumulative > percentile  # 计算有效概率分布
        ## [ B ]
        inds = np.argmax(np.arange(self.N, 0, -1) * valid, axis=-1)  # 计算最大索引
        left = thresholds[inds-1]  # 获取左边界阈值
        right = thresholds[inds]  # 获取右边界阈值
        avg = (left + right) / 2.  # 计算百分位数
        return avg  # 返回百分位数

    #---------------------------- wrappers for planning ----------------------------#

    def value_expectation(self, probs):
        """
        计算价值期望值
        :param probs: 概率分布，形状为 [B x 2 x (N + 1)]
        :return: 奖励期望值和下一状态价值期望值
        """
        if torch.is_tensor(probs):  # 如果输入概率是PyTorch张量
            probs = to_np(probs)  # 将其转换为NumPy数组
            return_torch = True  # 设置返回PyTorch张量标志
        else:
            return_torch = False  # 设置返回NumPy数组标志

        probs = probs[:, :, :-1]  # 去除最后一个概率
        assert probs.shape[-1] == self.N  # 断言概率分布的维度正确

        rewards = self.expectation(probs[:, 0], subslice=-2)  # 计算奖励期望值
        next_values = self.expectation(probs[:, 1], subslice=-1)  # 计算下一状态价值期望值

        if return_torch:  # 如果需要返回PyTorch张量
            rewards = to_torch(rewards)  # 将奖励期望值转换为PyTorch张量
            next_values = to_torch(next_values)  # 将下一状态价值期望值转换为PyTorch张量

        return rewards, next_values  # 返回奖励期望值和下一状态价值期望值

    def value_fn(self, probs, percentile):
        """
        计算价值函数
        :param probs: 概率分布，形状为 [B x 2 x (N + 1)]
        :param percentile: 百分位数
        :return: 奖励百分位数和下一状态价值百分位数
        """
        if percentile == 'mean':  # 如果百分位数为'mean'
            return self.value_expectation(probs)  # 返回价值期望值
        else:
            ## 百分位数应可解释为浮点数，即使通过命令行解析器传递为字符串
            percentile = float(percentile)  # 将百分位数转换为浮点数

        if torch.is_tensor(probs):  # 如果输入概率是PyTorch张量
            probs = to_np(probs)  # 将其转换为NumPy数组
            return_torch = True  # 设置返回PyTorch张量标志
        else:
            return_torch = False  # 设置返回NumPy数组标志

        probs = probs[:, :, :-1]  # 去除最后一个概率
        assert probs.shape[-1] == self.N  # 断言概率分布的维度正确

        rewards = self.percentile(probs[:, 0], percentile, subslice=-2)  # 计算奖励百分位数
        next_values = self.percentile(probs[:, 1], percentile, subslice=-1)  # 计算下一状态价值百分位数

        if return_torch:  # 如果需要返回PyTorch张量
            rewards = to_torch(rewards)  # 将奖励百分位数转换为PyTorch张量
            next_values = to_torch(next_values)  # 将下一状态价值百分位数转换为PyTorch张量

        return rewards, next_values  # 返回奖励百分位数和下一状态价值百分位数

def largest_nonzero_index(x, dim):
    """
    计算最大非零索引
    :param x: 输入数组
    :param dim: 维度
    :return: 最大非零索引
    """
    N = x.shape[dim]  # 获取维度大小
    arange = np.arange(N) + 1  # 创建范围数组

    for i in range(dim):  # 扩展范围数组的维度
        arange = np.expand_dims(arange, axis=0) # $ np.expand_dims：在指定的轴上扩展数组的维度
    for i in range(dim+1, x.ndim):
        arange = np.expand_dims(arange, axis=-1)

    inds = np.argmax(x * arange, axis=0)  # 计算最大非零索引
    ## 处理所有`False`或所有`True`的情况
    lt_mask = (~x).all(axis=0)  # 所有元素为`False`的掩码
    gt_mask = (x).all(axis=0)  # 所有元素为`True`的掩码

    inds[lt_mask] = 0  # 将所有`False`的索引设置为0
    inds[gt_mask] = N  # 将所有`True`的索引设置为N

    return inds  # 返回最大非零索引