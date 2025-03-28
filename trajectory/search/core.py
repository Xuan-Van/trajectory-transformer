import numpy as np
import torch
import pdb

from .. import utils
from .sampling import sample_n, get_logp, sort_2d

# 定义奖励和价值的维度
REWARD_DIM = VALUE_DIM = 1

@torch.no_grad() # 在函数执行期间禁用梯度计算 $ 通常在推理阶段使用这个装饰器，因为在推理阶段不需要进行反向传播和梯度更新
def beam_plan(
    model, value_fn, x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
    verbose=True, previous_actions=None,
):
    """
    用于规划的函数，它通过在每个时间步进行采样和评估来生成最佳序列
    输入：
        model: 用于采样的模型。
        value_fn: 用于计算奖励和价值的函数。
        x: 输入序列，形状为 [1 x input_sequence_length]。
        n_steps: 规划的步数。
        beam_width: 波束宽度，即每次保留的最佳序列数量。
        n_expand: 每次扩展的次数。
        observation_dim: 观察的维度。
        action_dim: 动作的维度。
        discount: 折扣因子，默认为0.99。
        max_context_transitions: 最大上下文转换次数。
        k_obs, k_act, k_rew: 采样时的top-k值。
        cdf_obs, cdf_act, cdf_rew: 采样时的CDF值。
        verbose: 是否打印日志，默认为True。
        previous_actions: 之前的动作序列
    """

    inp = x.clone()  # 复制输入数据

    # 将最大转换次数转换为最大token数
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## 将最大token数传递给采样函数
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    ## 重复输入数据以进行搜索
    x = x.repeat(beam_width, 1)

    ## 构造奖励和折扣张量以估计价值
    rewards = torch.zeros(beam_width, n_steps + 1, device=x.device)
    discounts = discount ** torch.arange(n_steps + 1, device=x.device)

    ## 日志记录
    progress = utils.Progress(n_steps) if verbose else utils.Silent()

    for t in range(n_steps):
        ## 在采样动作之前将所有内容重复 `n_expand` 次
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## 采样动作
        x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        ## 采样奖励和价值估计
        x, r_probs = sample_n(model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)

        ## 可选地，使用奖励和价值分布的百分位数或均值而不是采样的token
        r_t, V_t = value_fn(r_probs)

        ## 更新奖励张量
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        ## 使用直到 `t` 的奖励和 `t` 的终端价值估计价值
        values = (rewards * discounts).sum(dim=-1)

        ## 获取 `beam_width` 个最佳动作
        values, inds = torch.topk(values, beam_width)

        ## 索引搜索候选者以保留 `beam_width` 个最高奖励序列
        x = x[inds]
        rewards = rewards[inds]

        ## 采样下一个观察（除非我们已经到达规划时间线的末尾）
        if t < n_steps - 1:
            x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        ## 日志记录
        progress.update({
            'x': list(x.shape),
            'vmin': values.min(), 'vmax': values.max(),
            'vtmin': V_t.min(), 'vtmax': V_t.max(),
            'discount': discount
        })

    progress.stamp() # 结束日志记录

    ## [ batch_size x (n_context + n_steps) x transition_dim ]
    x = x.view(beam_width, -1, transition_dim) # 调整张量形状

    ## 裁剪出上下文转换
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]

    ## 返回最佳序列
    argmax = values.argmax()
    best_sequence = x[argmax]

    return best_sequence

@torch.no_grad()
def beam_search(model, x, n_steps, beam_width=512, goal=None, **sample_kwargs):
    """
    用于波束搜索的函数，它通过在每个时间步扩展和排序候选序列来生成最佳序列
    输入：
    model: 用于计算对数概率的模型。
    x: 输入序列。
    n_steps: 搜索的步数。
    beam_width: 波束宽度，即每次保留的最佳序列数量，默认为512。
    goal: 目标序列，可选。
    sample_kwargs: 采样函数的额外参数。
    """
    batch_size = len(x) # 输入序列的批量大小

    prefix_i = torch.arange(len(x), dtype=torch.long, device=x.device) # 用于记录前缀索引的张量
    cumulative_logp = torch.zeros(batch_size, 1, device=x.device) # 累积对数概率的张量，初始化为零

    for t in range(n_steps):
        # 计算对数概率
        if goal is not None:
            goal_rep = goal.repeat(len(x), 1)
            logp = get_logp(model, x, goal=goal_rep, **sample_kwargs)
        else:
            logp = get_logp(model, x, **sample_kwargs)
        # 计算候选对数概率
        candidate_logp = cumulative_logp + logp
        sorted_logp, sorted_i, sorted_j = sort_2d(candidate_logp) # 对候选对数概率进行排序，得到排序后的对数概率、索引和值
        # 选择保留的候选序列
        n_candidates = (candidate_logp > -np.inf).sum().item() # 计算有效的候选数量
        n_retain = min(n_candidates, beam_width) # 选择保留的候选数量，即beam_width和有效候选数量的最小值
        cumulative_logp = sorted_logp[:n_retain].unsqueeze(-1) # 更新累积对数概率
        # 更新输入序列和前缀索引
        sorted_i = sorted_i[:n_retain]
        sorted_j = sorted_j[:n_retain].unsqueeze(-1)

        x = torch.cat([x[sorted_i], sorted_j], dim=-1) # 更新排序后的索引和值
        prefix_i = prefix_i[sorted_i] # 更新输入序列x和前缀索引prefix_i
    # 返回结果
    x = x[0] # 选择最佳序列
    return x, cumulative_logp.squeeze() # 返回最佳序列和累积对数概率