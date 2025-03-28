import numpy as np
import torch
import pdb

#-------------------------------- 辅助函数 --------------------------------#

def top_k_logits(logits, k):
    """
    将logits中除了前k个最大值之外的所有值设为-inf。
    logits: tensor[ batch_size x vocab_size ]
    k: int
    """
    v, ix = torch.topk(logits, k)  # 获取前k个最大值及其索引
    out = logits.clone()  # 复制logits
    out[out < v[:, [-1]]] = -float('Inf')  # 将小于第k个最大值的值设为-inf
    return out

def filter_cdf(logits, threshold):
    """
    根据累积分布函数(CDF)过滤logits。
    logits: tensor[ batch_size x vocab_size ]
    threshold: float
    """
    batch_inds = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)  # 生成批次索引
    bins_inds = torch.arange(logits.shape[-1], device=logits.device)  # 生成词汇表索引
    probs = logits.softmax(dim=-1)  # 计算概率分布
    probs_sorted, _ = torch.sort(probs, dim=-1)  # 对概率进行排序
    probs_cum = torch.cumsum(probs_sorted, dim=-1)  # 计算累积概率
    ## 获取累积概率大于等于`threshold`的最小概率
    mask = probs_cum < threshold
    masked_inds = torch.argmax(mask * bins_inds, dim=-1)
    probs_threshold = probs_sorted[batch_inds, masked_inds]
    ## 过滤logits
    out = logits.clone()
    logits_mask = probs <= probs_threshold.unsqueeze(dim=-1)
    out[logits_mask] = -1000
    return out

def round_to_multiple(x, N):
    """
    将 `x` 向上舍入到最接近的 `N` 的倍数。
    x: int
    N: int
    """
    pad = (N - x % N) % N  # 计算需要填充的值
    return x + pad

def sort_2d(x):
    """
    对二维张量 `x` 进行排序，返回排序后的值及其行列索引。
    x: tensor[ M x N ]
    """
    M, N = x.shape
    x = x.view(-1)  # 将二维张量展平为一维
    x_sort, inds = torch.sort(x, descending=True)  # 对展平后的张量进行降序排序

    rows = inds // N  # 计算排序后的行索引
    cols = inds % N  # 计算排序后的列索引

    return x_sort, rows, cols

#-------------------------------- 前向传播 --------------------------------#

def forward(model, x, max_block=None, allow_crop=True, crop_increment=None, **kwargs):
    """
    对 transformer 模型进行一次前向传播的包装函数。如果序列过长，则进行裁剪。
    x: tensor[ batch_size x sequence_length ]
    """
    model.eval()  # 设置模型为评估模式 $ 禁用dropout和batch normalization等训练时特有的操作

    block_size = min(model.get_block_size(), max_block or np.inf)  # 获取块大小

    if x.shape[1] > block_size: # 检查输入序列的长度是否超过了块大小
        assert allow_crop, ( # 如果输入序列过长且不允许裁剪，则抛出断言错误
            f'[ search/sampling ] input size is {x.shape} and block size is {block_size}, '
            'but cropping not allowed')

        ## 以`crop_increment`为单位进行裁剪，确保第一个token始终为s_t^0
        n_crop = round_to_multiple(x.shape[1] - block_size, crop_increment) # 计算需要裁剪的长度，确保裁剪后的序列长度是crop_increment的倍数
        assert n_crop % crop_increment == 0 # 确保裁剪长度是crop_increment的倍数
        x = x[:, n_crop:] # 对输入序列进行裁剪，保留从n_crop开始的子序列

    logits, _ = model(x, **kwargs)  # 进行前向传播

    return logits

def get_logp(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    """
    计算给定输入序列 `x` 的 logits 并返回 log 概率。
    x: tensor[ batch_size x sequence_length ]
    """
    ## [ batch_size x sequence_length x vocab_size ]
    logits = forward(model, x, **forward_kwargs)

    ## 获取最后一个时间步的logits并按温度缩放
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## 根据CDF过滤logits
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## 根据topk过滤logits
    if topk is not None:
        logits = top_k_logits(logits, topk)

    ## 应用softmax转换为概率
    logp = logits.log_softmax(dim=-1)

    return logp

#-------------------------------- 采样 --------------------------------#

def sample(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    """
    从 `model(x)` 参数化的分布中进行采样。
    x: tensor[ batch_size x sequence_length ]
    """
    ## [ batch_size x sequence_length x vocab_size ]
    logits = forward(model, x, **forward_kwargs)

    ## 获取最后一个时间步的 logits 并按温度缩放
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## 记录修改前的概率
    raw_probs = logits.softmax(dim=-1)

    ## 根据CDF过滤logits
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## 根据topk过滤logits
    if topk is not None:
        logits = top_k_logits(logits, topk)

    ## 应用softmax转换为概率
    probs = logits.softmax(dim=-1)

    ## 从分布中采样
    ## [ batch_size x 1 ]
    indices = torch.multinomial(probs, num_samples=1)

    return indices, raw_probs

@torch.no_grad()
def sample_n(model, x, N, **sample_kwargs):
    """
    从 `model(x)` 参数化的分布中进行 `N` 次采样。
    x: tensor[ batch_size x sequence_length ]
    N: int
    """
    batch_size = len(x)

    ## 记录每一步的概率；
    ## `vocab_size + 1` 用于终止 token
    probs = torch.zeros(batch_size, N, model.vocab_size + 1, device=x.device)

    for n in range(N):
        indices, p = sample(model, x, **sample_kwargs)

        ## 将采样结果追加到序列中并继续
        ## [ batch_size x (sequence_length + n) ]
        x = torch.cat((x, indices), dim=1)

        probs[:, n] = p

    return x, probs