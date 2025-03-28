import os
import glob
import numpy as np
import json
import pdb

import trajectory.utils as utils

# 定义数据集列表，包含不同环境和缓冲区的组合
DATASETS = [
    f'{env}-{buffer}'
    for env in ['hopper', 'walker2d', 'halfcheetah', 'ant']
    for buffer in ['medium-expert-v2', 'medium-v2', 'medium-replay-v2']
]

# 定义日志文件的根目录
LOGBASE = 'logs'
# 定义试验的通配符
TRIAL = '*'
# 定义实验名称
EXP_NAME = 'plans/pretrained'

def load_results(paths):
    '''
        加载实验结果
        paths : 包含实验试验的目录路径
    '''
    scores = []
    for i, path in enumerate(sorted(paths)): # 遍历排序后的路径列表
        score = load_result(path) # 加载单个实验结果
        if score is None:
            print(f'跳过 {path}')
            continue
        scores.append(score) # 将分数添加到列表中

        suffix = path.split('/')[-1] # 获取路径的最后一部分作为后缀

    mean = np.mean(scores)  # 计算平均值
    err = np.std(scores) / np.sqrt(len(scores))  # 计算标准误差
    return mean, err, scores

def load_result(path):
    '''
        加载单个实验结果
        path : 实验目录的路径；期望目录中包含 `rollout.json` 文件
    '''
    fullpath = os.path.join(path, 'rollout.json') # 构建完整的文件路径
    suffix = path.split('/')[-1] # 获取路径的最后一部分作为后缀

    if not os.path.exists(fullpath): # 检查文件是否存在
        return None

    results = json.load(open(fullpath, 'rb'))  # 读取JSON文件
    score = results['score'] # 从JSON数据中提取分数
    return score * 100  # 将分数乘以100

#######################
######## 设置 ########
#######################

if __name__ == '__main__':

    class Parser(utils.Parser): # 定义一个解析器类，继承自utils.Parser
        dataset: str = None

    args = Parser().parse_args() # # 解析命令行参数

    # 遍历数据集
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        subdirs = glob.glob(os.path.join(LOGBASE, dataset, EXP_NAME)) # # 获取所有符合条件的子目录

        for subdir in subdirs: # 遍历子目录
            reldir = subdir.split('/')[-1] # 获取子目录的最后一部分作为相对路径
            paths = glob.glob(os.path.join(subdir, TRIAL)) # 获取子目录下的所有试验路径

            mean, err, scores = load_results(paths) # 加载并计算结果
            print(f'{dataset.ljust(30)} | {subdir.ljust(50)} | {len(scores)} 个分数 \n    {mean:.2f} +/- {err:.2f}\n')