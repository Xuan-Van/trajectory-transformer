import numpy as np  # 导入numpy库，用于数值计算
import pdb  # 导入pdb库，用于调试

from plotting.plot import get_mean  # 从plotting.plot模块导入get_mean函数
from plotting.scores import (  # 从plotting.scores模块导入MEANS和ERRORS
    means as MEANS,
    errors as ERRORS,
)

ALGORITHM_STRINGS = {  # 定义算法名称的字符串映射
    'Trajectory\nTransformer': 'TT (Ours)',
    'Decision\nTransformer': 'DT',
}

BUFFER_STRINGS = {  # 定义缓冲区名称的字符串映射
    'medium-expert': 'Medium-Expert',
    'medium': 'Medium',
    'medium-replay': 'Medium-Replay',
}

ENVIRONMENT_STRINGS = {  # 定义环境名称的字符串映射
    'halfcheetah': 'HalfCheetah',
    'hopper': 'Hopper',
    'walker2d': 'Walker2d',
    'ant': 'Ant',
}

SHOW_ERRORS = ['Trajectory\nTransformer']  # 定义需要显示误差的算法列表

def get_result(algorithm, buffer, environment, version='v2'):
    """
    获取特定算法、缓冲区和环境的结果
    :param algorithm: 算法名称
    :param buffer: 缓冲区名称
    :param environment: 环境名称
    :param version: 版本号，默认为'v2'
    :return: 结果（均值和标准差或仅均值）
    """
    key = f'{environment}-{buffer}-{version}'  # 生成结果的键
    mean = MEANS[algorithm].get(key, '-')  # 获取均值，如果不存在则返回'-'
    if algorithm in SHOW_ERRORS:  # 如果算法在需要显示误差的列表中
        error = ERRORS[algorithm].get(key)  # 获取标准差
        return (mean, error)  # 返回均值和标准差
    else:
        return mean  # 否则仅返回均值

def format_result(result):
    """
    格式化结果，如果是元组则格式化为均值±标准差，否则仅格式化为均值
    :param result: 结果（均值和标准差或仅均值）
    :return: 格式化后的结果字符串
    """
    if type(result) == tuple:  # 如果结果是元组
        mean, std = result  # 解包均值和标准差
        return f'${mean}$ \\scriptsize{{\\raisebox{{1pt}}{{$\\pm {std}$}}}}'  # 格式化为均值±标准差
    else:
        return f'${result}$'  # 否则仅格式化为均值

def format_row(buffer, environment, results):
    """
    格式化一行数据，包括缓冲区、环境和结果
    :param buffer: 缓冲区名称
    :param environment: 环境名称
    :param results: 结果列表
    :return: 格式化后的行字符串
    """
    buffer_str = BUFFER_STRINGS[buffer]  # 获取缓冲区的显示字符串
    environment_str = ENVIRONMENT_STRINGS[environment]  # 获取环境的显示字符串
    results_str = ' & '.join(format_result(result) for result in results)  # 格式化结果列表
    row = f'{buffer_str} & {environment_str} & {results_str} \\\\ \n'  # 生成行字符串
    return row

def format_buffer_block(algorithms, buffer, environments):
    """
    格式化一个缓冲区的数据块
    :param algorithms: 算法列表
    :param buffer: 缓冲区名称
    :param environments: 环境列表
    :return: 格式化后的缓冲区块字符串
    """
    block_str = '\\midrule\n'  # 添加分隔线
    for environment in environments:  # 遍历环境列表
        results = [get_result(alg, buffer, environment) for alg in algorithms]  # 获取每个算法的结果
        row_str = format_row(buffer, environment, results)  # 格式化行
        block_str += row_str  # 添加到块字符串中
    return block_str

def format_algorithm(algorithm):
    """
    格式化算法名称
    :param algorithm: 算法名称
    :return: 格式化后的算法名称字符串
    """
    algorithm_str = ALGORITHM_STRINGS.get(algorithm, algorithm)  # 获取算法的显示字符串
    return f'\multicolumn{{1}}{{c}}{{\\bf {algorithm_str}}}'  # 格式化为表格列

def format_algorithms(algorithms):
    """
    格式化算法列表
    :param algorithms: 算法列表
    :return: 格式化后的算法列表字符串
    """
    return ' & '.join(format_algorithm(algorithm) for algorithm in algorithms)  # 格式化算法列表

def format_averages(means, label):
    """
    格式化平均值
    :param means: 平均值列表
    :param label: 标签
    :return: 格式化后的平均值字符串
    """
    prefix = f'\\multicolumn{{2}}{{c}}{{\\bf Average ({label})}} & '  # 添加前缀
    formatted = ' & '.join(str(mean) for mean in means)  # 格式化平均值列表
    return prefix + formatted

def format_averages_block(algorithms):
    """
    格式化平均值块
    :param algorithms: 算法列表
    :return: 格式化后的平均值块字符串
    """
    means_filtered = [np.round(get_mean(MEANS[algorithm], exclude='ant'), 1) for algorithm in algorithms]  # 计算排除Ant的平均值
    means_all = [np.round(get_mean(MEANS[algorithm], exclude=None), 1) for algorithm in algorithms]  # 计算所有环境的平均值

    means_all = [
        means
        if 'ant-medium-expert-v2' in MEANS[algorithm]
        else '$-$'
        for algorithm, means in zip(algorithms, means_all)
    ]  # 如果某个算法没有Ant数据，则标记为'-'

    formatted_filtered = format_averages(means_filtered, 'without Ant')  # 格式化排除Ant的平均值
    formatted_all = format_averages(means_all, 'all settings')  # 格式化所有环境的平均值

    formatted_block = (
        f'{formatted_filtered} \\hspace{{.6cm}} \\\\ \n'
        f'{formatted_all} \\hspace{{.6cm}} \\\\ \n'
    )  # 生成平均值块字符串
    return formatted_block

def format_table(algorithms, buffers, environments):
    """
    格式化整个表格
    :param algorithms: 算法列表
    :param buffers: 缓冲区列表
    :param environments: 环境列表
    :return: 格式化后的表格字符串
    """
    justify_str = 'll' + 'r' * len(algorithms)  # 生成表格对齐字符串
    algorithm_str = format_algorithms(['Dataset', 'Environment'] + algorithms)  # 格式化算法列表
    averages_str = format_averages_block(algorithms)  # 格式化平均值块
    table_prefix = (
        '\\begin{table*}[h]\n'
        '\\centering\n'
        '\\small\n'
        f'\\begin{{tabular}}{{{justify_str}}}\n'
        '\\toprule\n'
        f'{algorithm_str} \\\\ \n'
    )  # 生成表格前缀
    table_suffix = (
        '\\midrule\n'
        f'{averages_str}'
        '\\bottomrule\n'
        '\\end{tabular}\n'
        '\\label{table:d4rl}\n'
        '\\end{table*}'
    )  # 生成表格后缀
    blocks = ''.join(format_buffer_block(algorithms, buffer, environments) for buffer in buffers)  # 生成缓冲区块
    table = (
        f'{table_prefix}'
        f'{blocks}'
        f'{table_suffix}'
    )  # 生成整个表格
    return table

# 定义算法、缓冲区和环境的列表
algorithms = ['BC', 'MBOP', 'BRAC', 'CQL', 'Decision\nTransformer', 'Trajectory\nTransformer']
buffers = ['medium-expert', 'medium', 'medium-replay']
environments = ['halfcheetah', 'hopper', 'walker2d', 'ant']

# 生成并打印表格
table = format_table(algorithms, buffers, environments)
print(table)