import numpy as np  # 导入numpy库，用于数值计算
import matplotlib  # 导入matplotlib库，用于绘图
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
import pdb  # 导入pdb库，用于调试

from plotting.scores import means  # 从plotting.scores模块中导入means变量

class Colors:  # 定义一个颜色类，包含一些预定义的颜色
    grey = '#B4B4B4'
    gold = '#F6C781'
    red = '#EC7C7D'
    blue = '#70ABCC'

LABELS = {  # 定义一个标签字典，用于存储算法名称
    # 'BC': 'Behavior\nCloning',
    # 'MBOP': 'Model-Based\nOffline Planning',
    # 'BRAC': 'Behavior-Reg.\nActor-Critic',
    # 'CQL': 'Conservative\nQ-Learning',
}

def get_mean(results, exclude=None):  # 定义一个函数，用于计算平均值，并可以排除某些环境
    '''
        results : { environment: score, ... }
    '''
    filtered = {  # 过滤掉需要排除的环境
        k: v for k, v in results.items()
        if (not exclude) or (exclude and exclude not in k)
    }
    return np.mean(list(filtered.values()))  # 返回过滤后结果的平均值

if __name__ == '__main__':  # 主程序入口

    #################
    ## latex
    #################
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})  # 设置字体为Computer Modern
    matplotlib.rc('text', usetex=True)  # 启用LaTeX渲染
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]  # 加载amsmath包
    #################

    fig = plt.gcf()  # 获取当前图表
    ax = plt.gca()  # 获取当前轴
    fig.set_size_inches(7.5, 2.5)  # 设置图表大小

    means = {k: get_mean(v, exclude='ant') for k, v in means.items()}  # 计算每个算法的平均值，排除'ant'环境
    print(means)  # 打印计算结果

    algs = ['BC', 'MBOP', 'BRAC', 'CQL', 'Decision\nTransformer', 'Trajectory\nTransformer']  # 定义算法列表
    vals = [means[alg] for alg in algs]  # 获取每个算法的平均值

    colors = [  # 定义每个算法的颜色
        Colors.grey, Colors.gold,
        Colors.red, Colors.red, Colors.blue, Colors.blue
    ]

    labels = [LABELS.get(alg, alg) for alg in algs]  # 获取每个算法的标签
    plt.bar(labels, vals, color=colors, edgecolor=Colors.gold, lw=0)  # 绘制柱状图
    plt.ylabel('Average normalized return', labelpad=15)  # 设置y轴标签
    # plt.title('Offline RL Results')  # 设置图表标题（注释掉了）

    legend_labels = ['Behavior Cloning', 'Trajectory Optimization', 'Temporal Difference', 'Sequence Modeling']  # 定义图例标签
    colors = [Colors.grey, Colors.gold, Colors.red, Colors.blue]  # 定义图例颜色
    handles = [plt.Rectangle((0,0),1,1, color=color) for label, color in zip(legend_labels, colors)]  # 创建图例句柄
    plt.legend(handles, legend_labels, ncol=4,  # 添加图例
        bbox_to_anchor=(1.07, -.18), fancybox=False, framealpha=0, shadow=False, columnspacing=1.5, handlelength=1.5)

    matplotlib.rcParams['hatch.linewidth'] = 7.5  # 设置填充线的宽度
    # ax.patches[-1].set_hatch('/')  # 设置最后一个柱状图的填充样式（注释掉了）

    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
    ax.spines['top'].set_visible(False)  # 隐藏顶部边框

    # plt.savefig('plotting/bar.pdf', bbox_inches='tight')  # 保存图表为PDF格式（注释掉了）
    plt.savefig('plotting/bar.png', bbox_inches='tight', dpi=500)  # 保存图表为PNG格式