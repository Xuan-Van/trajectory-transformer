from trajectory.utils import watch

#------------------------ 基础配置 ------------------------#

logbase = 'logs/'  # 日志文件的基础路径
gpt_expname = 'gpt/azure'  # GPT 实验名称

## 自动生成实验名称，通过标记文件夹中的这些参数
args_to_watch = [
    ('prefix', ''),  # 前缀
    ('plan_freq', 'freq'),  # 规划频率
    ('horizon', 'H'),  # 规划范围
    ('beam_width', 'beam'),  # 波束宽度
]

base = {

    'train': {
        'N': 100,  # 训练样本数
        'discount': 0.99,  # 折扣因子
        'n_layer': 4,  # 网络层数
        'n_head': 4,  # 注意力头数

        ## 对于一个1M大小的数据集，训练的epoch数；n_epochs = 1M / dataset_size * n_epochs_ref
        'n_epochs_ref': 50,  # 参考epoch数
        'n_saves': 3,  # 保存模型的次数
        'logbase': logbase,  # 日志基础路径
        'device': 'cuda',  # 设备类型

        'n_embd': 32,  # 嵌入维度
        'batch_size': 256,  # 批量大小
        'learning_rate': 6e-4,  # 学习率
        'lr_decay': True,  # 是否启用学习率衰减
        'seed': 42,  # 随机种子

        'embd_pdrop': 0.1,  # 嵌入层dropout率
        'resid_pdrop': 0.1,  # 残差连接dropout率
        'attn_pdrop': 0.1,  # 注意力dropout率

        'step': 1,  # 步长
        'subsampled_sequence_length': 10,  # 子采样的序列长度
        'termination_penalty': -100,  # 终止惩罚
        'exp_name': gpt_expname,  # 实验名称

        'discretizer': 'QuantileDiscretizer',  # 离散化器类型
        'action_weight': 5,  # 动作权重
        'reward_weight': 1,  # 奖励权重
        'value_weight': 1,  # 价值权重
    },

    'plan': {
        'logbase': logbase,  # 日志基础路径
        'gpt_loadpath': gpt_expname,  # GPT模型加载路径
        'gpt_epoch': 'latest',  # GPT模型加载的epoch
        'device': 'cuda',  # 设备类型
        'renderer': 'Renderer',  # 渲染器类型

        'plan_freq': 1,  # 规划频率
        'horizon': 15,  # 规划范围
        'beam_width': 128,  # 波束宽度
        'n_expand': 2,  # 扩展次数

        'k_obs': 1,  # 观察值的k值
        'k_act': None,  # 动作的k值
        'cdf_obs': None,  # 观察值的CDF值
        'cdf_act': 0.6,  # 动作的CDF值
        'percentile': 'mean',  # 百分位数

        'max_context_transitions': 5,  # 最大上下文转换次数
        'prefix_context': True,  # 是否使用前缀上下文

        'vis_freq': 50,  # 可视化频率
        'exp_name': watch(args_to_watch),  # 实验名称
        'prefix': 'plans/defaults/',  # 前缀
        'suffix': '0',  # 后缀
        'verbose': True,  # 是否打印详细信息
    },

}

#------------------------ 运动控制配置 ------------------------#

## 对于所有halfcheetah环境，可以减少规划范围和波束宽度，而不影响性能。
## 这样可以提高速度和减少计算量。

halfcheetah_medium_v2 = halfcheetah_medium_replay_v2 = {
    'plan': {
        'horizon': 5,  # 规划范围
        'beam_width': 32,  # 波束宽度
    }
}

halfcheetah_medium_expert_v2 = {
    'plan': {
        'beam_width': 32,  # 波束宽度
    },
}

## 如果字典为空，则使用基础参数
hopper_medium_expert_v2 = hopper_medium_v2 = walker2d_medium_v2 = {}

## hopper和walker2d对规划超参数更敏感；
## 在减少规划范围或增加规划频率时要小心

hopper_medium_replay_v2 = {
    'train': {
        ## 在medium-replay数据集上训练更长时间
        'n_epochs_ref': 80,  # 参考epoch数
    },
}

walker2d_medium_expert_v2 = {
    'plan': {
        ## 这里也可以安全地减少规划范围
        'horizon': 5,  # 规划范围
    },
}

walker2d_medium_replay_v2 = {
    'train': {
        ## 在medium-replay数据集上训练更长时间
        'n_epochs_ref': 80,  # 参考epoch数
    },
    'plan': {
        ## 可以减少波束宽度，但需要调整动作采样分布并增加规划范围
        'horizon': 20,  # 规划范围
        'beam_width': 32,  # 波束宽度
        'k_act': 40,  # 动作的k值
        'cdf_act': None,  # 动作的CDF值
    }
}

ant_medium_v2 = ant_medium_replay_v2 = ant_random_v2 = {
    'train': {
        ## 减少批量大小，因为维度更大
        'batch_size': 128,  # 批量大小
    },
    'plan': {
        'horizon': 5,  # 规划范围
    }
}