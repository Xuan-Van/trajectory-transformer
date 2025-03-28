import os
import pdb

from doodad.wrappers.easy_launch import sweep_function, save_doodad_config

# 代码路径
codepath = '/home/code'
# 脚本路径
script = 'scripts/plan.py'

def remote_fn(doodad_config, variant):
    ## 获取后缀范围以允许每个作业运行多个试验
    n_suffices = variant['n_suffices']
    suffix_start = variant['suffix_start']
    # 删除不需要的变量
    del variant['n_suffices']
    del variant['suffix_start']

    # 生成命令行参数字符串
    kwarg_string = ' '.join([
        f'--{k} {v}' for k, v in variant.items()
    ])
    print(kwarg_string)

    # 数据集路径
    d4rl_path = os.path.join(doodad_config.output_directory, 'datasets/')
    # 列出代码路径下的文件
    os.system(f'ls -a {codepath}')
    # 重命名git目录
    os.system(f'mv {codepath}/git {codepath}/.git')

    # 循环运行多个试验
    for suffix in range(suffix_start, suffix_start + n_suffices):
        os.system(
            f'''export PYTHONPATH=$PYTHONPATH:{codepath} && '''
            f'''export CUDA_VISIBLE_DEVICES=0 && '''
            f'''export D4RL_DATASET_DIR={d4rl_path} && '''
            f'''python {os.path.join(codepath, script)} '''
            f'''--suffix {suffix} '''
            f'''{kwarg_string}'''
        )

    # 保存doodad配置
    save_doodad_config(doodad_config)

if __name__ == "__main__":

    # 环境列表
    environments = ['ant']
    # 缓冲区列表
    buffers = ['medium-expert-v2', 'medium-v2', 'medium-replay-v2', 'random-v2']
    # 数据集列表
    datasets = [f'{env}-{buf}' for env in environments for buf in buffers]

    # Azure日志路径
    azure_logpath = 'defaults/'

    # 需要扫描的参数
    params_to_sweep = {
        'dataset': datasets,
        'horizon': [15],
    }

    # 默认参数
    default_params = {
        'logbase': os.path.join('/doodad_tmp', azure_logpath, 'logs'),
        'prefix': 'plans/azure/',
        'verbose': False,
        'suffix_start': 0,
        'n_suffices': 3,
    }

    # 打印参数
    print(params_to_sweep)
    print(default_params)

    # 扫描函数
    sweep_function(
        remote_fn,
        params_to_sweep,
        default_params=default_params,
        config_path=os.path.abspath('azure/config.py'),
        log_path=azure_logpath,
        azure_region='westus2',
        # gpu_model='nvidia-tesla-v100',
        gpu_model='nvidia-tesla-t4',
        filter_dir=['logs', 'bin', 'mount'],
        use_gpu=True,
    )