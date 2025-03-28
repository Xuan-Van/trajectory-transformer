import os
import pdb

from doodad.wrappers.easy_launch import sweep_function, save_doodad_config

# 代码路径
codepath = '/home/code'
# 训练脚本路径
script = 'scripts/train.py'

def remote_fn(doodad_config, variant):
    # 将变量转换为命令行参数字符串
    kwarg_string = ' '.join([
        f'--{k} {v}' for k, v in variant.items()
    ])
    print(kwarg_string)

    # 设置D4RL数据集路径
    d4rl_path = os.path.join(doodad_config.output_directory, 'datasets/')
    # 列出代码路径下的所有文件和目录
    os.system(f'ls -a {codepath}')
    # 将git目录重命名为隐藏目录
    os.system(f'mv {codepath}/git {codepath}/.git')
    # 设置环境变量并运行训练脚本
    os.system(
        f'''export PYTHONPATH=$PYTHONPATH:{codepath} && '''
        f'''export CUDA_VISIBLE_DEVICES=0 && '''
        f'''export D4RL_DATASET_DIR={d4rl_path} && '''
        f'''python {os.path.join(codepath, script)} '''
        f'''{kwarg_string}'''
    )
    # 保存doodad配置
    save_doodad_config(doodad_config)

if __name__ == "__main__":
    # 定义环境列表
    environments = ['halfcheetah', 'hopper', 'walker2d', 'ant']
    # 定义缓冲区列表
    buffers = ['expert-v2']
    # 生成数据集列表
    datasets = [f'{env}-{buf}' for env in environments for buf in buffers]

    # Azure日志路径
    azure_logpath = 'defaults/'

    # 定义要扫描的参数
    params_to_sweep = {
        'dataset': datasets,
    }

    # 定义默认参数
    default_params = {
        'logbase': os.path.join('/doodad_tmp', azure_logpath, 'logs'),
        'exp_name': 'gpt/azure',
    }

    # 扫描函数
    sweep_function(
        remote_fn,
        params_to_sweep,
        default_params=default_params,
        config_path=os.path.abspath('azure/config.py'),
        log_path=azure_logpath,
        gpu_model='nvidia-tesla-v100',
        filter_dir=['logs', 'bin'],
        use_gpu=True,
    )