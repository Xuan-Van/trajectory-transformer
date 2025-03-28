import os  # 导入操作系统接口模块
import importlib  # 导入动态导入模块
import random  # 导入随机数生成模块
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
from tap import Tap  # 导入Tap库，用于命令行参数解析
import pdb  # 导入Python调试器

from .serialization import mkdir  # 从当前包的serialization模块导入mkdir函数
from .arrays import set_device  # 从当前包的arrays模块导入set_device函数
from .git_utils import (  # 从当前包的git_utils模块导入get_git_rev和save_git_diff函数
    get_git_rev,
    save_git_diff,
)


def set_seed(seed):
    """
    设置随机种子以确保结果的可重复性。
    参数:
    seed (int): 随机种子值。
    """
    random.seed(seed)  # 设置Python内置随机数生成器的种子
    np.random.seed(seed)  # 设置NumPy随机数生成器的种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数生成器的种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch GPU随机数生成器的种子

def watch(args_to_watch):
    """
    生成一个函数，用于根据给定的参数生成实验名称。
    参数:
    args_to_watch (list of tuples): 包含要监视的参数键和标签的元组列表。
    返回:
    function: 一个函数，接受参数对象并返回生成的实验名称。
    """

    def _fn(args): # 接受一个参数对象args，并根据args_to_watch中的键和标签生成实验名称
        exp_name = [] # 存储生成的实验名称片段
        for key, label in args_to_watch:
            if not hasattr(args, key): # 如果参数对象args中没有key这个属性，则跳过该元组 $ hasattr：检查对象是否具有指定的属性
                continue
            val = getattr(args, key)
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/') # 替换实验名称中的`/_`为`/`，以避免路径问题
        return exp_name

    return _fn

class Parser(Tap):
    """
    自定义命令行参数解析器，继承自Tap库的Tap类。
    """

    def save(self):
        """
        将解析后的参数保存到指定路径。
        """
        fullpath = os.path.join(self.savepath, 'args.json')
        print(f'[ utils/setup ] Saved args to {fullpath}')
        super().save(fullpath, skip_unpicklable=True) # 调用父类Tap的save方法，将解析后的参数保存到指定的路径fullpath

    def parse_args(self, experiment=None):
        """
        解析命令行参数并进行一系列设置。
        参数:
        experiment (str, optional): 实验名称，默认为None。
        返回:
        args: 解析后的参数对象。
        """
        args = super().parse_args(known_only=True) # 调用父类的解析方法
        ## 如果没有加载配置脚本，跳过设置步骤
        if not hasattr(args, 'config'): return args # 检查是否加载配置脚本
        args = self.read_config(args, experiment) # 读取配置文件
        self.add_extras(args) # 添加额外参数
        self.set_seed(args) # 设置随机种子
        self.get_commit(args) # 获取当前Git提交的哈希值
        self.generate_exp_name(args) # 生成实验名称
        self.mkdir(args) # 创建保存路径
        self.save_diff(args) # 保存Git差异文件
        return args

    def read_config(self, args, experiment):
        """
        从配置文件中加载参数。
        参数:
        args: 参数对象。
        experiment (str): 实验名称。
        返回:
        args: 更新后的参数对象。
        """
        dataset = args.dataset.replace('-', '_') # 确保数据集名称的格式一致
        print(f'[ utils/setup ] Reading config: {args.config}:{dataset}') # 打印读取配置信息
        module = importlib.import_module(args.config) # 动态导入配置模块
        params = getattr(module, 'base')[experiment] # 获取基本参数

        if hasattr(module, dataset) and experiment in getattr(module, dataset): # 检查导入的模块中是否存在与数据集名称dataset对应的字典，并且该字典中是否包含与experiment对应的覆盖参数
            print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides) # 使用这些覆盖参数更新params字典
        else:
            print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        for key, val in params.items():
            setattr(args, key, val) # 更新参数对象

        return args

    def add_extras(self, args):
        """
        使用命令行参数覆盖配置文件中的参数。
        参数:
        args: 参数对象。
        """
        extras = args.extra_args # 获取额外参数
        if not len(extras): # 检查是否有额外参数
            return

        print(f'[ utils/setup ] Found extras: {extras}') # 打印额外参数信息
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}' # 检查额外参数的数量
        for i in range(0, len(extras), 2): # 遍历额外参数并覆盖配置
            key = extras[i].replace('--', '') # 去掉参数键前面的`--`前缀
            val = extras[i + 1] # 获取参数值
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}' # 检查args对象是否具有key属性，如果没有则抛出断言错误
            old_val = getattr(args, key) # 获取args对象中key属性的旧值
            old_type = type(old_val) # 获取旧值的类型
            print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}') # 打印出覆盖信息
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]: # 如果旧值的类型是布尔型或None，则使用eval解析参数值
                val = eval(val)
            else: # 否则将参数值转换为旧值的类型
                val = old_type(val)
            setattr(args, key, val) # 将新的参数值设置到args对象中

    def set_seed(self, args):
        """
        设置随机种子。
        参数:
        args: 参数对象。
        """
        if not 'seed' in dir(args):
            return
        set_seed(args.seed)

    def generate_exp_name(self, args):
        """
        生成实验名称。
        参数:
        args: 参数对象。
        """
        if not 'exp_name' in dir(args): # 检查是否存在exp_name属性
            return
        exp_name = getattr(args, 'exp_name') # 获取exp_name属性
        if callable(exp_name): # 检查exp_name是否可调用
            exp_name_string = exp_name(args) # 生成实验名称
            print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string) # 设置exp_name属性

    def mkdir(self, args):
        """
        创建保存路径并保存参数。
        参数:
        args: 参数对象。
        """
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name) # 构建保存路径
            if 'suffix' in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath): # 创建保存路径
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save()

    def get_commit(self, args):
        """
        获取当前Git提交的哈希值。
        参数:
        args: 参数对象。
        """
        args.commit = get_git_rev()

    def save_diff(self, args):
        """
        保存当前Git提交的差异文件。
        参数:
        args: 参数对象。
        """
        try:
            save_git_diff(os.path.join(args.savepath, 'diff.txt'))
        except:
            print('[ utils/setup ] WARNING: did not save git diff')