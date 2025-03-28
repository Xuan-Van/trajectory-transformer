import os  # 导入os模块，用于处理文件路径
import collections  # 导入collections模块，用于处理数据结构
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象

class Config(collections.Mapping):
    """
    配置类，继承自collections.Mapping，用于存储和管理配置参数
    """

    def __init__(self, _class, verbose=True, savepath=None, **kwargs): # $ 魔术方法，与普通函数不同，可以自动调用，还可以与内置函数和操作符交互
        """
        初始化配置对象
        :param _class: 配置对应的类
        :param verbose: 是否打印配置信息，默认为True
        :param savepath: 配置保存路径，默认为None
        :param kwargs: 配置参数
        """
        self._class = _class  # 存储配置对应的类
        self._dict = {}  # 初始化配置字典

        for key, val in kwargs.items():  # 遍历传入的配置参数
            self._dict[key] = val  # 将参数存储到配置字典中

        if verbose:  # 如果verbose为True $ verbose用于控制程序输出的详细程度
            print(self)  # 打印配置信息

        if savepath is not None:  # 如果指定了保存路径
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath  # 处理保存路径 $ 如果是元组则拼接为字符串，否则直接使用
            pickle.dump(self, open(savepath, 'wb'))  # 将配置对象序列化并保存到文件 $ wb：二进制形式写入；反序列化：使用pickle.load；序列化的原因：在程序下次运行时可以恢复对象的状态
            print(f'Saved config to: {savepath}\n')  # 打印保存路径

    def __repr__(self):
        """
        返回配置对象的字符串表示
        :return: 配置对象的字符串表示
        """
        string = f'\nConfig: {self._class}\n'  # 初始化字符串
        for key in sorted(self._dict.keys()):  # 遍历配置字典的键
            val = self._dict[key]  # 获取键对应的值
            string += f'    {key}: {val}\n'  # 将键值对添加到字符串中
        return string  # 返回字符串

    def __iter__(self):
        """
        返回配置字典的迭代器
        :return: 配置字典的迭代器
        """
        return iter(self._dict)  # 返回配置字典的迭代器 $ 迭代器用于遍历序列等可迭代对象

    def __getitem__(self, item):
        """
        获取配置字典中指定键的值
        :param item: 键
        :return: 键对应的值
        """
        return self._dict[item]  # 返回键对应的值

    def __len__(self):
        """
        返回配置字典的长度
        :return: 配置字典的长度
        """
        return len(self._dict)  # 返回配置字典的长度

    def __call__(self):
        """
        调用配置对象时，创建并返回配置对应的类的实例
        :return: 配置对应的类的实例
        """
        return self.make()  # 调用make方法创建并返回实例

    def __getattr__(self, attr):
        """
        获取配置对象的属性
        :param attr: 属性名
        :return: 属性值
        """
        if attr == '_dict' and '_dict' not in vars(self):  # 如果属性名为'_dict'且未初始化
            self._dict = {}  # 初始化配置字典
        try:
            return self._dict[attr]  # 返回属性值
        except KeyError:  # 如果键不存在
            raise AttributeError(attr)  # 抛出属性错误

    def make(self):
        """
        创建并返回配置对应的类的实例
        :return: 配置对应的类的实例
        """
        if 'GPT' in str(self._class) or 'Trainer' in str(self._class):  # 如果配置对应的类是GPT或Trainer
            ## GPT类期望将配置作为唯一输入
            return self._class(self)  # 创建并返回实例
        else:
            return self._class(**self._dict)  # 使用配置字典创建并返回实例