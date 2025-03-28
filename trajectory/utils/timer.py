import time  # 导入时间模块，用于获取当前时间

class Timer:
    """
    一个简单的计时器类，用于测量时间间隔。
    """

    def __init__(self):
        """
        初始化计时器，记录当前时间为起始时间。
        """
        self._start = time.time()  # 记录当前时间为起始时间

    def __call__(self, reset=True):
        """
        计算自上次调用或初始化以来的时间差，并返回该时间差。
        参数:
        reset (bool): 如果为True，则在计算时间差后重置起始时间。
        返回:
        float: 自上次调用或初始化以来的时间差（以秒为单位）。
        """
        now = time.time()  # 获取当前时间
        diff = now - self._start  # 计算时间差
        if reset:
            self._start = now  # 如果reset为True，则重置起始时间为当前时间
        return diff  # 返回时间差