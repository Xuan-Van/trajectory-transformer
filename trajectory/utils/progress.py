import time
import math
import re
import pdb

class Progress:
    """
    进度条类，用于显示任务的进度和相关信息。
    """

    def __init__(self, total, name='Progress', ncol=3, max_length=30, indent=8, line_width=100, speed_update_freq=100):
        """
        初始化进度条对象。
        :param total: 总步骤数
        :param name: 进度条名称
        :param ncol: 参数列数
        :param max_length: 每列最大长度
        :param indent: 缩进长度
        :param line_width: 每行宽度
        :param speed_update_freq: 更新速度的频率
        """
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self._speed_update_freq = speed_update_freq

        self._step = 0
        self._prev_line = '\033[F'  # 用于回到上一行的控制字符
        self._clear_line = ' ' * self.line_width  # 用于清空一行的字符串

        self._pbar_size = self.ncol * self.max_length  # 进度条的总长度
        self._complete_pbar = '#' * self._pbar_size  # 完成的进度条字符串
        self._incomplete_pbar = ' ' * self._pbar_size  # 未完成的进度条字符串

        self.lines = ['']  # 存储描述信息的行
        self.fraction = '{} / {}'.format(0, self.total)  # 当前进度

        self.resume()  # 开始计时

    def update(self, description, n=1):
        """
        更新进度条，增加步骤数并更新描述信息。
        :param description: 描述信息
        :param n: 增加的步骤数
        """
        self._step += n
        if self._step % self._speed_update_freq == 0: # 倍数
            self._time0 = time.time()  # 记录当前时间
            self._step0 = self._step  # 记录当前步骤数
        self.set_description(description)  # 更新描述信息

    def resume(self):
        """
        恢复进度条，开始计时。
        """
        self._skip_lines = 1 # 在更新进度条时需要跳过的行数
        print('\n', end='')
        self._time0 = time.time()  # 记录当前时间
        self._step0 = self._step  # 记录当前步骤数

    def pause(self):
        """
        暂停进度条，清空当前显示。
        """
        self._clear()
        self._skip_lines = 1

    def set_description(self, params=[]):
        """
        设置描述信息并更新进度条显示。
        :param params: 描述信息的参数列表或字典
        """
        if isinstance(params, dict): # $ isinstance：检查params是否是字典
            params = sorted([(key, val) for key, val in params.items()])

        # 清空当前显示
        self._clear()

        # 计算进度百分比
        percent, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

        # 计算速度
        speed = self._format_speed(self._step)

        # 格式化参数
        num_params = len(params) # 计算参数数量
        nrow = math.ceil(num_params / self.ncol) # 计算行数
        params_split = self._chunk(params, self.ncol) # 分块处理
        params_string, lines = self._format(params_split) # 格式化为字符串
        self.lines = lines

        # 生成描述信息字符串并打印
        description = '{} | {}{}'.format(percent, speed, params_string)
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr):
        """
        追加描述信息。
        :param descr: 追加的描述信息
        """
        self.lines.append(descr)

    def _clear(self):
        """
        清空当前显示。
        """
        position = self._prev_line * self._skip_lines # 计算位置
        empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)]) # 生成空行
        print(position, end='')
        print(empty)
        print(position, end='')

    def _format_percent(self, n, total):
        """
        格式化进度百分比。
        :param n: 当前步骤数
        :param total: 总步骤数
        :return: 格式化后的百分比字符串和当前进度字符串
        """
        if total: # 检查总步骤数
            percent = n / float(total) # 计算进度百分比
            complete_entries = int(percent * self._pbar_size) # 计算完成的进度条长度
            incomplete_entries = self._pbar_size - complete_entries # 计算未完成的进度条长度
            pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries] # 生成进度条字符串
            fraction = '{} / {}'.format(n, total) # 生成当前进度字符串
            string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100)) # 生成格式化后的百分比字符串
        else: # 处理总步骤数为零的情况
            fraction = '{}'.format(n)
            string = '{} iterations'.format(n)
        return string, fraction

    def _format_speed(self, n):
        """
        格式化速度。
        :param n: 当前步骤数
        :return: 格式化后的速度字符串
        """
        num_steps = n - self._step0 # 计算步骤数差值
        t = time.time() - self._time0 # 计算时间差值
        speed = num_steps / t # 计算速度
        string = '{:.1f} Hz'.format(speed) # 格式化速度字符串
        if num_steps > 0: # 更新速度字符串
            self._speed = string
        return string

    def _chunk(self, l, n):
        """
        将列表分成指定大小的块。
        :param l: 列表
        :param n: 每块的大小
        :return: 分块后的列表
        """
        return [l[i:i + n] for i in range(0, len(l), n)] # 使用列表推导式将列表l分成大小为n的块

    def _format(self, chunks):
        """
        格式化参数块。
        :param chunks: 参数块列表
        :return: 格式化后的字符串和行列表
        """
        lines = [self._format_chunk(chunk) for chunk in chunks] # 格式化每个块
        lines.insert(0, '') # 插入空行
        padding = '\n' + ' ' * self.indent # 生成填充字符串
        string = padding.join(lines) # 连接行列表
        return string, lines

    def _format_chunk(self, chunk):
        """
        格式化单个参数块。
        :param chunk: 参数块
        :return: 格式化后的字符串
        """
        line = ' | '.join([self._format_param(param) for param in chunk])
        return line

    def _format_param(self, param, str_length=8):
        """
        格式化单个参数。
        :param param: 参数
        :param str_length: 参数名称的长度
        :return: 格式化后的字符串
        """
        k, v = param
        k = k.rjust(str_length) # 将键k右对齐，使其长度为str_length
        if isinstance(v, float) or hasattr(v, 'item'):
            string = '{}: {:12.4f}' # 格式化浮点数或具有item属性的值
        else:
            string = '{}: {}'
            v = str(v).rjust(12) # 将其转换为字符串并右对齐，使其长度为12
        return string.format(k, v)[:self.max_length]

    def stamp(self):
        """
        打印当前进度和描述信息。
        """
        if self.lines != ['']:
            params = ' | '.join(self.lines)
            string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed) # 生成最终字符串
            string = re.sub(r'\s+', ' ', string) # 去除多余空格
            self._clear()
            print(string, end='\n')
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self):
        """
        关闭进度条。
        """
        self.pause()

class Silent:
    """
    静默模式类，用于在不显示进度条的情况下执行任务。
    """

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        """
        返回一个空函数，用于静默模式下的所有方法调用。
        """
        return lambda *args: None

if __name__ == '__main__':
    # 创建静默模式对象
    silent = Silent()
    silent.update()
    silent.stamp()
    # 创建进度条对象
    num_steps = 1000
    progress = Progress(num_steps)
    for i in range(num_steps): # 更新进度条
        progress.update()
        params = [
            ['A', '{:06d}'.format(i)],
            ['B', '{:06d}'.format(i)],
            ['C', '{:06d}'.format(i)],
            ['D', '{:06d}'.format(i)],
            ['E', '{:06d}'.format(i)],
            ['F', '{:06d}'.format(i)],
            ['G', '{:06d}'.format(i)],
            ['H', '{:06d}'.format(i)],
        ]
        progress.set_description(params)
        time.sleep(0.01)
    progress.close() # 关闭进度条