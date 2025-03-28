import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym # OpenAI开发的一个用于开发和比较强化学习算法的工具包
import mujoco_py as mjc # 用于与MuJoCo物理引擎交互的Python库
import pdb

from .arrays import to_np
from .video import save_video, save_videos
from ..datasets import load_environment, get_preprocess_fn

def make_renderer(args):
    """
    创建渲染器对象。
    参数:
    - args: 包含渲染器类型和其他参数的对象。
    返回:
    - 渲染器对象。
    """
    render_str = getattr(args, 'renderer')  # 获取渲染器类型字符串
    render_class = getattr(sys.modules[__name__], render_str)  # 获取渲染器类

    # 加载环境并获取预处理函数
    env = load_environment(args.dataset) # 使用load_environment函数加载指定数据集的环境
    preprocess_fn = get_preprocess_fn(args.dataset) # 使用get_preprocess_fn函数获取指定数据集的预处理函数
    observation = env.reset() # 重置环境并获取初始观察值
    observation = preprocess_fn(observation) # 使用预处理函数对观察值进行预处理

    return render_class(args.dataset, observation_dim=observation.size) # 使用获取的渲染器类实例化渲染器对象，并传入数据集名称和观察值维度

def split(sequence, observation_dim, action_dim):
    """
    将序列数据拆分为观察、动作、奖励和值。
    参数:
    - sequence: 包含观察、动作、奖励和值的序列数据。
    - observation_dim: 观察维度。
    - action_dim: 动作维度。
    返回:
    - 观察、动作、奖励和值的数组。
    """
    assert sequence.shape[1] == observation_dim + action_dim + 2 # 确保输入序列的第二维度（列数）等于观察维度、动作维度和两个额外值（奖励和值）的总和
    observations = sequence[:, :observation_dim] # 提取序列中前observation_dim列作为观察值
    actions = sequence[:, observation_dim:observation_dim + action_dim] # 提取序列中从observation_dim到observation_dim+action_dim列作为动作
    rewards = sequence[:, -2] # 提取序列中倒数第二列作为奖励
    values = sequence[:, -1] # 提取序列中最后一列作为值
    return observations, actions, rewards, values

def set_state(env, state):
    """
    设置环境的状态。
    参数:
    - env: 环境对象。
    - state: 状态数组。
    """
    qpos_dim = env.sim.data.qpos.size # 获取环境中的关节位置维度
    qvel_dim = env.sim.data.qvel.size # 获取环境中的关节速度维度
    qstate_dim = qpos_dim + qvel_dim # 计算总状态维度（关节位置和速度的总和）

    if 'ant' in env.name: # 如果环境名称中包含'ant'，则在状态前添加一个零值
        ypos = np.zeros(1)
        state = np.concatenate([ypos, state])

    if state.size == qpos_dim - 1 or state.size == qstate_dim - 1: # 如果状态维度比关节位置维度少1或比总状态维度少1，则在状态前添加一个零值
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])

    if state.size == qpos_dim: # 如果状态维度等于关节位置维度，则在状态后添加零值
        qvel = np.zeros(qvel_dim)
        state = np.concatenate([state, qvel])

    if 'ant' in env.name and state.size > qpos_dim + qvel_dim: # 如果环境名称中包含'ant'且状态维度大于总状态维度，则在状态前添加一个零值（xpos），并截取前qstate_dim个元素
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])[:qstate_dim]

    assert state.size == qpos_dim + qvel_dim # 确保状态维度等于总状态维度

    env.set_state(state[:qpos_dim], state[qpos_dim:]) # 将状态的前qpos_dim个元素设置为关节位置，剩余元素设置为关节速度

def rollout_from_state(env, state, actions):
    """
    从给定的状态开始，执行一系列动作并生成观察序列。
    参数:
    - env: 环境对象。
    - state: 初始状态。
    - actions: 动作序列。
    返回:
    - 观察序列。
    """
    qpos_dim = env.sim.data.qpos.size # 获取环境中的关节位置维度
    env.set_state(state[:qpos_dim], state[qpos_dim:]) # 将状态的前qpos_dim个元素设置为关节位置，剩余元素设置为关节速度
    observations = [env._get_obs()] # 获取初始状态的观察值，并将其添加到观察序列中
    for act in actions: # 遍历动作序列中的每个动作
        obs, rew, term, _ = env.step(act) # 执行动作，获取新的观察值、奖励、是否终止和其他信息
        observations.append(obs) # 将新的观察值添加到观察序列中
        if term: # 如果环境终止，则跳出循环
            break
    for i in range(len(observations), len(actions) + 1): # 如果动作序列提前终止，则用零值填充观察序列，使其长度与动作序列相同
        ## if terminated early, pad with zeros
        observations.append(np.zeros(obs.size))
    return np.stack(observations) # 将观察序列堆叠成一个数组并返回

class DebugRenderer:
    """
    调试渲染器类，用于在调试时生成简单的渲染结果。
    """

    def __init__(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        """
        生成一个简单的渲染结果。
        返回:
        - 10x10的黑色图像。
        """
        return np.zeros((10, 10, 3))

    def render_plan(self, *args, **kwargs):
        """
        渲染计划，当前实现为空。
        """
        pass

    def render_rollout(self, *args, **kwargs):
        """
        渲染回放，当前实现为空。
        """
        pass


class Renderer:
    """
    渲染器类，用于渲染环境和生成视频。
    """

    def __init__(self, env, observation_dim=None, action_dim=None):
        """
        初始化渲染器。
        参数:
        - env: 环境对象或环境名称。
        - observation_dim: 观察维度（可选）。
        - action_dim: 动作维度（可选）。
        """
        if type(env) is str: # 如果传入的env是字符串类型，则调用load_environment函数加载环境
            self.env = load_environment(env)
        else: # 否则，直接将env赋值给self.env
            self.env = env

        self.observation_dim = observation_dim or np.prod(self.env.observation_space.shape) # 初始化观察维度
        self.action_dim = action_dim or np.prod(self.env.action_space.shape) # 初始化动作维度
        self.viewer = mjc.MjRenderContextOffscreen(self.env.sim) # 初始化渲染上下文

    def __call__(self, *args, **kwargs):
        """
        调用渲染器对象时，调用renders方法。
        """
        return self.renders(*args, **kwargs)

    def render(self, observation, dim=256, render_kwargs=None):
        """
        渲染单个观察结果。
        参数:
        - observation: 观察结果。
        - dim: 渲染图像的尺寸。
        - render_kwargs: 渲染参数（可选）。
        返回:
        - 渲染的图像。
        """
        observation = to_np(observation) # 将观察结果转换为NumPy数组

        if render_kwargs is None: # 如果未传入渲染参数，则使用默认参数
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [0, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items(): # 遍历渲染参数，并设置到渲染上下文中
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        set_state(self.env, observation) # 设置环境状态

        if type(dim) == int: # 如果dim是整数，则将其转换为元组
            dim = (dim, dim)

        self.viewer.render(*dim) # 使用渲染上下文渲染图像
        data = self.viewer.read_pixels(*dim, depth=False) # 读取渲染的像素数据
        data = data[::-1, :, :] # 反转图像数据以正确显示
        return data # 返回渲染图像

    def renders(self, observations, **kwargs):
        """
        渲染多个观察结果。
        参数:
        - observations: 观察结果列表。
        - kwargs: 其他渲染参数。
        返回:
        - 渲染的图像序列。
        """
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs) # 调用render方法渲染单个观察结果，并获取渲染的图像
            images.append(img)
        return np.stack(images, axis=0) # 使用np.stack将图像列表堆叠成一个数组，并返回

    def render_plan(self, savepath, sequence, state, fps=30):
        """
        渲染计划并保存为视频。
        参数:
        - savepath: 保存路径。
        - sequence: 序列数据。
        - state: 初始状态。
        - fps: 视频帧率。
        """
        if len(sequence) == 1:
            return

        sequence = to_np(sequence) # 将序列数据转换为NumPy数组

        actions = sequence[:-1, self.observation_dim: self.observation_dim + self.action_dim] # 从序列中提取动作
        rollout_states = rollout_from_state(self.env, state, actions) # 使用提取的动作生成回放状态

        videos = [
            self.renders(sequence[:, :self.observation_dim]), # 渲染计划中的观察序列
            self.renders(rollout_states), # 渲染回放状态
        ]

        save_videos(savepath, *videos, fps=fps) # 将渲染的视频保存到指定路径，并设置帧率

    def render_rollout(self, savepath, states, **video_kwargs):
        """
        渲染回放并保存为视频。
        参数:
        - savepath: 保存路径。
        - states: 状态序列。
        - video_kwargs: 视频保存参数。
        """
        images = self(states)
        save_video(savepath, images, **video_kwargs)

class KitchenRenderer:
    """
    厨房环境渲染器类。
    """

    def __init__(self, env):
        """
        初始化厨房渲染器。
        参数:
        - env: 环境对象或环境名称。
        """
        if type(env) is str: # 如果传入的env是字符串类型，则调用gym.make函数加载环境
            self.env = gym.make(env)
        else:
            self.env = env

        self.observation_dim = np.prod(self.env.observation_space.shape) # 计算环境观察空间的维度
        self.action_dim = np.prod(self.env.action_space.shape) # 计算环境动作空间的维度

    def set_obs(self, obs, goal_dim=30):
        """
        设置环境的状态。
        参数:
        - obs: 观察结果。
        - goal_dim: 目标维度。
        """
        robot_dim = self.env.n_jnt # 获取环境中机器人的关节数量
        obj_dim = self.env.n_obj # 获取环境中物体的数量
        assert robot_dim + obj_dim + goal_dim == obs.size or robot_dim + obj_dim == obs.size # 确保观察结果的维度与机器人和物体的维度之和相匹配，或者与机器人、物体和目标的维度之和相匹配
        self.env.sim.data.qpos[:robot_dim] = obs[:robot_dim] # 将观察结果中前robot_dim个元素设置为机器人的关节位置
        self.env.sim.data.qpos[robot_dim:robot_dim + obj_dim] = obs[robot_dim:robot_dim + obj_dim] # 将观察结果中接下来的obj_dim个元素设置为物体的关节位置
        self.env.sim.forward() # 调用forward方法更新环境

    def rollout(self, obs, actions):
        """
        从给定的观察结果开始，执行一系列动作并生成观察序列。
        参数:
        - obs: 初始观察结果。
        - actions: 动作序列。
        返回:
        - 观察序列。
        """
        self.set_obs(obs) # 调用set_obs方法设置环境的初始状态
        observations = [env._get_obs()] # 获取初始状态的观察值，并将其添加到观察序列中
        for act in actions: # 遍历动作序列中的每个动作
            obs, rew, term, _ = env.step(act) # 执行动作，获取新的观察值、奖励、是否终止和其他信息
            observations.append(obs) # 将新的观察值添加到观察序列中
            if term:
                break
        for i in range(len(observations), len(actions) + 1): # 如果动作序列提前终止，则用零值填充观察序列，使其长度与动作序列相同
            ## if terminated early, pad with zeros
            observations.append(np.zeros(observations[-1].size))
        return np.stack(observations) # 将观察序列堆叠成一个数组并返回

    def render(self, observation, dim=512, onscreen=False):
        """
        渲染单个观察结果。
        参数:
        - observation: 观察结果。
        - dim: 渲染图像的尺寸。
        - onscreen: 是否在屏幕上显示。
        返回:
        - 渲染的图像。
        """
        self.env.sim_robot.renderer._camera_settings.update({ # 更新渲染器的相机设置，包括距离、方位角、仰角和目标点
            'distance': 4.5,
            'azimuth': 90,
            'elevation': -25,
            'lookat': [0, 1, 2],
        })
        self.set_obs(observation) # 调用set_obs方法设置环境的观察结果
        if onscreen: # 如果onscreen为True，则在屏幕上显示渲染结果
            self.env.render()
        return self.env.sim_robot.renderer.render_offscreen(dim, dim) # 使用渲染器渲染图像，并返回渲染的图像

    def renders(self, observations, **kwargs):
        """
        渲染多个观察结果。
        参数:
        - observations: 观察结果列表。
        - kwargs: 其他渲染参数。
        返回:
        - 渲染的图像序列。
        """
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def render_plan(self, *args, **kwargs):
        """
        渲染计划，当前实现为调用render_rollout方法。
        """
        return self.render_rollout(*args, **kwargs)

    def render_rollout(self, savepath, states, **video_kwargs):
        """
        渲染回放并保存为视频。
        参数:
        - savepath: 保存路径。
        - states: 状态序列。
        - video_kwargs: 视频保存参数。
        """
        images = self(states)
        save_video(savepath, images, **video_kwargs)

    def __call__(self, *args, **kwargs):
        """
        调用渲染器对象时，调用renders方法。
        """
        return self.renders(*args, **kwargs)

ANTMAZE_BOUNDS = { # 存储不同AntMaze环境的边界值
    'antmaze-umaze-v0': (-3, 11), # 小型迷宫环境
    'antmaze-medium-play-v0': (-3, 23), # 中型迷宫环境，适合玩耍
    'antmaze-medium-diverse-v0': (-3, 23), # 中型迷宫环境，多样性较高
    'antmaze-large-play-v0': (-3, 39), # 大型迷宫环境，适合玩耍
    'antmaze-large-diverse-v0': (-3, 39), # 大型迷宫环境，多样性较高
}

class AntMazeRenderer:
    """
    AntMaze环境渲染器类。
    """

    def __init__(self, env_name):
        """
        初始化AntMaze渲染器。
        参数:
        - env_name: 环境名称。
        """
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped # 使用gym.make函数加载指定名称的环境，并使用unwrapped方法获取原始环境对象
        self.observation_dim = np.prod(self.env.observation_space.shape) # 计算环境观察空间的维度
        self.action_dim = np.prod(self.env.action_space.shape) # 计算环境动作空间的维度

    def renders(self, savepath, X):
        """
        渲染多个路径并保存为图像。
        参数:
        - savepath: 保存路径。
        - X: 路径数据。
        """
        plt.clf() # 清除当前的图形，以便绘制新的图形

        if X.ndim < 3: # 如果路径数据的维度小于3，则将其扩展为三维数组
            X = X[None]

        N, path_length, _ = X.shape # 获取路径数据的数量N和每个路径的长度
        if N > 4: # 如果路径数量大于4，则创建4行N/4列的子图
            fig, axes = plt.subplots(4, int(N / 4))
            axes = axes.flatten()
            fig.set_size_inches(N / 4, 8)
        elif N > 1: # 如果路径数量大于1但小于等于4，则创建1行N列的子图
            fig, axes = plt.subplots(1, N)
            fig.set_size_inches(8, 8)
        else: # 如果路径数量为1，则创建单个子图
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(8, 8)

        colors = plt.cm.jet(np.linspace(0, 1, path_length)) # 生成颜色映射，用于绘制路径
        for i in range(N): # 遍历每个路径
            ax = axes if N == 1 else axes[i] # 获取当前子图
            xlim, ylim = self.plot_boundaries(ax=ax) # 调用plot_boundaries方法绘制环境边界
            x = X[i] # 获取当前路径数据
            ax.scatter(x[:, 0], x[:, 1], c=colors) # 在子图中绘制路径
            # 隐藏坐标轴刻度
            ax.set_xticks([])
            ax.set_yticks([])
            # 设置子图的x和y轴范围
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        # 保存图像
        plt.savefig(savepath + '.png')
        plt.close()
        print(f'[ attentive/utils/visualization ] Saved to: {savepath}')

    def plot_boundaries(self, N=100, ax=None):
        """
        绘制AntMaze环境的边界。
        参数:
        - N: 网格点数。
        - ax: 子图对象（可选）。
        返回:
        - x和y的边界。
        """
        ax = ax or plt.gca() # 获取子图对象
        # 获取环境边界
        xlim = ANTMAZE_BOUNDS[self.env_name]
        ylim = ANTMAZE_BOUNDS[self.env_name]
        # 生成网格点
        X = np.linspace(*xlim, N)
        Y = np.linspace(*ylim, N)
        # 检查碰撞
        Z = np.zeros((N, N))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                collision = self.env.unwrapped._is_in_collision((x, y)) # 检查当前点是否在碰撞中
                Z[-j, i] = collision # 将碰撞结果存储在矩阵Z中

        ax.imshow(Z, extent=(*xlim, *ylim), aspect='auto', cmap=plt.cm.binary) # 使用imshow方法在子图中绘制边界
        return xlim, ylim

    def render_plan(self, savepath, discretizer, state, sequence):
        """
        渲染计划并保存为图像。
        参数:
        - savepath: 保存路径。
        - discretizer: 离散化器对象。
        - state: 初始状态。
        - sequence: 序列数据。
        """
        if len(sequence) == 1:
            return

        sequence = to_np(sequence)

        sequence_recon = discretizer.reconstruct(sequence) # 使用离散化器对象重建序列数据

        observations, actions, *_ = split(sequence_recon, self.observation_dim, self.action_dim) # 使用split函数将重建后的序列数据拆分为观察、动作、奖励和值

        rollout_states = rollout_from_state(self.env, state, actions[:-1]) # 使用提取的动作生成回放状态

        X = np.stack([observations, rollout_states], axis=0) # 将观察序列和回放状态堆叠成一个数组

        self.renders(savepath, X) # 调用renders方法渲染路径并保存为图像

    def render_rollout(self, savepath, states, **video_kwargs):
        """
        渲染回放并保存为图像
        参数:
        - savepath: 保存路径。
        - states: 状态序列。
        - video_kwargs: 视频保存参数。
        """
        if type(states) is list: # 如果states是列表类型，则将其转换为NumPy数组
            states = np.stack(states, axis=0)[None] # 将列表堆叠成一个数组，并在第一个维度上添加一个维度
        images = self.renders(savepath, states) # 调用renders方法渲染状态序列并保存为图像

class Maze2dRenderer(AntMazeRenderer):
    """
    Maze2d环境渲染器类，继承自AntMazeRenderer。
    """

    def _is_in_collision(self, x, y):
        """
        检查是否在碰撞中。
        参数:
        - x: x坐标。
        - y: y坐标。
        返回:
        - 是否在碰撞中。
        """
        maze = self.env.maze_arr # 获取环境的迷宫数组
        ind = maze[int(x), int(y)] # 获取迷宫数组中给定坐标的值
        return ind == 10 # 如果迷宫数组中的值为10，则表示在碰撞中，返回True；否则返回False

    def plot_boundaries(self, N=100, ax=None, eps=1e-6):
        """
        绘制Maze2d环境的边界。
        参数:
        - N: 网格点数。
        - ax: 子图对象（可选）。
        - eps: 边界误差。
        返回:
        - x和y的边界。
        """
        ax = ax or plt.gca()

        maze = self.env.maze_arr
        # 设置x和y的边界，减去一个小的误差eps以避免边界问题
        xlim = (0, maze.shape[1] - eps)
        ylim = (0, maze.shape[0] - eps)
        # 在x和y方向上生成N个网格点
        X = np.linspace(*xlim, N)
        Y = np.linspace(*ylim, N)
        # 检查碰撞
        Z = np.zeros((N, N))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                collision = self._is_in_collision(x, y)
                Z[-j, i] = collision

        ax.imshow(Z, extent=(*xlim, *ylim), aspect='auto', cmap=plt.cm.binary) # 使用imshow方法在子图中绘制边界
        return xlim, ylim

    def renders(self, savepath, X):
        """
        渲染多个路径并保存为图像。
        参数:
        - savepath: 保存路径。
        - X: 路径数据。
        """
        return super().renders(savepath, X + 0.5) # 调用父类的renders方法，并将路径数据X加上0.5后传递给父类方法

# --------------------------------- planning callbacks ---------------------------------#