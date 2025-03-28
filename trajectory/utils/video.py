import os  # 导入os模块，用于处理文件路径和目录操作
import numpy as np  # 导入numpy库，用于处理数组和矩阵运算
import skvideo.io  # 导入skvideo.io模块，用于视频的读写操作

def _make_dir(filename):
    """
    创建文件所在的目录（如果不存在）
    Args:
        filename (str): 文件的完整路径
    """
    folder = os.path.dirname(filename)  # 获取文件所在的目录路径
    if not os.path.exists(folder):  # 如果目录不存在
        os.makedirs(folder)  # 创建目录

def save_video(filename, video_frames, fps=60, video_format='mp4'):
    """
    将视频帧保存为视频文件
    Args:
        filename (str): 保存视频的文件路径
        video_frames (np.ndarray): 视频帧的numpy数组，形状为 [N x H x W x C]
        fps (int): 视频的帧率，默认为60
        video_format (str): 视频的格式，默认为'mp4'
    """
    assert fps == int(fps), fps  # 确保fps是整数
    _make_dir(filename)  # 创建文件所在的目录

    skvideo.io.vwrite(
        filename,  # 保存视频的文件路径
        video_frames,  # 视频帧的numpy数组
        inputdict={
            '-r': str(int(fps)),  # 设置输入视频的帧率
        },
        outputdict={
            '-f': video_format,  # 设置输出视频的格式
            '-pix_fmt': 'yuv420p',  # 设置像素格式为yuv420p，解决某些系统上的兼容性问题
        }
    )

def save_videos(filename, *video_frames, **kwargs):
    """
    将多个视频帧拼接后保存为视频文件
    Args:
        filename (str): 保存视频的文件路径
        *video_frames (np.ndarray): 多个视频帧的numpy数组，每个数组的形状为 [N x H x W x C]
        **kwargs: 传递给save_video函数的其他参数
    """
    ## video_frame : [ N x H x W x C ]
    video_frames = np.concatenate(video_frames, axis=2)  # 将多个视频帧在宽度方向（axis=2）拼接
    save_video(filename, video_frames, **kwargs)  # 调用save_video函数保存拼接后的视频