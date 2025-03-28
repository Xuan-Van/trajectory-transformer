import os  # 导入os模块，用于处理文件路径
import git  # 导入git模块，用于与Git仓库交互
import pdb  # 导入pdb模块，用于调试

# 获取项目根目录的路径
PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..'))) # os.path.dirname：返回指定路径的目录部分；os.path.realpath：返回指定路径的规范化绝对路径

def get_repo(path=PROJECT_PATH, search_parent_directories=True):
    """
    获取Git仓库对象
    :param path: 仓库路径，默认为项目根目录
    :param search_parent_directories: 是否在父目录中搜索仓库，默认为True
    :return: Git仓库对象
    """
    repo = git.Repo(
        path, search_parent_directories=search_parent_directories)
    return repo

def get_git_rev(*args, **kwargs):
    """
    获取当前Git提交的版本信息
    :param args: 传递给get_repo函数的参数
    :param kwargs: 传递给get_repo函数的键值对参数
    :return: 当前Git提交的版本信息，如果失败则返回None
    """
    try:
        repo = get_repo(*args, **kwargs)
        if repo.head.is_detached:  # 判断是否处于分离头指针状态
            git_rev = repo.head.object.name_rev  # 获取分离头指针的版本信息
        else:
            git_rev = repo.active_branch.commit.name_rev  # 获取当前分支的版本信息
    except:
        git_rev = None  # 如果发生异常，返回None

    return git_rev

def git_diff(*args, **kwargs):
    """
    获取当前Git仓库的差异信息
    :param args: 传递给get_repo函数的参数
    :param kwargs: 传递给get_repo函数的键值对参数
    :return: 当前Git仓库的差异信息
    """
    repo = get_repo(*args, **kwargs)
    diff = repo.git.diff()  # 获取差异信息
    return diff

def save_git_diff(savepath, *args, **kwargs):
    """
    保存当前Git仓库的差异信息到文件
    :param savepath: 保存差异信息的文件路径
    :param args: 传递给git_diff函数的参数
    :param kwargs: 传递给git_diff函数的键值对参数
    """
    diff = git_diff(*args, **kwargs)
    with open(savepath, 'w') as f:  # 打开文件以写入模式
        f.write(diff)  # 将差异信息写入文件

if __name__ == '__main__':
    """
    主程序入口
    """
    git_rev = get_git_rev()  # 获取当前Git提交的版本信息
    print(git_rev)  # 打印版本信息

    save_git_diff('diff_test.txt')  # 保存当前Git仓库的差异信息到文件