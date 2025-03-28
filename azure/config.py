import os

def get_docker_username():
    import subprocess
    import shlex
    # 使用subprocess运行'docker info'命令，并获取输出
    ps = subprocess.Popen(shlex.split('docker info'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 使用sed命令从输出中提取用户名
    output = subprocess.check_output(shlex.split("sed '/Username:/!d;s/.* //'"), stdin=ps.stdout)
    # 将提取的用户名解码为UTF-8格式，并去除换行符
    username = output.decode('utf-8').replace('\n', '')
    # 打印提取的用户名
    print(f'[ azure/config ] Grabbed username from `docker info`: {username}')
    return username

# 获取当前文件的目录路径
CWD = os.path.dirname(__file__)
# 获取模块的根目录路径
MODULE_PATH = os.path.dirname(CWD)

# 需要挂载的代码目录列表
CODE_DIRS_TO_MOUNT = [
]
# 需要挂载的非代码目录列表
NON_CODE_DIRS_TO_MOUNT = [
    dict(
        local_dir=MODULE_PATH,  # 本地目录路径
        mount_point='/home/code',  # 挂载点路径
    ),
]
# 需要挂载的远程目录列表
REMOTE_DIRS_TO_MOUNT = [
    dict(
        local_dir='/doodad_tmp/',  # 本地目录路径
        mount_point='/doodad_tmp/',  # 挂载点路径
    ),
]
# 本地日志目录路径
LOCAL_LOG_DIR = '/tmp'

# 默认的Azure GPU型号
DEFAULT_AZURE_GPU_MODEL = 'nvidia-tesla-t4'
# 默认的Azure实例类型
DEFAULT_AZURE_INSTANCE_TYPE = 'Standard_DS1_v2'
# 默认的Azure区域
DEFAULT_AZURE_REGION = 'eastus'
# 默认的Azure资源组
DEFAULT_AZURE_RESOURCE_GROUP = 'traj'
# 默认的Azure虚拟机名称
DEFAULT_AZURE_VM_NAME = 'traj-vm'
# 默认的Azure虚拟机密码
DEFAULT_AZURE_VM_PASSWORD = 'Azure1'

# 从环境变量中获取Docker用户名，如果没有则调用get_docker_username函数获取
DOCKER_USERNAME = os.environ.get('DOCKER_USERNAME', get_docker_username())
# 默认的Docker镜像名称
DEFAULT_DOCKER = f'docker.io/{DOCKER_USERNAME}/trajectory:latest'

# 打印本地目录路径
print(f'[ azure/config ] Local dir: {MODULE_PATH}')
# 打印默认的GPU型号
print(f'[ azure/config ] Default GPU model: {DEFAULT_AZURE_GPU_MODEL}')
# 打印默认的Docker镜像名称
print(f'[ azure/config ] Default Docker image: {DEFAULT_DOCKER}')

# 从环境变量中获取Azure订阅ID
AZ_SUB_ID = os.environ['AZURE_SUBSCRIPTION_ID']
# 从环境变量中获取Azure客户端ID
AZ_CLIENT_ID = os.environ['AZURE_CLIENT_ID']
# 从环境变量中获取Azure租户ID
AZ_TENANT_ID = os.environ['AZURE_TENANT_ID']
# 从环境变量中获取Azure客户端密钥
AZ_SECRET = os.environ['AZURE_CLIENT_SECRET']
# 从环境变量中获取Azure存储容器名称
AZ_CONTAINER = os.environ['AZURE_STORAGE_CONTAINER']
# 从环境变量中获取Azure存储连接字符串
AZ_CONN_STR = os.environ['AZURE_STORAGE_CONNECTION_STRING']