# 创建挂载点目录
mkdir -p ~/azure_mount

# 使用 blobfuse 挂载 Azure Blob Storage
blobfuse ~/azure_mount --tmp-path=/tmp --config-file=./azure/fuse.cfg