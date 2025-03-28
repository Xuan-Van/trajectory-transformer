# 设置下载目录
DOWNLOAD_DIR="bin"

# 创建下载目录
mkdir -p $DOWNLOAD_DIR

# 下载 AzCopy 压缩包并保存到指定目录
wget https://aka.ms/downloadazcopy-v10-linux -O $DOWNLOAD_DIR/download.tar.gz

# 解压缩下载的文件到指定目录
tar -xvf $DOWNLOAD_DIR/download.tar.gz --one-top-level=$DOWNLOAD_DIR

# 将解压后的 azcopy 可执行文件移动到下载目录
mv $DOWNLOAD_DIR/*/azcopy $DOWNLOAD_DIR

# 删除下载的压缩包
rm $DOWNLOAD_DIR/download.tar.gz

# 删除解压后的多余目录
rm -r $DOWNLOAD_DIR/azcopy_linux*