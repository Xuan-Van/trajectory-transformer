# 设置下载路径
export DOWNLOAD_PATH=logs

# 如果下载路径不存在，则创建该路径
[ ! -d ${DOWNLOAD_PATH} ] && mkdir ${DOWNLOAD_PATH}

# 下载预训练模型（16个数据集）
# 数据集包括：{halfcheetah, hopper, walker2d, ant}
# 每个数据集有以下变体：{expert-v2, medium-expert-v2, medium-v2, medium-replay-v2}

# 下载预训练模型压缩包
wget https://www.dropbox.com/sh/r09lkdoj66kx43w/AACbXjMhcI6YNsn1qU4LParja?dl=1 -O dropbox_models.zip

# 解压缩预训练模型到指定路径
unzip dropbox_models.zip -d ${DOWNLOAD_PATH}

# 删除下载的压缩包
rm dropbox_models.zip

# 下载每个预训练模型的15个计划
wget https://www.dropbox.com/s/5sn79ep79yo22kv/pretrained-plans.tar?dl=1 -O dropbox_plans.tar

# 解压缩计划文件
tar -xvf dropbox_plans.tar

# 将解压后的计划文件复制到下载路径
cp -r pretrained-plans/* ${DOWNLOAD_PATH}

# 删除解压后的计划文件目录
rm -r pretrained-plans

# 删除下载的计划文件压缩包
rm dropbox_plans.tar