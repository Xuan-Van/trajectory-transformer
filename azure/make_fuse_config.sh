# 从 AZURE_STORAGE_CONNECTION_STRING 中提取存储账户名称
export AZURE_STORAGE_ACCOUNT=$(echo $AZURE_STORAGE_CONNECTION_STRING | grep -o -P '(?<=AccountName=).*(?=;AccountKey)')

# 从 AZURE_STORAGE_CONNECTION_STRING 中提取存储账户密钥
export AZURE_STORAGE_KEY=$(echo $AZURE_STORAGE_CONNECTION_STRING | grep -o -P '(?<=AccountKey=).*(?=;EndpointSuffix)')

# 创建配置文件目录（如果不存在）
mkdir -p ./azure

# 将提取的账户名称、账户密钥和容器名称写入配置文件
echo "accountName" ${AZURE_STORAGE_ACCOUNT} > ./azure/fuse.cfg
echo "accountKey" ${AZURE_STORAGE_KEY} >> ./azure/fuse.cfg
echo "containerName" ${AZURE_STORAGE_CONTAINER} >> ./azure/fuse.cfg