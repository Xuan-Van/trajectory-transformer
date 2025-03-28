# 检查是否已经登录
if keyctl show 2>&1 | grep -q "workaroundSession"; then
    echo "Already logged in"
else
    echo "Logging in with tenant id:" ${AZURE_TENANT_ID}
    keyctl session workaroundSession
    ./bin/azcopy login --tenant-id=$AZURE_TENANT_ID
fi

# 设置日志基础路径
export LOGBASE=defaults

# 从 AZURE_STORAGE_CONNECTION_STRING 中提取存储账户名称
export AZURE_STORAGE_ACCOUNT=$(echo $AZURE_STORAGE_CONNECTION_STRING | grep -o -P '(?<=AccountName=).*(?=;AccountKey)')

# 输出同步信息
echo "Syncing from" ${AZURE_STORAGE_ACCOUNT}"/"${AZURE_STORAGE_CONTAINER}"/"${LOGBASE}

# 使用 azcopy 同步日志文件
./bin/azcopy sync "https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${AZURE_STORAGE_CONTAINER}/${LOGBASE}/logs" "logs/" --recursive