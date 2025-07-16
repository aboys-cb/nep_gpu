# 声明
代码从GPUMD仓库复制并修改的版本
https://github.com/brucefan1983/GPUMD
目前支持正常的nep模型，其他我用不到 没处理。
# 安装
使用cuda 直接make nep 即可
# 使用
nep nep.txt structure.xyz [batch_size]

如果爆显存了可以使用batch_size 指定

输出的结果可以直接拖进neptrainkit
