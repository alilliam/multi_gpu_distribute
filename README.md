# multi_gpu_distribute
### PyTorch DDP接口
1. 实验原理 Ring AllReduce
2. 由于多机环境不可控因素影响，可能导致实验不成功，所以使用Docker。
3. 多机跑实验

### Horovod
分布式框架
http://yuchaoyuan.com/2019/12/09/Horovod/


#### 搭建Docker环境
以下是使用PyTorch DDP接口实验的流程
1. Ububtu 安装Docker(docker-ce 和docker-ce-cli)
https://blog.csdn.net/weixin_30414305/article/details/101669345

2. Ubuntu 安装nvidia-docker，这样docker环境就可以检测到主机的显卡了
https://github.com/NVIDIA/nvidia-docker

3. 修改Docker源，这样拉取镜像更快。

4. 拉取镜像， 可以使用Dockerfile选择自己想要的配置，也可以在[DockerHub][https://hub.docker.com/r/horovod/horovod/tags]中直接下载预先构建好的镜像。

5. 在各主机上启用容器 (配置网络为主机网络, 配置共享文件夹)
```bash
#修改your-image-id为新建镜像id
sudo docker run -it --gpus all --privileged --network=host -v /mnt/share/ssh:/root/.ssh your-image-id
```

6. 多机跑实验
* 在各主机容器目标目录下放置相同的测试代码和测试数据集
* 运行实验
```bash
#primary node
#修改your-dataPath为数据集路径
#修改ip-of-primary为主结点ip
CUDA_VISIBLE_DEVICES=0 python mnist.py -a resnet101 --dist url 'tcp://ip-of-primary:8001' --dist-backend 'nccl' --world-size 2 --rank 0 your-dataPath
```

```bash
#secondary node
#修改your-dataPath为数据集路径
#修改ip-of-primary为主结点ip
#修改rank
CUDA_VISIBLE_DEVICES=0 python mnist.py -a resnet101 --dist url 'tcp://ip-of-primary:8001' --dist-backend 'nccl' --world-size 2 --rank 1 your-dataPath
```
