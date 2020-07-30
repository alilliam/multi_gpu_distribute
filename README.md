# multi_gpu_distribute
### 1.PyTorch DDP接口
#### 特点
1. 通信框架 Ring AllReduce
2. 多进程 一个结点一个进程
3. 支持数据并行和模型并行

#### 具体实现
1. Backend：第三方通信机制，进程间通信 NCCL
2. TCP initialization：联系其他机器上的进程，初始化方式：TCP

### 2. Horovod
分布式框架
http://yuchaoyuan.com/2019/12/09/Horovod/


#### 3.实验流程
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
#### Ref：
https://pytorch.org/tutorials/intermediate/dist_tuto.html
https://zhuanlan.zhihu.com/p/136372142
