# 项目名称

## 项目概述

本项目是一个基于Python的深度学习项目，主要涉及卷积神经网络（CNN）在一维信号处理上的应用。
该项目旨在通过深度学习模型对电梯曳引系统的数据进行分析和处理，以实现对设备运行状态的监测和故障诊断。
## 项目结构
20240515电梯数据采集 该文件为原始采集数据（采集方式.jpg对齐有具体说明）
dataset 从原始数据切片的测试以及训练样本
model 为模型文件夹 包含CNN_1D CNN_AFF;
result 为结果存放文件
main.py 为入口函数文件

### 关键说明
这条命令可设定电脑运行为单核单线程
```python
import os
# os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```
采用anconda工具自行安装相关的依赖

