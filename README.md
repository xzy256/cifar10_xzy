@ author :xzy
@ email: xuezhiyou001@foxmail.com
@ time: 2017.7
CIFAR-10 是一个常用于图像识别的一个benchmark例子，关于它的介绍可以参考
http://www.cs.toronto.edu/~kriz/cifar.html
本项目是针对官方例子改写，并对注释加入了自己的理解，使用训练模型能实现对单张图片的验证

文件说明：
cifar10.py : 构建CIFAR-10神经网络操作，包括inference、loss、train、evaluation
cifar10_eval.py : 运行验证集模型
cifar10_eval_1photo.py : 原创代码，实现单张图片的验证
cifar10_train.py : 运行训练集模型
cifar10_input.py : 输入数据，包括数据集的大小配置、神经层的配置、filter卷积核的参数配置等
cifar10_multi_gpu.py :本人的机器只能跑CPU版本的TF, 所以对于多GPU代码并不理解
photo_to_lmdb.py : 将图片转化成lmdb格式，暂时未使用
<test>
transpose_test.py : 针对理解矩阵转置函数而写的一个简单例子
lrn.py ： 针对局部响应标准化而写的一个建安例子
strided_slice.py ： 针对跨步长分片的一个例子


