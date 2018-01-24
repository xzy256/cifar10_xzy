
@ author :xzy  <br/>
@ email: xuezhiyou001@foxmail.com <br/>
@ time: 2017.7 <br/>


CIFAR-10 是一个常用于图像识别的一个benchmark例子，关于它的介绍可以参考[The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)
本项目是针对官方例子改写，并对注释加入了自己的理解，使用训练模型能实现对单张图片的验证
<pre>
文件说明：
cifar10.py : 构建CIFAR-10神经网络操作，包括inference、loss、train、evaluation
cifar10_eval.py : 运行验证集模型
cifar10_eval_1photo.py : 原创代码，实现单张图片的验证
cifar10_train.py : 运行训练集模型
cifar10_input.py : 输入数据，包括数据集的大小配置、神经层的配置、filter卷积核的参数配置等
cifar10_multi_gpu.py :本人的机器只能跑CPU版本的TF, 所以对于多GPU代码并未翻译和了解
photo_to_lmdb.py : 将图片转化成lmdb格式，暂时未使用
<test>
transpose_test.py : 针对理解矩阵转置函数而写的一个简单例子
lrn.py ： 针对理解局部响应标准化而写的一个简单例子
strided_slice.py ： 针对理解跨步长分片的一个简单例子


最终的效果：
+-------+-------+-------------+
| index | class | probability |
+-------+-------+-------------+
| 3     |  cat  |   0.530751  |
| 5     |  dog  |   0.491245  |
| 2     |  bird |   0.139152  |
+-------+-------+-------------+
</pre>

注意：<br/>
1.事先安装prettytable<br/>
`sudo pip install prettytable`

2.先执行train，再执行验证<br/>
`python cifar10_train.py
python cifar10_eval_1photo.py`


3.图片的大小为24x24不能随便的修改，官方的数据集就是24x24的彩色照片，
改变它会引起类似下面的错误
>InvalidArgumentError (see above for traceback): Assign requires shapes of both tensors to match. lhs shape= [4096,384] rhs shape= [2304,384]
>	 [[Node: save/Assign_5 = Assign[T=DT_FLOAT, _class=["loc:@local3/weights"], use_locking=true, validate_shape=true, _device="/job:localhost/replica:0/task:0/de>vice:CPU:0"](local3/weights, save/RestoreV2_5)]]


关于验证单张图片步骤，可以访问我的博客链接[Tensorflow如何使用自己cifar10训练模型检测一张任意的图片](http://blog.csdn.net/banana1006034246/article/details/76239147)
