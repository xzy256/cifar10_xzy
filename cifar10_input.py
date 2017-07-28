# -*- coding:utf-8 -*-
# @time:  2017.7.14
# @author:  xzy
# @call: distorted_inputs()/inputs()调用read_cifar10()读取数据,
# 再调用_generate_image_and_label_batch()产生样本
# ---------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 原始CIFAR10图片是32x32, 6w张彩色图片,10个类别,每个类别6k张,5w是训练图片
# 1w是验证数据,截取变成24x24
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """从二进制数据中获取一个样本的信息结构体(大小、数据、标签)
  执行N次实现N路并行
  Args:
    filename_queue: 由filenames组成的字符串队列

  Returns:
    一个包含下诉变量的对象:
      height: 行数
      width: 列数
      depth: 颜色通道数
      key: 描述样本的名字&记录数的一个字符串scalar张量
      label: 标识类别0..9.
      uint8image: 一个带图片信息[height, width, depth] uint8 张量
  """

    class CIFAR10Record(object):
        pass  # 保持句式结构,代表什么都不做

    result = CIFAR10Record()

    # 输入格式.
    label_bytes = 1  # 2 for CIFAR-100  ?????
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # 每个记录都由定长字节图片和紧随其后的标签组成
    record_bytes = label_bytes + image_bytes

    # 读取一个记录,从filename_queue中获取文件名,CIFAR10没有header 或者footer,对应的字节数使用默认值0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # 将一个字符串转换为一个uint8的张量
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 获取代表标签的第一字节分片,并做uint8-->int32转换,strided_slice可以参看本人博客
    # http://blog.csdn.net/banana1006034246/article/details/75092388
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 获取图片信息,并reshape[depth * height * width]-->[depth, height, width].
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],
                              [label_bytes + image_bytes]), [result.depth, result.height, result.width])
    # 使用tranpose将张量[depth, height, width]转置成[height, width, depth]
    # 关于transpose函数参看本人博客http://blog.csdn.net/banana1006034246/article/details/75126815
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """构造batch_size样本集

  Args:
    image: 3-D float32张量[height, width, 3] .
    label: 1-D int32张量
    min_queue_examples: int32, 维持在队列中最小的样本数
    batch_size: 每批次大小
    shuffle: 是否使用乱序队列

  Returns:
    images: Images. 4D 张量[batch_size, height, width, 3] 
    labels: Labels. 1D 张量 [batch_size]
  """
    # 使用shuffle_batch可以随机打乱输入,然后从样本队列读取 'batch_size' images + labels
    num_preprocess_threads = 16  # 开启read操作16线程
    if shuffle:  # 使用乱序
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # 收集image信息便于可视化
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])  # cifar10_train.py 中定义的batch_size=128


def distorted_inputs(data_dir, batch_size):
    """对CIFAR训练图片进行截取,翻转,亮度和对比度调整来得到更多的新数据

  Args:
    data_dir: 存CIFAR-10数据的目录.
    batch_size: 每批次图片数.

  Returns:
    images: Images. 4D  [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] 张量
    labels: Labels. 1D  [batch_size] 张量.
  """
    # 使用循环创建多个文件 data_batch_i.bin i=1~5
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 步骤1：随机截取一个以[高，宽]为大小的图矩阵
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # 步骤2：随机颠倒图片的左右。概率为50%
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # 步骤3：随机改变图片的亮度以及色彩对比
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # 对图片进行标准化,做的操作(x - mean) / max(stddev, 1.0/sqrt(image.NumElements()))
    float_image = tf.image.per_image_standardization(distorted_image)

    # 设置张量的形状
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4   # 在queue里有了不少于40%的数据的时候训练才能开始
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # 产生一个批次的images和labels
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """使用 Reader ops构建评估输入数据.

  Args:
    eval_data: 布尔类型, 暗示是train[训练]还是eval[验证]数据集.
    data_dir: CIFAR-10数据目录.
    batch_size: 每批次图片大小.

  Returns:
    images: Images. 4D [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3].
    labels: Labels. 1D [batch_size].
  """
    if not eval_data:  # 训练数据
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:  # 验证数据
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 生成一个先入先出的队列
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 截取并标准化, resized_image = (24, 24, 3)的tensor
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    print ("-------cifar10_input image--------", resized_image)
    float_image = tf.image.per_image_standardization(resized_image)
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size, shuffle=False)
