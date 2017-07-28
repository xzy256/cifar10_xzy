# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from prettytable import PrettyTable  # PrettyTable使用参看http://www.ithao123.cn/content-2560565.html

import cifar10
import numpy as np

FLAGS = tf.app.flags.FLAGS
# 设置存储模型训练结果的路径
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/xzy/cifar10_train_xzy',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('class_dir', '/home/xzy/cifar10-input/cifar-10-batches-bin/',
                           """存储文件batches.meta.txt的目录""")
tf.app.flags.DEFINE_string('test_file', 'dog.jpg', """测试用的图片，可以是绝对路径""")
IMAGE_SIZE = 24


def evaluate_images(images):  # 执行验证
    logits = cifar10.inference(images, batch_size=1)
    load_trained_model(logits=logits)


def load_trained_model(logits):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 从训练模型恢复数据
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

        # 下面两行是预测最有可能的分类
        # predict = tf.argmax(logits, 1)
        # output = predict.eval()

        # 从文件以字符串方式获取10个类标签，使用制表格分割
        cifar10_class = np.loadtxt(FLAGS.class_dir + "batches.meta.txt", str, delimiter='\t')
        # 预测最大的三个分类
        top_k_pred = tf.nn.top_k(logits, k=3)
        output = sess.run(top_k_pred)
        probability = np.array(output[0]).flatten()  # 取出概率值，将其展成一维数组
        index = np.array(output[1]).flatten()
        # 使用表格的方式显示
        tabel = PrettyTable(["index", "class", "probability"])
        tabel.align["index"] = "l"  # 左边界
        tabel.padding_width = 1  # padding值，默认为1
        for i in np.arange(index.size):
            tabel.add_row([index[i], cifar10_class[index[i]], probability[i]])
        print tabel


def img_read(filename):
    if not tf.gfile.Exists(filename):
        tf.logging.fatal('File does not exists %s', filename)
    image_data = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(filename),
                                                                   channels=3), dtype=tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    image = tf.image.resize_images(image_data, (height, width), method=ResizeMethod.BILINEAR)
    image = tf.expand_dims(image, -1)
    image = tf.reshape(image, (1, 24, 24, 3))
    return image


def main(argv=None):  # pylint: disable=unused-argument
    filename = FLAGS.test_file
    images = img_read(filename)
    evaluate_images(images)


if __name__ == '__main__':
    tf.app.run()
