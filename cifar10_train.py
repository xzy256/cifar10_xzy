# -*- coding:utf-8 -*-
# @time: 2017.7.17
# @author: xzy
# 训练数据
# -----------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'cifar10_train_xzy', """Directory where to write
                                                               event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100, """How often to log results to the console.""")


def train():
    """ 多步训练CIFAR-10数据集 """
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # 指定 CPU:0 进行数据的变形处理
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        # inference 前向预测，定义在cifar10.py文件
        logits = cifar10.inference(images)

        # loss.
        loss = cifar10.loss(logits, labels)

        # train
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """记录 loss 和 runtime."""

            def __init__(self):
                self._start_time = time.time()
                self._step = -1

            def begin(self):
                pass

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # 将loss 操作加到Session.run()调用

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)'
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),  # 在运行指定步长后停止
                       tf.train.NanTensorHook(loss),  # 监视loss，如果loss==NaN,停止训练
                       _LoggerHook()],
                # log_device_placement是否打印设备分配日志
                config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()  # 调用cifar10.py中的方法, 准备数据
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()  # 开始训练


if __name__ == '__main__':
    tf.app.run()
