import tensorflow as tf

a = tf.constant([
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0],
     [8.0, 7.0, 6.0, 5.0],
     [4.0, 3.0, 2.0, 1.0]],
    [[4.0, 3.0, 2.0, 1.0],
     [8.0, 7.0, 6.0, 5.0],
     [1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]]
])
# reshape 1批次  2x2x8的feature map
a = tf.reshape(a, [1, 2, 2, 8])

normal_a = tf.nn.lrn(a, 2, 0, 1, 1)
with tf.Session() as sess:
    print("feature map:")
    image = sess.run(a)
    print(image)
    print("normalized feature map:")
    normal = sess.run(normal_a)
    print(normal)