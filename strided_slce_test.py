import tensorflow as tf

a = tf.constant(
[
             [[1, 1, 1], [2, 2, 2]],
             [[3, 3, 3], [4, 4, 4]],
             [[5, 5, 5], [6, 6, 6]]
]
    )
normal_a = tf.strided_slice(a, [0, 0, 0], [4, 4, 4], [1, 2, 1])
with tf.Session() as sess:
    print("feature map:")
    image = sess.run(a)
    print(image)
    print("normalized feature map:")
    normal = sess.run(normal_a)
    print(normal)