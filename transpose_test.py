import tensorflow as tf
import numpy as np

input_x = [
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]

]

result = tf.transpose(input_x, perm=[0, 2, 1])
print(result)
with tf.Session() as sess:
    print(sess.run(result))
