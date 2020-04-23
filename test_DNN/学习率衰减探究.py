import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# learning_rate = 0.1
# decay_rate = 0.96
# global_steps = 1000
# decay_steps = 100
#
# global_ = tf.Variable(tf.constant(0))
# c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
# d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)
#
# T_C = []
# F_D = []
#
# with tf.Session() as sess:
#     for i in range(global_steps):
#         T_c = sess.run(c, feed_dict={global_: i})
#         T_C.append(T_c)
#         F_d = sess.run(d, feed_dict={global_: i})
#         F_D.append(F_d)
#
# plt.figure(1)
# plt.plot(range(global_steps), F_D, 'r-')
# plt.plot(range(global_steps), T_C, 'b-')
#
# plt.show()


global_step=tf.Variable(0.,trainable=False)
initial_learning_rate=0.001
learning_rate=tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps=10,decay_rate=0.96,staircase=True)
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    for i in range(100):
        print(sess.run(learning_rate,feed_dict={global_step:i}))