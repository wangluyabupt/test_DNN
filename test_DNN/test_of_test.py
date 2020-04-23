import tensorflow as tf
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# a=tf.Variable([[1,2],[2,3]])
# b=tf.Variable([[1,1],[1,1]])
a=tf.Variable(tf.constant(1))
b=tf.Variable(tf.constant(1))
# test1=np.array(pd.read_csv('test_y_data.csv',header=None))
# test1=test1.astype(np.float32)
# test2=np.array(pd.read_csv('test_y_data1.csv',header=None))
# test2=test2.astype(np.float32)
#
# accuracy=tf.cast(tf.equal(test1,test2),'float')

# init=tf.global_variables_initializer()
# sess=tf.Session()
# sess.run(init)
# print(sess.run(tf.multiply(a,b)))
# print(sess.run(tf.multiply(a,b),feed_dict={a:3,b:5}))
# print(sess.run(accuracy))
# print(sess.run(tf.reduce_mean(accuracy)))


a=tf.constant()
b=tf.round(a)

with tf .Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
