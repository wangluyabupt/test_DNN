from sympy.printing.tests.test_tensorflow import tf

global_step=tf.Variable(0,trainable=False)
print(global_step)
with tf.Session() as sess:
    sess.run(global_step.initializer)     #运行变量的initializer。调用op之前，所有变量都应被显式地初始化过。
    sess.run(global_step)     #查看v的值，结果是：array([1, 2, 3])
    print(sess.run(global_step))

v = tf.Variable([1,2,3])   #创建变量v，为一个array
print(v)  #查看v的shape，不是v的值。结果是： <tf.Variable 'Variable:0' shape=(3,) dtype=int32_ref>
with tf.Session() as sess:
    sess.run(v.initializer)     #运行变量的initializer。调用op之前，所有变量都应被显式地初始化过。
    sess.run(v)     #查看v的值，结果是：array([1, 2, 3])
    print(sess.run(v))

W1=tf.Variable(0,[400])
print(W1)
with tf.Session() as sess:
    sess.run(W1.initializer)     #运行变量的initializer。调用op之前，所有变量都应被显式地初始化过。
    sess.run(W1)     #查看v的值，结果是：array([1, 2, 3])
    print(sess.run(W1))