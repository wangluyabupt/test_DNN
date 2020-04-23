import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#产生数据
M=np.random.randint(1,24,(12,8))
dt_x=pd.DataFrame(M,columns=None)
dt_x.to_csv('x_data.csv',index=False,header=None)

N=np.random.randint(1,24,(12,8))
dt_y=pd.DataFrame(N,columns=None)
dt_y.to_csv('y_data.csv',index=False,header=None)

x_data=np.array(pd.read_csv('x_data.csv',header=None))
x_data=tf.convert_to_tensor(tf.cast(x_data,tf.float32))
y_data=np.array(pd.read_csv('y_data.csv',header=None))
y_data=tf.convert_to_tensor(tf.cast(y_data,tf.float32))
print(x_data.shape)

#输入featrue数目
num_features=x_data.shape[1]
#样本数目
M=x_data.shape[0]

#设置超参数
num_hidden_units=5
training_iterations=20
learning_rate=0.1

x=tf.placeholder(tf.float32,shape=[None,num_features])
y=tf.placeholder(tf.float32,shape=[None,num_features])

#神经网络（单层）
#参数初始化
W1=tf.Variable(tf.random_normal([num_features,num_hidden_units],stddev=0.1))#从输入层到hidden层
B1=tf.Variable(tf.constant(0.1),[num_hidden_units])

y_layer_output=tf.matmul(x,W1)+B1
y_layer_output=tf.nn.relu(y_layer_output)

#y=np.round(y_layer_output)

#损失函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_data),name='loss')

#梯度下降
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#训练
train=optimizer.minimize(loss,name='train')


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for i in range(training_iterations):
    sess.run([train,loss],feed_dict={x:x_data,y:y_data})
    print("W=", sess.run(W1), "b=", sess.run(B1), "loss=", sess.run(loss))