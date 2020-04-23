import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# x=np.array(pd.read_csv('x_data.csv',header=None))
# x=x.astype(np.float32)
# y=np.array(pd.read_csv('y_data.csv',header=None))
# y=y.astype(np.float32)
input_size=182
num_classes=1
num_hidden_units1=300
num_hidden_units2=400
#num_hidden_units3=400

M=10000
epoch=700


batch_size=200
i=0

global_step=tf.Variable(0,trainable=False)#下面会通过session参数占位符符值
initial_learning_rate=0.001
learning_rate=tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps=100,decay_rate=0.97,staircase=True)
#上面这个exponential_decay()是有一个带指数的计算公式的
    # learning_rate=tf.train.polynomial_decay(initial_learning_rate,global_step,decay_steps=100,decay_rate=0.97,staircase=True)

# learning_rate=tf.constant(0.001)
x=tf.placeholder(tf.float32,shape=[None,input_size])
y=tf.placeholder(tf.float32,shape=[None,num_classes])
#keep_prob=tf.placeholder(tf.float32)




W1=tf.Variable((tf.random_normal([input_size,num_hidden_units1],stddev=0.1)))#从输入层到hidden层
B1=tf.Variable(tf.constant(0.0),[num_hidden_units1])
W2=tf.Variable((tf.random_normal([num_hidden_units1,num_hidden_units2],stddev=0.1)))
B2=tf.Variable(tf.constant(0.0),[num_hidden_units2])
# W3=tf.Variable((tf.random_normal([num_hidden_units2,num_hidden_units3],stddev=0.1)))
# B3=tf.Variable(tf.constant(0.0),[num_hidden_units3])
W4=tf.Variable((tf.random_normal([num_hidden_units2,num_classes],stddev=0.1)))
B4=tf.Variable(tf.constant(0.0),[num_classes])

hidden_layer_output1=tf.matmul(x,W1)+B1
hidden_layer_output1=tf.nn.relu(hidden_layer_output1)
hidden_layer_output2=tf.matmul(hidden_layer_output1,W2)+B2
hidden_layer_output2=tf.nn.relu(hidden_layer_output2)
#hidden_layer_output2_drop=tf.nn.dropout(hidden_layer_output2,keep_prob)
# hidden_layer_output3=tf.matmul(hidden_layer_output2,W3)+B3
# hidden_layer_output3=tf.nn.relu(hidden_layer_output3)
final_output=tf.matmul(hidden_layer_output2,W4)+B4
final_output=tf.nn.relu(final_output)


#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_output,labels=y),name='loss')
loss=tf.reduce_mean(tf.square(final_output-y))
#采用梯度下降来优化参数
#optimizer=tf.train.GradientDescentOptimizer(0.00001)
optimizer=tf.train.AdamOptimizer(learning_rate)
    # optimizer=tf.keras.optimizers(learning_rate)
#训练
train = optimizer.minimize(loss,name='train')





#清空两个文件
f=open('y_hat_data.csv','w')
f.close()
f1 = open('final_output_data.csv', 'w')
f1.close()



def get_next(filename,batch_size,M,i):#batch_size=200；M=10000
    if batch_size*i>=M:
        i=0
    x = np.array(pd.read_csv(filename, skiprows=batch_size * i, nrows=batch_size, header=None))#每次读进一个batch
    x = x.astype(np.float32)#数值类型的转换
    return x,i+1

T_A = []
T_B = []
T_C = []


with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for m in range(int(epoch*(M/batch_size))):# 700*（10000/200）=700*50=35000
    #for m in range(3):
        batch_x,i=get_next('x_data.csv',batch_size,M,i)
        batch_y,i=get_next('y_data.csv',batch_size,M,i)

        sess.run([learning_rate,train],feed_dict={global_step:m,x:batch_x,y:batch_y})

        if m%100==0:
            print('loss:',sess.run(loss,feed_dict={x:batch_x,y:batch_y,global_step:m}))
            Loss=sess.run(loss,feed_dict={x:batch_x,y:batch_y,global_step:m})
            T_A.append(Loss)
            y_hat = tf.round(sess.run(final_output, feed_dict={x: batch_x, y: batch_y}))
                #print(sess.run(y_hat))
            correct_prediction = tf.cast(tf.equal(y_hat, batch_y), 'float')
                #print(sess.run(correct_prediction))
            accuracy = tf.reduce_mean(correct_prediction)
            print("train_accuracy",sess.run(accuracy))
            TrainAccuracy=sess.run(accuracy)
            T_B.append(TrainAccuracy)
        if (m*batch_size)%(M*10)==0:
            print("%d th epoch in total %d epoches"%((m*batch_size)/(M),epoch))
            print("学习率：",sess.run(learning_rate,feed_dict={global_step:m}))
            LearningRate=sess.run(learning_rate, feed_dict={global_step: m})
            T_C.append(LearningRate)
        if m==int(epoch*(M/batch_size))-1:#最后一次的时候，第35000次的时候
        #if m==3-1:
            #print('final_output:',sess.run(final_output,feed_dict={x:batch_x,y:batch_y}))
            print(sess.run(final_output,feed_dict={x:batch_x,y:batch_y}))
            final_output_hat=tf.round(sess.run(final_output,feed_dict={x:batch_x,y:batch_y}))
            print('final_output_hat:\n',sess.run(final_output_hat))
            print('batch_y:\n',batch_y)
            correct_prediction = tf.cast(tf.equal(final_output_hat, batch_y), 'float')
            accuracy = tf.reduce_mean(correct_prediction)
            print(sess.run(accuracy))

# plt.figure(1)
# plt.plot(range(int((epoch*(M/batch_size))/100)), T_B, 'b-')
# # plt.plot(range(70), T_C, 'y-')
# plt.show()
# plt.figure(2)
# plt.plot(range(int((epoch*(M/batch_size))/100)), T_A, 'r-')
# plt.show()
fig,axes=plt.subplots(1,2)
ax1=axes[0]
ax2=axes[1]
ax1.plot(range(350),T_A)
ax2.plot(range(350),T_B)
plt.show()
    # test_x=np.array(pd.read_csv('test_x_data.csv',header=None))
    # test_x=test_x.astype(np.float32)
    # test_y=np.array(pd.read_csv('test_y_data.csv',header=None))
    # test_y=test_y.astype(np.float32)
    #
    #
    # y_hat=tf.round(sess.run(final_output,feed_dict={x:test_x,y:test_y}))
    # print(sess.run(y_hat))
    # correct_prediction = tf.cast(tf.equal(y_hat, test_y),'float')
    # print(sess.run(correct_prediction))
    # accuracy = tf.reduce_mean(correct_prediction)
    # print(sess.run(accuracy))


    # for n in  final_output.eval():
    #     final_output_data = pd.DataFrame(n, columns=None)
    #     final_output_data.to_csv('final_output_data.csv', mode="a+", index=False, header=None)
    # for h in y_hat.eval():
    #     y_hat_data=pd.DataFrame(h,columns=None)
    #     y_hat_data.to_csv('y_hat_data.csv',mode="a+", index=False, header=None)