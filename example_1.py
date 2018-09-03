#!/usr/bin/python
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 50
BATCH_SIZE = 100 #batch大小
TRAINING_STEPS=10000  #训练轮数
#梯度下降算法参数
LEARNING_RATE_BASE = 0.8  #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
#正则化参数
REGULARIZATION_RATE = 0.0001  #正则化比例
#滑动平均参数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率

#神经网络前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
	if avg_class == None:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
		return tf.matmul(layer1, weights2) + biases2
	else:
		layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
		return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)
#训练过程
def train(mnist):
	#初始化变量
	x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
	y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
	#不使用滑动平均的计算前向传播结果
	y = inference(x, None, weights1, biases1, weights2, biases2)
	#定义滑动平均类
	global_step = tf.Variable(0, trainable=False)
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_average_op = variable_averages.apply(tf.trainable_variables())
	#使用滑动平均的结果
	average_y =  inference(x, variable_averages, weights1, biases1, weights2, biases2)
	#计算损失函数，交叉熵和正则化
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=y, labels=tf.argmax(y_,1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	regularization = regularizer(weights1) + regularizer(weights2)
	loss = cross_entropy_mean + regularization
	#设置学习率
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 	
	mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
#优化损失函数
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#反向传播更新参数，并使用滑动平均值
	train_op = tf.group(train_step, variables_average_op)
#计算正确率
	correct_prediction = tf.equal(tf.argmax(average_y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化会话
	with tf.Session() as sess:
		#初始变量
		tf.global_variables_initializer().run()
		validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
		test_feed = {x:mnist.test.images, y_:mnist.test.labels}
		#训练过程
		for i in range(TRAINING_STEPS):
			if i%1000 == 0:
				validation_acc = sess.run(accuracy, feed_dict=validate_feed)
				print("After %d training step(s), validation accuracy is %g" % (i, validation_acc))
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			print("shape:",ys.shape)
			sess.run(train_op, feed_dict={x:xs, y_:ys})
		#测试
		test_acc = sess.run(accuracy, feed_dict=test_feed)
		print("After %d training step(s), test accuracy is %g" % (TRAINING_STEPS, test_acc))
#主程序
def main(argv=None):
	mnist = input_data.read_data_sets("/home/djt/文档/MNIST/MNIST_data", one_hot=True)
	train(mnist)
if __name__ =='__main__':
	tf.app.run()
