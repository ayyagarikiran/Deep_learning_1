
from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
num_steps = 500
batch_size = 256
display_step = 10

num_input=784
num_classes=10
dropout=0.9

# tf graph input
X=tf.placeholder("float",[None,num_input])
Y=tf.placeholder("float",[None,num_classes])
keep_prob = tf.placeholder(tf.float32) 

def conv2DD(x,w,b,strides=1):
	x=tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
	x=tf.nn.bias_add(x,b)
	return tf.nn.relu(x)
def maxpool2d(x,k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
#creating model
def conv_net(x,weights,biases,dropout):
	x=tf.reshape(x,shape=[-1,28,28,1])

	#convolutional layer
	conv1=conv2DD(x,weights['wc1'],biases['bc1'])
	conv1 = maxpool2d(conv1, k=2)

	conv2 = conv2DD(conv1, weights['wc2'], biases['bc2'])
	conv2= maxpool2d(conv2, k=2)


	#Fully connected layers
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)

	fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
	fc2 = tf.nn.relu(fc2)

	fc2=tf.nn.dropout(fc2,dropout)
	out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
	return out

weights={
	 # 5x5 conv, 1 input, 32 outputs
	'wc1' : tf.Variable(tf.random_normal([5,5,1,32])),
	 # 5x5 conv, 32 inputs, 64 outputs
	'wc2' : tf.Variable(tf.random_normal([5,5,32,64])),


	 # fully connected, 7*7*64 inputs, 1024 outputs
	'wd1' : tf.Variable(tf.random_normal([7*7*64,1024])),
	'wd2' : tf.Variable(tf.random_normal([1024,2048])),
	# 1024 inputs, 10 outputs (class prediction)
	'out' : tf.Variable(tf.random_normal([2048,num_classes]))

}	

biases={
	'bc1' : tf.Variable(tf.random_normal([32])),
	'bc2' : tf.Variable(tf.random_normal([64])),
	'bd1' : tf.Variable(tf.random_normal([1024])),
	'bd2' : tf.Variable(tf.random_normal([2048])),
	'out' : tf.Variable(tf.random_normal([num_classes]))
}

logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()	

with tf.Session() as sess:
	sess.run(init)

	for step in range(1,num_steps+1):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		#run optimizer and back prop
		sess.run(train_op,feed_dict={X: batch_x, Y: batch_y,keep_prob: 0.9})

		if step % display_step == 0 or step == 1:
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,keep_prob: 1.0})
			print("Validation Accuracy:", \
        		sess.run(accuracy, feed_dict={X: mnist.validation.images,
                                      Y: mnist.validation.labels,keep_prob: 1.0}))

			print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
	print("Optimization Finished!")

	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels,keep_prob: 1.0}))




