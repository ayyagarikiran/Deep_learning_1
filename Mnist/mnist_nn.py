from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

#parameters
learning_rate=0.1
num_steps=1000
batch_size=3000
display_step=100

n_hidden_1=256
n_hidden_2=256
n_hidden_3=256
num_input=784
num_classes=10

# tf graph input
X=tf.placeholder("float",[None,num_input])
Y=tf.placeholder("float",[None,num_classes])

# weights and biases
weights={
	'h1': tf.Variable(tf.random_normal([num_input,n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
	'out': tf.Variable(tf.random_normal([n_hidden_3,num_classes]))
}

biases={
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}


#create model

def neural_net(x):
	layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])

	#layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

	layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
	layer_3=tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
	out_layer=tf.matmul(layer_3,weights['out'])+biases['out']
	#out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

	return out_layer
logits=neural_net(X)
prediction=tf.nn.softmax(logits)

#define loss and optimizer
loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss_op)


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()	

with tf.Session() as sess:
	sess.run(init)

	for step in range(1,num_steps+1):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		#run optimizer and back prop
		sess.run(train_op,feed_dict={X: batch_x, Y: batch_y})

		if step % display_step == 0 or step == 1:
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})

			print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
	print("Optimization Finished!")

	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))

