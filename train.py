import tensorflow as tf
import fcn8_vgg
import loss 
import numpy as np


vgg_fcn = fcn8_vgg.FCN8VGG()

num_classes = 20

training_epochs = 1000
batch_size      = 1
display_step    = 100
ntrain = 800
#global_step     = 

images_data = np.load('synth_train_images.npy',mmap_mode='r')
labels_data = np.load('synth_train_labels.npy',mmap_mode='r')

print(images_data.shape)

images = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
labels = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 20))

vgg_fcn.build(images, train=True, num_classes=num_classes, random_init_fc8=True)

loss = loss.loss(vgg_fcn.upscore32, labels, num_classes=num_classes)



with tf.name_scope('train'):
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(1e-6)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss) # global_step=global_step)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
	sess.run(init_op)
	for epoch in range(training_epochs):
	    num_batch = int(ntrain/batch_size)+1
	    # Loop over all batches
	    for i in range(num_batch): 
		randidx = np.random.randint(ntrain, size=batch_size)
		batch_xs = np.transpose(images_data[:, :, :, randidx], axes=[3,2,1,0])
		batch_ys = np.transpose(labels_data[:, :, :, randidx], axes=[3,2,1,0])                
		# Fit training using batch data
		sess.run(train_op, feed_dict={images:batch_xs, labels:batch_ys})
		#sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
		# Compute average loss
	
	    # Display logs per epoch step
	    if epoch % display_step == 0:
		print ("Epoch: %03d/%03d" % (epoch, training_epochs))
		train_acc = sess.run(loss, feed_dict={images:batch_xs, labels:batch_ys})
		print (" Training accuracy: %.3f" % (train_acc))
		test_acc = sess.run(loss, feed_dict={images:batch_xs, labels:batch_ys})
	

print ("Optimization Finished!")        

            
