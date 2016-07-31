import tensorflow as tf
import fcn8_vgg
import loss 

vgg_fcn = fcn8_vgg.FCN8VGG()

num_classes = 20

labels = tf.placeholder()

vgg_fcn.build(images, train=True, num_classes=num_classes, random_init_fc8=True)

loss = loss.loss(vgg_fcn.pred_up, labels, num_classes=num_classes)

with tf.name_scope('train'):
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(1e-6)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    with tf.Session() as sess:
        for i in range(1000):
            image_batch =
            label_batch =
            sess.run(train_op, feed_dict={images:image_batch, labels:label_batch})