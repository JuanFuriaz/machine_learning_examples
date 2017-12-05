import tensorflow as tf
# use here to create or models
x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1,x2)

# Need to close sessions like writing opening docs
with tf.Session() as sess:
    output = sess.run(result)
    print(result)
