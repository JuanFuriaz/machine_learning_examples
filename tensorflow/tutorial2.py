import tensorflow as tf
# use here to create or models
'''
1- input > weight > hidden layer 1 act func 1
2- > weights > hidden l 2 second act 
3- >weights > output lay
4-final training step do cost or loss function (compare results and classified data )using [cross entropy]
5- optimization fucntion > minimize by AdamOptimizer SGD,etc)
--> we start with backpropagation

REMEMBER: feed forward + backpro = epoch
'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)  # Allow our classificators to show results as 0 = [1,0,0,0,0,0,0,0,0]

#Defining  Model
#Neurones for each layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

batch_size = 100  # Interesting in case of having huge amounts of data

x = tf.placeholder('float', shape=[None, 784]) # Use this to define a matrix high x weight 28x28 but we just use the form as a line, usefull to throug error
y = tf.placeholder('float')

#
def neural_network_model (data):
    # biases to activate neurones if all data is zero it can be more interesting
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal( n_nodes_hl1)) }
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal( n_nodes_hl2)) }
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal(n_nodes_hl3))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal(n_classes))}
     # Modelling  r = (input_data * weights) + biases and
    l1 = tf.add (tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer ['biases'])
    l1= tf.nn.relu(l1) # Activition function
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add (tf.matmul(l3, output_layer['weights']) + output_layer ['biases'])
    return output