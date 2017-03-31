from __future__ import division

import random
import tensorflow as tf

frame_width = 28
frame_height = 28
frame_chan = 4
sfilter_conv0 = 6
nfilter_conv0 = 16
sfilter_conv1 = 4
nfilter_conv1 = 32
nunits_fc = 256

class Q_agent():
    def __init__(self,dim_act,learning_rate,prefixe_name=""):
        self.state =  tf.placeholder(shape=[None,frame_width,frame_height,frame_chan],dtype=tf.float32)
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.learning_rate = learning_rate
        self.dim_action = dim_act
        self.Build_Qnet(prefixe_name)
        self.training_agent()

    # Build tf graph for function approximtor
    def Build_Qnet(self,prefixe_name=""):
        # Conv layer 0
        self.weights0 = tf.get_variable(prefixe_name+"weights0",
                                        [sfilter_conv0,sfilter_conv0,frame_chan,nfilter_conv0],
                                        initializer = tf.random_normal_initializer(stddev=0.01))
        self.bias0 = tf.get_variable(prefixe_name+"bias0",
                                        [nfilter_conv0],
                                        initializer = tf.constant_initializer(0.1))
        conv0 = tf.nn.conv2d(self.state, self.weights0, strides=[1, 2, 2, 1], padding='SAME')
        conv0_relu = tf.nn.relu(conv0 + self.bias0)
        # Conv layer 1
        self.weights1 = tf.get_variable(prefixe_name+"weights1",
                                        [sfilter_conv1,sfilter_conv1,nfilter_conv0,nfilter_conv1],
                                        initializer = tf.random_normal_initializer(stddev=0.01))
        self.bias1 = tf.get_variable(prefixe_name+"bias1",
                                        [nfilter_conv1],
                                        initializer = tf.constant_initializer(0.1))
        conv1 = tf.nn.conv2d(conv0_relu, self.weights1, strides=[1, 2, 2, 1], padding='SAME')
        conv1_relu = tf.nn.relu(conv1 + self.bias1)
        # Flatten conv1
        conv1_output_shpe = conv1_relu.get_shape().as_list()
        conv1_flat_dim = conv1_output_shpe[1]*conv1_output_shpe[2]*conv1_output_shpe[3]
        conv1_flat = tf.reshape(conv1_relu,[-1,conv1_flat_dim])
        # FC layer
        self.weights2 = tf.get_variable(prefixe_name+"weights2",
                                        [conv1_flat_dim,nunits_fc],
                                        initializer = tf.random_normal_initializer(stddev=0.01))
        self.bias2 = tf.get_variable(prefixe_name+"bias2",
                                        [nunits_fc],
                                        initializer = tf.constant_initializer(0.1))
        fc_relu = tf.nn.relu(tf.matmul(conv1_flat,self.weights2)+self.bias2)
        # linear layer
        self.weights3 = tf.get_variable(prefixe_name+"weights3",
                                        [nunits_fc,self.dim_action],
                                        initializer = tf.random_normal_initializer(stddev=0.01))
        self.bias3 = tf.get_variable(prefixe_name+"bias3",
                                        [self.dim_action],
                                        initializer = tf.constant_initializer(0.1))
        self.Qout = tf.matmul(fc_relu,self.weights3) + self.bias3

    def training_agent(self):
        # Prediction of action given Qout
        self.predict = tf.argmax(self.Qout,1)
        # Approximate q(s,a) with Q(s)[a]
        self.actions_onehot = tf.one_hot(self.actions,self.dim_action,dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),axis=1)
        # Build loss (L2) and update function using td difference
        td_error = tf.square(self.targetQ - self.Q) / 2
        loss = tf.reduce_mean(td_error)
        self.l = loss
        trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        #trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updateModel = trainer.minimize(loss)

class experience_replay():
    def __init__(self, replay_size = 1000):
        self.replay = []
        self.replay_size = replay_size

    def add(self,experience):
        if len(self.replay) + 1 >= self.replay_size:
            self.replay[0:(1+len(self.replay))-self.replay_size] = []
        self.replay.append(experience)

    def sample(self,batch_size):
        return random.sample(self.replay,batch_size)
