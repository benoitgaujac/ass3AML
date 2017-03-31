from __future__ import division

import random
import tensorflow as tf

dim_act = 2
dim_state = 4

class Q_agent():
    def __init__(self,approx,learning_rate,nunits=100,prefixe_name=""):
        self.state =  tf.placeholder(shape=[None,4],dtype=tf.float32)
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.approximator = approx
        self.learning_rate = learning_rate
        self.nunits = nunits
        #self.discount_factor = discount_factor
        self.Build_Qnet(prefixe_name)
        self.training_agent()

    # Build tf graph for function approximtor
    def Build_Qnet(self,prefixe_name=""):
        if self.approximator=="linear":
            self.weights0 = tf.get_variable("weights0", [dim_state, dim_act], initializer = tf.random_normal_initializer(stddev=0.01))
            self.bias0 = tf.get_variable("bias0", [dim_act], initializer = tf.constant_initializer(0.1))
            self.Qout = tf.matmul(self.state, self.weights0) + self.bias0
        elif self.approximator=="hidden":
            self.weights0 = tf.get_variable(prefixe_name+"weights0", [dim_state, self.nunits], initializer = tf.random_normal_initializer(stddev=0.01))
            self.bias0 = tf.get_variable(prefixe_name+"bias0", [self.nunits], initializer = tf.constant_initializer(0.1))
            self.weights1 = tf.get_variable(prefixe_name+"weights1", [self.nunits, dim_act], initializer = tf.random_normal_initializer(stddev=0.01))
            self.bias1 = tf.get_variable(prefixe_name+"bias1", [dim_act], initializer = tf.constant_initializer(0.1))
            self.Qout = tf.matmul(tf.nn.relu(tf.matmul(self.state, self.weights0) + self.bias0),self.weights1) + self.bias1

    def training_agent(self):
        # Prediction of action given Qout
        self.predict = tf.argmax(self.Qout,1)
        # Approximate q(s,a) with Q(s)[a]
        self.actions_onehot = tf.one_hot(self.actions,dim_act,dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),axis=1)
        # Build loss (L2) and update function using td difference
        td_error = tf.square(self.targetQ - self.Q) / 2
        loss = tf.reduce_mean(td_error)
        self.l = loss
        #trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updateModel = trainer.minimize(loss)

class experience_replay():
    def __init__(self, replay_size = 1000):
        self.replay = []
        self.replay_size = replay_size

    def add(self,experience):
        if len(self.replay) > self.replay_size:
            self.replay[0:(len(experience)+len(self.replay))-self.replay_size] = []
        self.replay.append(experience)

    def sample(self,batch_size):
        return random.sample(self.replay,batch_size)
