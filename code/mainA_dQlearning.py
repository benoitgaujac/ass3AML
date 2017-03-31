from __future__ import division

import os
import sys
import time
import pdb

import numpy as np
import csv
import pandas as pd
import tensorflow as tf

import gym
import RLagent

env = gym.make('CartPole-v0')

max_len_episode = 300 # maximum lenth allowed
df = 0.99 # discount factor
n_episodes = 2001 # number of training episodes
n_test_episode = 5 # number of testing episodes
test_frequency = 20 # test frequency of greedy policy
startE = 0.5 # starting exploration probability
endE = 0.01 # ending exploration probability
decay = 2000 # decay for the exploration probability
stepDrop = (startE - endE)/decay # decay step size of the exploration policy
batch_size = 128 # batch size for experience replay buffer
exp_replay_buffer_size = 1024 # size of the experience replay buffer
n_runs = 10 # number of runs for average performances
log_frequency = 1000 # frequencyof login

double_Q = {"name":"doubleQ","nunits": 100,"lr":0.001,"exp_replay_buffer":True}

models = [double_Q,]

MODEL_DIR = "../models"
if not tf.gfile.Exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
PERF_DIR = "../Perfs"
if not tf.gfile.Exists(PERF_DIR):
    os.makedirs(PERF_DIR)
SUB_DIR = "onlineQlearning"

######################################## utils functions ########################################
def f_reward(end_of_episode):
    r = 0
    if end_of_episode:
        r = -1
    return r

def update_target_net(variables,sess):
    for var in variables:
        sess.run(var)

def save_csv(to_save,columns_list,dst_path,idx_frq=1):
    if os.path.isfile(dst_path):
        os.remove(dst_path)
    if len(to_save)!=len(columns_list):
        raise Exception("Error in formating files to save")
    episodes_idx = np.arange(0,n_episodes,idx_frq)
    file_tosave = pd.DataFrame(episodes_idx, columns=["episode"])
    for i in range(1,len(to_save)+1):
        file_tosave.insert(i,columns_list[i-1] , np.array(to_save[i-1]))
    file_tosave.to_csv(dst_path, index=False)
    print("CSV file successfully created.")

######################################## Main online Qlearning ########################################
def online_Qlearning(model,fct_approximator="hidden",save_model=False):
    # Reset tf graph
    tf.reset_default_graph()
    # Build agent A
    QagentA = RLagent.Q_agent(fct_approximator,model["lr"],model["nunits"],"A_")
    # Build agent B
    QagentB = RLagent.Q_agent(fct_approximator,model["lr"],model["nunits"],"B_")
    # Build experience replay
    exp_replay = RLagent.experience_replay(exp_replay_buffer_size)
    # Initializer tf variables
    init = tf.global_variables_initializer()
    # Saver
    saver = tf.train.Saver()
    # Initialize tf session
    with tf.Session() as sess:
        sess.run(init)
        start_time = time.time()
        episodes_loss, episodes_len, episodes_return = [], [], []
        totals_steps = 0
        eps = startE
        # Training with e-greedy policy
        for iepisode in range(n_episodes):
            state = env.reset()
            loss, l, t = 0.0, 0.0, 0
            done = False
            while t<max_len_episode and not done:
                t+=1
                # choose action with e-greedy
                if np.random.rand(1) < eps:
                    a = np.array(env.action_space.sample(),dtype=np.int32).reshape((-1))
                else:
                    """
                    QoutA = sess.run(QagentA.Qout,feed_dict={QagentA.state: state.reshape(-1,4)})
                    aA = np.argmax(QoutA,axis=1)
                    QoutB = sess.run(QagentB.Qout,feed_dict={QagentB.state: state.reshape(-1,4)})
                    aB = np.argmax(QoutB,axis=1)
                    if QoutA[0][aA]>QoutB[0][aB]:
                        a = aA
                    else:
                        a = aB
                    """
                    QoutA = sess.run(QagentA.Qout,feed_dict={QagentA.state: state.reshape(-1,4).astype("float32")})
                    QoutB = sess.run(QagentB.Qout,feed_dict={QagentB.state: state.reshape(-1,4).astype("float32")})
                    a = np.argmax(QoutA+QoutB,axis=1).astype("int32")

                # Take action and observe next state and reward
                next_state, _, done,_  = env.step(a[0])
                r = f_reward(done)
                #next_state, r, done,_  = env.step(a[0])
                # Add experience to replay buffer
                state = np.reshape(state,(-1,4)).astype("float32")
                next_state = np.reshape(next_state,(-1,4)).astype("float32")
                a = np.reshape(a,(-1))
                experience = (state,a,next_state,r,done)
                exp_replay.add(experience)
                if len(exp_replay.replay)>=(exp_replay_buffer_size):
                    minibatch = exp_replay.sample(batch_size)
                    states_batch = np.stack([minibatch[i][0] for i in range(len(minibatch))], axis=0).reshape((-1,4))
                    actions_batch = np.stack([minibatch[i][1] for i in range(len(minibatch))], axis=0).reshape((-1))
                    next_state_batch = np.stack([minibatch[i][2] for i in range(len(minibatch))], axis=0).reshape((-1,4))
                    r_batch = np.stack([minibatch[i][3] for i in range(len(minibatch))], axis=0).reshape((-1))
                    # Update randomly QagentA or QagentB
                    if np.random.rand(1)<0.5:
                        a_ = sess.run(QagentA.predict,feed_dict={QagentA.state: next_state_batch})
                        QoutB = sess.run(QagentB.Qout,feed_dict={QagentB.state: next_state_batch})
                        targetQ_batch = r_batch + (1+np.transpose(r_batch))*df * QoutB[np.arange(np.shape(QoutB)[0]),a_]
                        l,_ = sess.run([QagentA.l,QagentA.updateModel],feed_dict={QagentA.state: states_batch,
                                                                                QagentA.targetQ: targetQ_batch.astype("float32"),
                                                                                QagentA.actions: actions_batch})
                    else:
                        a_ = sess.run(QagentB.predict,feed_dict={QagentB.state: next_state_batch})
                        QoutA = sess.run(QagentA.Qout,feed_dict={QagentA.state: next_state_batch})
                        targetQ_batch = r_batch + (1+np.transpose(r_batch))*df * QoutA[np.arange(np.shape(QoutA)[0]),a_]
                        l,_ = sess.run([QagentB.l,QagentB.updateModel],feed_dict={QagentB.state: states_batch,
                                                                                QagentB.targetQ: targetQ_batch.astype("float32"),
                                                                                QagentB.actions: actions_batch})
                loss += l
                state = next_state
                totals_steps+=1
            # update exploration prob
            if eps>endE:
                eps-=stepDrop
            episodes_loss.append(loss)
            # log
            if (iepisode)%log_frequency==0:
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("Episodes {} to {} done, took {:2f}s".format(max(0,iepisode+1-log_frequency),iepisode,time.time()-start_time))
                print(" Train loss: {:.4f}\tTotal steps: {:d} ".format(loss,totals_steps))
                #print(" Episode len: {:d}\tEpisode return: {:.2f} ".format(t,(df**(t))*r))
                start_time = time.time()

            # Testing greedy policy every test_frequency
            if (iepisode)%test_frequency==0:
                test_episodes_len, test_episodes_return = [], []
                for ie in range(n_test_episode):
                    obs = env.reset()
                    tstep = 0
                    test_done = False
                    while tstep<max_len_episode and not test_done:
                        tstep+=1
                        """
                        # Environment render
                        if epoch==(nepochs-1):
                            env.render()
                        """
                        #QoutA = sess.run(QagentA.Qout,feed_dict={QagentA.state: state.reshape(-1,4)})
                        #QoutB = sess.run(QagentB.Qout,feed_dict={QagentB.state: state.reshape(-1,4)})
                        #a = np.argmax(QoutA+QoutB,axis=1)
                        a = sess.run(QagentA.predict,feed_dict={QagentA.state: obs.reshape(-1,4).astype("float32")})
                        obs, _, test_done,_  = env.step(a[0])
                        test_r = f_reward(test_done)
                        #obs, test_r, test_done,_  = env.step(a[0])
                    test_episodes_len.append(tstep)
                    test_episodes_return.append((df**(tstep))*test_r)
                mean_test_len = np.mean(np.array(test_episodes_len),axis=0)
                mean_test_return = np.mean(np.array(test_episodes_return),axis=0)
                episodes_len.append(mean_test_len)
                episodes_return.append(mean_test_return)
                if (iepisode)%log_frequency==0:
                    print(" Testing greedy after {} episodes:".format(iepisode))
                    print(" Mean len:  {:.3f}\tMean return:  {:.3f}".format(mean_test_len,mean_test_return))

        if save_model:
            sub_model_path = os.path.join(MODEL_DIR,SUB_DIR)
            if not tf.gfile.Exists(sub_model_path):
                os.makedirs(sub_model_path)
            model_name = model["name"] + ".ckpt"
            model_path = os.path.join(sub_model_path,model_name)
            saver.save(sess,model_path)
            print("Model saved")

    sess.close()
    return episodes_loss, episodes_len, episodes_return

if __name__ == '__main__':
    sub_perf_path = os.path.join(PERF_DIR,SUB_DIR)
    if not tf.gfile.Exists(sub_perf_path):
        os.makedirs(sub_perf_path)
    for model in models:
        runs_loss, runs_len, runs_return = [], [], []
        for run in range(n_runs):
            print("Starting run {}/{} model {}...".format(run+1,n_runs,model["name"]))
            str_time = time.time()
            episodes_loss, episodes_len, episodes_return = online_Qlearning(model,"hidden",run==0)
            # losses
            runs_loss.append(episodes_loss)
            # len
            runs_len.append(episodes_len)
            # returns
            runs_return.append(episodes_return)
            print("\nRun {}/{} model {}, done, tooks {:.2f}s".format(run+1,n_runs,model["name"],time.time()-str_time))

        # Compute stats for loss and save csv
        mean_loss = np.mean(np.array(runs_loss),axis=0)
        std_loss = np.std(np.array(runs_loss),axis=0)
        name_file = model["name"] + "_losses.csv"
        file_path = os.path.join(sub_perf_path,name_file)
        columns = ["mean_loss","std_loss"]
        save_csv([mean_loss,std_loss],columns,file_path)
        # Compute stats for len and returns and save csv
        mean_len = np.mean(np.array(runs_len),axis=0)
        std_len = np.std(np.array(runs_len),axis=0)
        mean_returns = np.mean(np.array(runs_return),axis=0)
        std_returns = np.std(np.array(runs_return),axis=0)
        name_file = model["name"] + "_perf.csv"
        file_path = os.path.join(sub_perf_path,name_file)
        columns = ["mean_len","std_len","mean_returns","std_returns"]
        save_csv([mean_len,std_len,mean_returns,std_returns],columns,file_path,test_frequency)
