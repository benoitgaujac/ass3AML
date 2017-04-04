from __future__ import division

import os
import sys
import time
from argparse import ArgumentParser
import pdb

import numpy as np
import csv
import pandas as pd
import tensorflow as tf

import gym
import RLagent

env = gym.make('CartPole-v0')
test_env = gym.make('CartPole-v0')

max_len_episode = 300
df = 0.99 # discount factor
batch_size = 256
nepochs = 201
log_frequency = 100
test_freq = 4
n_test_episode = 20
n_runs = 4
MODEL_DIR = "../models"
if not tf.gfile.Exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
PERF_DIR = "../Perfs"
if not tf.gfile.Exists(PERF_DIR):
    os.makedirs(PERF_DIR)
SUB_DIR = "batchQlearning"

parser = ArgumentParser(description='Generate random episodes and batch Qlearning')
parser.add_argument('--use',action='store',default="rand100",dest="use",nargs ='?',
                    help='use of the script. Can be print, rand100, batchQ. Default=rand100')
parser.add_argument('--mode',action='store',default="test",dest="mode",nargs ='?',
                    help='train or test for batchQ use. Default=test')
parser.add_argument('--lr',action='store',default=0.01,dest="lr",nargs ='?',
                    help='learning rate for train mode in batchQ use. Default=None')
parser.add_argument('--approx',action='store',default="hidden",dest="approx",nargs ='?',
                    help='frunction approximator for batchQ use. Default=hidden')


######################################## Part A.1 & A.2 ########################################
def print_episodes():
    for i_episode in range(3):
        observation = env.reset()
        t = 0
        return_0 = 0
        done = False
        while t<max_len_episode and not done:
            action = env.action_space.sample()
            observation, _, done, info = env.step(action)
            reward = f_reward(done)
            return_0 += (df**t)*reward
            t+=1
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode+1,t+1))
                print("Discounted returns {:.4f}".format(return_0))

def print_rnd_episodes(nb_episodes):
    results = []
    for i_episode in range(nb_episodes):
        observation = env.reset()
        t = 0
        return_0 = 0
        done = False
        while t<max_len_episode and not done:
            action = env.action_space.sample()
            observation, _, done, info = env.step(action)
            reward = f_reward(done)
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode+1,t+1))
            return_0 += (df**t)*reward
            t+=1
        results.append([t,return_0])
    results_array = np.array(results)
    results_stats = [np.mean(results_array,axis=0)[0],np.std(results_array,axis=0)[0],np.mean(results_array,axis=0)[1],np.std(results_array,axis=0)[1]]
    print("\nEpisode len:")
    print("Mean: {}\t Std: {}".format(results_stats[0],results_stats[1]))
    print("Episode return:")
    print("Mean: {}\t Std: {}".format(results_stats[2],results_stats[3]))

######################################## utils functions ########################################
def collect_rnd_episodes(nb_episodes):
    episodes = []
    for i_episode in range(nb_episodes):
        # Initialize environment
        observation = env.reset()
        # Initialize history
        t = 0
        done = False
        observations,rewards,actions,terminal_state = [], [], [], []
        # episode loop
        while t<max_len_episode and not done:
            action = env.action_space.sample()
            actions.append(action)
            observations.append(observation)
            observation, _, done, _ = env.step(action)
            reward = f_reward(done)
            # Update obs, reward and action
            rewards.append(reward)
            terminal_state.append(done)
            t+=1
        if (i_episode+1)%500==0:
            print("Episode {} generated".format(i_episode+1))
        # Update episodes
        episodes.append([actions,observations,rewards,terminal_state])
    return episodes

def f_reward(end_of_episode):
    r = 0
    if end_of_episode:
        r = -1
    return r

def save_csv(to_save,columns_list,dst_path,idx_frq=1):
    if os.path.isfile(dst_path):
        os.remove(dst_path)
    if len(to_save)!=len(columns_list):
        raise Exception("Error in formating files to save")
    epoch_idx = np.arange(0,nepochs,idx_frq)
    file_tosave = pd.DataFrame(epoch_idx, columns=["epochs"])
    for i in range(1,len(to_save)+1):
        file_tosave.insert(i,columns_list[i-1] , np.array(to_save[i-1]))
    file_tosave.to_csv(dst_path, index=False)
    print("CSV file successfully created.")

######################################## Main Batch Qlearning ########################################
def Batch_Qlearning(mode,episodes_histo,learning_rate,fct_approximator="linear",save_model=False):
    # Collect episodes
    action_history = np.concatenate([episodes_histo[i][0] for i in range(len(episodes_histo))], axis=0)
    state_history  = np.concatenate([episodes_histo[i][1] for i in range(len(episodes_histo))], axis=0)
    reward_history = np.concatenate([episodes_histo[i][2] for i in range(len(episodes_histo))], axis=0)
    terminal_state_history = np.concatenate([episodes_histo[i][3] for i in range(len(episodes_histo))], axis=0)
    total_steps = len(state_history)
    # Reset tf graph
    tf.reset_default_graph()
    # Build agent
    batchQ_agent = RLagent.Q_agent(fct_approximator,learning_rate)
    # Initializer tf variables
    init = tf.global_variables_initializer()
    # Saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        losses = []
        mean_l, mean_r = [], []
        if mode=="train":
            start_time = time.time()
            for epoch in range(nepochs):
                loss = 0
                for ibatch in range(int(total_steps/batch_size)):
                    actions = action_history[ibatch*batch_size:(ibatch+1)*batch_size]
                    states = state_history[ibatch*batch_size:(ibatch+1)*batch_size]
                    next_states = state_history[ibatch*batch_size+1:(ibatch+1)*batch_size+1]
                    rewards = reward_history[ibatch*batch_size:(ibatch+1)*batch_size]
                    terminal_state = terminal_state_history[ibatch*batch_size:(ibatch+1)*batch_size]
                    # Get predictions from agent
                    Qout = sess.run(batchQ_agent.Qout,feed_dict={batchQ_agent.state: next_states})
                    maxQ = np.amax(Qout, axis=1)
                    # Build target
                    targetQ = rewards + (1-np.transpose(terminal_state))*df * maxQ
                    # training
                    l, _ = sess.run([batchQ_agent.l,batchQ_agent.updateModel], feed_dict={batchQ_agent.state: states,
                                                                                        batchQ_agent.actions: actions,
                                                                                        batchQ_agent.targetQ: targetQ})
                    loss += l/(int(total_steps/batch_size))
                if (epoch)%log_frequency==0:
                    print("")
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print("Epochs {} to {} done, took {:2f}s".format(max(1,epoch+1-log_frequency),epoch+1,time.time()-start_time))
                    print("Training loss: {:.4f}".format(loss))
                    start_time = time.time()
                losses.append(loss)
                # Testing
                if (epoch)%test_freq==0:
                    episodes_len, episodes_return = [], []
                    for iepisode in range(n_test_episode):
                        state = test_env.reset()
                        t, return_ = 0, 0
                        done = False
                        while t<max_len_episode and not done:
                            """
                            # Environment render
                            if epoch==(nepochs-1) and (iepisode==0 or iepisode==(n_test_episode-1)):
                                env.render()
                            """
                            a = sess.run(batchQ_agent.predict,feed_dict={batchQ_agent.state: state.reshape(-1,4)})
                            state, _, done,_  = test_env.step(a[0])
                            r = f_reward(done)
                            return_ += (df**t)*r
                            t+=1
                        episodes_len.append(t)
                        episodes_return.append(return_)
                    mean_len = np.mean(np.array(episodes_len),axis=0)
                    std_len = np.std(np.array(episodes_len),axis=0)
                    mean_rewards = np.mean(np.array(episodes_return),axis=0)
                    std_rewards = np.std(np.array(episodes_return),axis=0)
                    if (epoch)%log_frequency==0:
                        print("Testing after {} epochs:".format(epoch+1))
                        print(" Mean len:  {:.3f}\t\tStd len:  {:.3f}".format(mean_len,std_len))
                        print(" Mean returns:  {:.4f}\t\tStd returns:  {:.4f}\n".format(mean_rewards,std_rewards))
                    mean_l.append(mean_len)
                    mean_r.append(mean_rewards)
            if save_model:
                sub_model_path = os.path.join(MODEL_DIR,SUB_DIR)
                if not tf.gfile.Exists(sub_model_path):
                    os.makedirs(sub_model_path)
                model_name = fct_approximator + "_" + str(learning_rate) + ".ckpt"
                model_path = os.path.join(sub_model_path,model_name)
                saver.save(sess,model_path)
                print("Model saved")

        elif mode=="test":
            sub_model_path = os.path.join(MODEL_DIR,SUB_DIR)
            model_name = fct_approximator + "_" + str(learning_rate) + ".ckpt"
            model_path = os.path.join(sub_model_path,model_name)
            if not tf.gfile.Exists(model_path+".meta"):
                raise Exception("no weights given")
            saver.restore(sess, model_path)
            # Testing
            episodes_len, episodes_return = [], []
            for iepisode in range(n_test_episode):
                state = env.reset()
                t, return_ = 0, 0
                done = False
                while t<max_len_episode and not done:
                    """
                    # Environment render
                    if epoch==(nepochs-1) and (iepisode==0 or iepisode==(n_test_episode-1)):
                        env.render()
                    """
                    a = sess.run(batchQ_agent.predict,feed_dict={batchQ_agent.state: state.reshape(-1,4)})
                    state, _, done,_  = env.step(a[0])
                    r = f_reward(done)
                    return_ += (df**t)*r
                    t+=1
                episodes_len.append(t)
                episodes_return.append(return_)
            mean_len = np.mean(np.array(episodes_len),axis=0)
            std_len = np.std(np.array(episodes_len),axis=0)
            mean_rewards = np.mean(np.array(episodes_return),axis=0)
            std_rewards = np.std(np.array(episodes_return),axis=0)
            print("Testing for {} episodes".format(n_test_episode))
            print(" Mean len:  {:.3f}\t\tStd len:  {:.3f}".format(mean_len,std_len))
            print(" Mean returns:  {:.4f}\t\tStd returns:  {:.4f}\n".format(mean_rewards,std_rewards))
            mean_l.append(mean_len)
            mean_r.append(mean_rewards)
            losses.append([0])

    sess.close()
    return losses, mean_l, mean_r

if __name__ == '__main__':
    options = parser.parse_args()
    if options.use=="print":
        print_episodes()
    elif options.use=="rand100":
        print_rnd_episodes(100)
    elif options.use=="batchQ":
        # Create perf folder
        sub_perf_path = os.path.join(PERF_DIR,SUB_DIR)
        if not tf.gfile.Exists(sub_perf_path):
            os.makedirs(sub_perf_path)
        # collect 2000 episodes
        episodes_histo = collect_rnd_episodes(2000)
        # options
        appro = options.approx
        lr = options.lr
        """
        lr_list = [0.00001,0.0001,0.001,0.01,0.1,0.5]
        appro_list = ["hidden","linear"]

        # initialize perfo and lost for runs
        for appro in appro_list:
            for lr in lr_list:
        """
        runs_loss, runs_mean_l, runs_mean_r = [], [], []
        for run in range(n_runs):
            print("\nStarting run {}/{}, {} {:.5f} ...".format(run+1,n_runs,appro,lr))
            str_time = time.time()
            loss, mean_l, mean_r = Batch_Qlearning(options.mode,episodes_histo,lr,appro)
            runs_loss.append(loss)
            runs_mean_l.append(mean_l)
            runs_mean_r.append(mean_r)
            print("Run {}/{}, {} {:.5f} done, tooks {:.2f}s".format(run+1,n_runs,appro,lr,time.time()-str_time))
        # Save loss
        if options.mode=="train":
            runs_loss = np.mean(np.array(runs_loss),axis=0)
            name_file = "losses_" + appro + "_" + str(lr) + ".csv"
            file_path = os.path.join(sub_perf_path,name_file)
            columns = ["training_loss"]
            save_csv([runs_loss],columns,file_path)
        # Save perfs
        runs_std_l = np.std(np.array(runs_mean_l),axis=0)
        runs_mean_l = np.mean(np.array(runs_mean_l),axis=0)
        runs_std_r = np.std(np.array(runs_mean_r),axis=0)
        runs_mean_r = np.mean(np.array(runs_mean_r),axis=0)
        name_file = "perf_" + appro + "_" + str(lr) + ".csv"
        idx_frq = test_freq
        if options.mode=="test":
            name_file = "test_" + name_file
            idx_frq = nepochs
        file_path = os.path.join(sub_perf_path,name_file)
        columns = ["mean_len","std_len","mean_returns","std_returns"]
        save_csv([runs_mean_l,runs_std_l,runs_mean_r,runs_std_r],columns,file_path,idx_frq)
