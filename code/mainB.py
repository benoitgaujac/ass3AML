from __future__ import division

import os
import sys
import time
import pdb

import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
import warnings

import gym
import ATARIagent

#env = gym.make('Pong-v3')
#print(env.action_space)
#env = gym.make('MsPacman-v3')
#env = gym.make('Boxing-v3')

# Experience set up
df = 0.99 # discount factor
#learning_rate = 0.001 # learning rate
nsteps = 1000000 # number of training steps
n_test_episode = 3 # number of testing episodes
test_frequency = 40000 #50000 # test frequency of greedy policy
update_frequency = 5000 # frequency of update for target_net
store_loss_frequency = 1000 # frequency of store loss
startE = 0.5 # starting exploration probability
endE = 0.05 # ending exploration probability
stepDrop = (startE - endE)/nsteps # decay step size of the exploration policy
batch_size = 64 # batch size for experience replay buffer
exp_replay_buffer_size = 500000 # size of the experience replay buffer
n_runs = 1 # number of runs for average performances
log_frequency = 4000 # frequency of login
# Environment setting
frame_width = 84
frame_height = 84
frame_chan = 4
shape = (-1,frame_width,frame_height,frame_chan)

#lr_list = [0.00005,0.0001,0.0005]
lr_list = [0.0001,]

pong = {"name":"Pong-v3","dim_action_space": 6,}
pacman = {"name":"MsPacman-v3","dim_action_space": 9}
boxe = {"name":"Boxing-v3","dim_action_space": 18}
models = [pong, pacman, boxe]
#models = [pong, ]

"""
env = gym.make(pacman["name"])
for i_episode in range(2):
    # Initialize environment
    observation = env.reset()
    done = False
    # episode loop
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, r, done, _ = env.step(action)
    pdb.set_trace()
"""

MODEL_DIR = "../models"
if not tf.gfile.Exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
PERF_DIR = "../Perfs"
if not tf.gfile.Exists(PERF_DIR):
    os.makedirs(PERF_DIR)
SUB_DIR = "Atari"

######################################## utils functions ########################################
def f_reward(gym_reward):
    return np.clip(gym_reward,-1, 1)

def update_target_net(variables,sess):
    for var in variables:
        sess.run(var)

def process_frame(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized = resize(frame, (frame_width,frame_height),preserve_range=True)
        return rgb2gray(resized).astype("uint8")

def save_csv(to_save,columns_list,dst_path,idx_frq=1):
    if os.path.isfile(dst_path):
        os.remove(dst_path)
    if len(to_save)!=len(columns_list):
        raise Exception("Error in formating files to save")
    episodes_idx = np.arange(0,nsteps,idx_frq)
    file_tosave = pd.DataFrame(episodes_idx, columns=["steps"])
    for i in range(1,len(to_save)+1):
        file_tosave.insert(i,columns_list[i-1] , np.array(to_save[i-1]))
    file_tosave.to_csv(dst_path, index=False)
    print("CSV file successfully created.")


######################################## 1&2 ########################################
def collect_rnd_episodes(environment,nb_episodes):
    env = gym.make(environment)
    nframes,returns = [], []
    for i_episode in range(nb_episodes):
        # Initialize environment
        observation = env.reset()
        # Initialize history
        t, re = 0, 0.0
        done = False
        histo_frames = []
        # episode loop
        while not done:
            t+=1
            action = env.action_space.sample()
            observation, r, done, _ = env.step(action)
            reward = f_reward(r)
            re += (df**t)*reward
        if (i_episode+1)%50==0:
            print("Episode {} generated".format(i_episode+1))
        nframes.append(t)
        returns.append(re)
    return nframes, returns

def collect_rnd_episodes_from_Qnet(environment,nb_episodes):
    env = gym.make(environment)
    nframes,returns = [], []
    for i_episode in range(nb_episodes):
        # Reset tf graph
        tf.reset_default_graph()
        # Build agent
        Qagent = ATARIagent.Q_agent(model["dim_action_space"],learning_rate,"")
        # Initializer tf variables
        init = tf.global_variables_initializer()
        # Initialize environment
        observation = env.reset()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Initialize history
            t, re = 0, 0.0
            done = False
            histo_frames = []
            observation = env.reset()
            # episode loop
            while not done:
                t+=1
                # add fram to histo
                histo_frames.append(process_frame(observation).astype('uint8'))
                if len(histo_frames)<frame_chan:
                    # sample random action
                    action = env.action_space.sample()
                    action = np.array(action).reshape((-1))
                else:
                    state = np.stack(histo_frames,axis=-1)
                    action = sess.run(Qagent.predict,feed_dict={Qagent.state: state.reshape(shape).astype('float32')})
                observation, r, done, _ = env.step(action[0])
                reward = f_reward(r)
                re += (df**t)*reward
            if (i_episode+1)%50==0:
                print("Episode {} generated".format(i_episode+1))
            nframes.append(t)
            returns.append(re)
        sess.close()
    return nframes, returns


######################################## Main online Qlearning ########################################
def online_Qlearning(model,learning_rate,save_model=False):
    env = gym.make(model["name"])
    test_env = gym.make(model["name"])
    # Reset tf graph
    tf.reset_default_graph()
    # Build agent
    Qagent = ATARIagent.Q_agent(model["dim_action_space"],learning_rate,"")
    # Build target network
    Qtarget_net = ATARIagent.Q_agent(model["dim_action_space"],learning_rate,"target_")
    # Collect train var for target_net
    trainvars = tf.trainable_variables()
    ntrainvars = len(trainvars)
    target_net_vars = []
    for idx,var in enumerate(trainvars[0:ntrainvars//2]):
        target_net_vars.append(trainvars[idx+ntrainvars//2].assign(var.value()))
    # Build experience replay
    exp_replay = ATARIagent.experience_replay(exp_replay_buffer_size)
    # Initializer tf variables
    init = tf.global_variables_initializer()
    # Saver
    saver = tf.train.Saver()
    # Initialize tf session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Start time
        start_time = time.time()
        # Initialize epsilon
        eps = startE
        # Initialize state
        frame = env.reset()
        # keep histo last 4 frames
        histo_frames = []
        # Initialize indicators
        training_loss, episodes_return, episodes_score = [], [], []
        l = 0.0
        # Training with e-greedy policy
        for step in range(nsteps):
            #if(step)%5000==0:
            #    print("Step {}...".format(step))
            # Training
            if len(histo_frames)<frame_chan:
                # add fram to histo
                histo_frames.append(process_frame(frame))
                # sample random action
                rand_action = env.action_space.sample()
                # env step from random action
                frame, _, done, _ = env.step(rand_action)
            else:
                state = np.stack(histo_frames,axis=-1)
                # choose action with e-greedy
                if np.random.rand(1) < eps:
                    a = np.array(env.action_space.sample(),dtype=np.int32).reshape((-1))
                else:
                    a = sess.run(Qagent.predict,feed_dict={Qagent.state: state.reshape(shape).astype('float32')})
                # Take action and observe next state and reward
                frame, reward, done, _  = env.step(a[0])
                # Clip reward
                r = f_reward(reward)
                # Add new frame and remove oldest one
                histo_frames.append(process_frame(frame))
                histo_frames[0:1] = []
                # Add experience to replay buffer
                next_state = np.stack(histo_frames,axis=-1)
                a = np.reshape(a,(-1)).astype('int32')
                #terminal_state = np.array(done).reshape((-1))
                experience = (state,a,next_state,r,done)
                exp_replay.add(experience)
                if len(exp_replay.replay)>=65: #(1024):
                    minibatch = exp_replay.sample(batch_size)
                    states_batch = np.stack([minibatch[i][0] for i in range(len(minibatch))], axis=0).astype('float32')
                    actions_batch = np.stack([minibatch[i][1] for i in range(len(minibatch))], axis=0).reshape((-1))
                    next_state_batch = np.stack([minibatch[i][2] for i in range(len(minibatch))], axis=0).astype('float32')
                    r_batch = np.stack([minibatch[i][3] for i in range(len(minibatch))], axis=0).reshape((-1))
                    terminal_state_batch = np.stack([minibatch[i][4] for i in range(len(minibatch))], axis=0)
                    Qout = sess.run(Qtarget_net.Qout,feed_dict={Qtarget_net.state: next_state_batch})
                    maxQ = np.amax(Qout, axis=1)
                    targetQ_batch = r_batch + df * (1-terminal_state_batch) * maxQ
                    l, _ = sess.run([Qagent.l,Qagent.updateModel],feed_dict={Qagent.state: states_batch,
                                                                            Qagent.targetQ: targetQ_batch.astype('float32'),
                                                                            Qagent.actions: actions_batch})
            if (step)%store_loss_frequency==0:
                training_loss.append(l)
            if done:
                frame = env.reset()
            # update exploration prob
            if eps>endE:
                eps-=stepDrop
            # update target_net
            if (step+1)%update_frequency==0:
                update_target_net(target_net_vars,sess)
            # log
            if (step)%log_frequency==0:
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("Steps {} to {} done, took {:.2f}s".format(max(0,step+1-log_frequency),step,time.time()-start_time))
                print(" Train loss: {:.4f}".format(l))
                start_time = time.time()

            # Testing greedy policy every test_frequency
            if (step)%test_frequency==0:
                test_episodes_return, test_episodes_score = [], []
                for ie in range(n_test_episode):
                    obs = test_env.reset()
                    s, return_, scores = 0, 0.0, 0.0
                    histo_obs = []
                    test_done = False
                    while not test_done:
                        s+=1
                        if len(histo_obs)<frame_chan:
                            # add frame to histo
                            histo_obs.append(process_frame(obs))
                            # sample random action
                            rand_action = test_env.action_space.sample()
                            # env step from random action
                            obs, _, test_done, _ = env.step(rand_action)
                        else:
                            # Take greedy action
                            agreedy = sess.run(Qagent.predict,feed_dict={Qagent.state: np.stack(histo_obs,axis=-1).reshape(shape).astype('float32')})
                            # Env step from greedy
                            obs, rewardgreedy, test_done, _  = test_env.step(agreedy[0])
                            # Clip reward
                            rgreedy = f_reward(rewardgreedy)
                            # Compute return
                            return_ += (df**s)*rgreedy
                            scores += rgreedy
                            # Update histo
                            histo_obs.append(process_frame(obs))
                            histo_obs[0:1] = []
                    test_episodes_return.append(return_)
                    test_episodes_score.append(scores)
                # Compute stat on performances
                mean_test_return = np.mean(np.array(test_episodes_return),axis=0)
                mean_test_score = np.mean(np.array(test_episodes_score),axis=0)
                episodes_return.append(mean_test_return)
                episodes_score.append(mean_test_score)
                if (step)%log_frequency==0:
                    print(" Testing greedy after {} steps:".format(step+1))
                    print(" Mean return: {:.3f}\tMean scores: {:.1f}".format(mean_test_return,mean_test_score))

        if save_model:
            sub_model_path = os.path.join(MODEL_DIR,SUB_DIR)
            if not tf.gfile.Exists(sub_model_path):
                os.makedirs(sub_model_path)
            model_name = model["name"][:-3] + ".ckpt"
            model_path = os.path.join(sub_model_path,model_name)
            saver.save(sess,model_path)
            print("Model saved")

    sess.close()
    return training_loss, episodes_return, episodes_score


if __name__ == '__main__':
    sub_perf_path = os.path.join(PERF_DIR,SUB_DIR)
    if not tf.gfile.Exists(sub_perf_path):
        os.makedirs(sub_perf_path)
    for lr in lr_list:
        for model in models:
            #runs_loss, runs_return, runs_score = [], [], []
            runs_loss = []
            for run in range(n_runs):
                print("Starting run {}/{} model {}...".format(run+1,n_runs,model["name"]))
                str_time = time.time()
                #episodes_loss, episodes_return, episodes_score = online_Qlearning(model,lr,run==0)
                episodes_loss, runs_return, runs_score = online_Qlearning(model,lr,run==0)
                # losses
                runs_loss.append(episodes_loss)
                # returns
                #runs_return.append(episodes_return)
                # scores
                #runs_score.append(episodes_score)
                print("\nRun {}/{} model {}, done, tooks {:.2f}s".format(run+1,n_runs,model["name"],time.time()-str_time))

            # Save loss and perf
            name_file = model["name"][:-3] + "_losses_" + str(lr) + ".csv"
            file_path = os.path.join(sub_perf_path,name_file)
            columns = ["losses"]
            save_csv(runs_loss,columns,file_path,store_loss_frequency)
            # Compute stats for len and returns and save csv
            name_file = model["name"][:-3] + "_perf_" + str(lr) + "..csv"
            file_path = os.path.join(sub_perf_path,name_file)
            columns = ["return","score"]
            save_csv([runs_return,runs_score],columns,file_path,test_frequency)
