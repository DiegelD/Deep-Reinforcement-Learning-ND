[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 2: Continous Control


## Abstract
In the following you will find the parameters for the experiments and the results.

Overview
---
1. Parameters
2.  Experiment Settings
3. Future Work

## 1) Final Parameters
### NN Architecture
#### Actor 
1. Fully-connected layer - input 33 (observation states), output 400
2. Fully-connected layer - input 400, output 300
3. Fully-connected layer - input 300, output 4 (action size)

#### Critic
1. Fully-connected layer - input 33 (observation states), output 256
2. Fully-connected layer - input 256, output 256
4. Fully-connected layer - input 256, output 256
3. Fully-connected layer - input 256, output 1 (TD error)

#### Hyper-Parameters
-BUFFER_SIZE = int(1e6)&nbsp;&nbsp;&nbsp;&nbsp;# replay buffer size  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-BATCH_SIZE = 256 &nbsp;#&nbsp;&nbsp;&nbsp; minibatch size<br />
-GAMMA = 0.99&nbsp;&nbsp;&nbsp;&nbsp;# discount factor     &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  -TAU = 1e-3&nbsp;&nbsp;&nbsp;&nbsp;# for soft update of target parameters<br />
-LR_ACTOR = 1e-4 &nbsp;&nbsp;&nbsp;&nbsp;# learning rate of the actor &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -LR_CRITIC = 1e-3 &nbsp;&nbsp;&nbsp;&nbsp;# learning rate of the critic<br />
-WEIGHT_DECAY = 0.0&nbsp;&nbsp;&nbsp;&nbsp;# L2 weight decay&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-EPSILON_DECLAY = 1e-6&nbsp;&nbsp;&nbsp;&nbsp;# noise reduction -> like greedy reduction<br />
-EPSIOLON       = 1.0&nbsp;&nbsp;&nbsp;# Initial noise reduction level&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;-N_LEARN_UPDATES = 10&nbsp;&nbsp;&nbsp; #Number of learning updates<br />
-N_TIME_STEPS    = 20&nbsp;&nbsp;&nbsp;&nbsp;# every n time step to update<br />

## 2) Experiment Settings
### 2.1 Model Comparison
#### 2.1.1 Net Architectures
1. Model: The original model from Udacity with an architecture
    - actor one fully connected layer 256
    - critic three fully connected layers size 256 256 128
2. Model Introduction of the actor net from the original DDPG paper[1] and staying with the critic net from former projects
    - actor two fully connected layers size  400 300
    - critic three fully connected layers size 256 256 128
3. Model Going all in the net size from the original DDPG paper[1]
    - actor two fully connected layers size  400 300
    - critic two fully connected layers size 200 200
#### 2.1.2 Hyper Parameter
-BUFFER_SIZE = int(1e6)&nbsp;# replay buffer size  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -BATCH_SIZE = 128 &nbsp;# minibatch size<br />
-GAMMA = 0.99&nbsp;# discount factor     &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  -TAU = 1e-3&nbsp;# for soft update of target parameters<br />
-LR_ACTOR = 1e-4 &nbsp;# learning rate of the actor &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; LR_CRITIC = 1e-3 &nbsp;# learning rate of the critic<br />
-WEIGHT_DECAY = 0.0&nbsp;# L2 weight decay&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-EPSILON_DECLAY = 1e-6&nbsp;# noise reduction -> like greedy reduction<br />
-EPSIOLON       = 1.0&nbsp;# Initial noise reduction level&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;-N_LEARN_UPDATES = 10&nbsp; #Number of learning updates<br />
-N_TIME_STEPS    = 20&nbsp;# every n time step to update<br />

### 2.2 Batchsize Comparison
#### 2.2.1 Net Architectures
    - actor two fully connected layers size  400 300
    - critic three fully connected layers size 256 256 128
#### 2.1.2 Hyper Parameter
Batch-size variation between 256, 128 and 64.<br />
The left settings are followings:<br />
-BUFFER_SIZE = int(1e6)&nbsp;# replay buffer size  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -BATCH_SIZE =  &nbsp;# minibatch size<br />
-GAMMA = 0.99&nbsp;# discount factor     &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  -TAU = 1e-3&nbsp;# for soft update of target parameters<br />
-LR_ACTOR = 1e-4 &nbsp;# learning rate of the actor &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; LR_CRITIC = 1e-3 &nbsp;# learning rate of the critic<br />
-WEIGHT_DECAY = 0.0&nbsp;# L2 weight decay&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-EPSILON_DECLAY = 1e-6&nbsp;# noise reduction -> like greedy reduction<br />
-EPSIOLON       = 1.0&nbsp;# Initial noise reduction level&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;-N_LEARN_UPDATES = 10&nbsp; #Number of learning updates<br />
-N_TIME_STEPS    = 20&nbsp;# every n time step to update<br />


### 2.3 Weight Decay Comparison
#### 2.3.1 Net Architectures
    - actor two fully connected layers size  400 300
    - critic three fully connected layers size 256 256 128
#### 2.3.2 Hyper Parameter
L2 Weight Decay variation between 0,1e-2, 1e-4. The left settings are followings:
-BUFFER_SIZE = int(1e6)&nbsp;# replay buffer size  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -BATCH_SIZE = 128 &nbsp;# minibatch size<br />
-GAMMA = 0.99&nbsp;# discount factor     &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  -TAU = 1e-3&nbsp;# for soft update of target parameters<br />
-LR_ACTOR = 1e-4 &nbsp;# learning rate of the actor &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; LR_CRITIC = 1e-3 &nbsp;# learning rate of the critic<br />
-WEIGHT_DECAY = &nbsp;# L2 weight decay&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-EPSILON_DECLAY = 1e-6&nbsp;# noise reduction -> like greedy reduction<br />
-EPSIOLON       = 1.0&nbsp;# Initial noise reduction level&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;-N_LEARN_UPDATES = 10&nbsp; #Number of learning updates<br />
-N_TIME_STEPS    = 20&nbsp;# every n time step to update<br />

### 2.4 Results
As a result the DDPG algorithm with an actor size of 400-300, a critic size of 256-256-128, a batch-size of 256 and no weight decay shows the best performance. 
All the other parameters can be found here Report.md.

<figure>
 <img src="./img/Conclusion.png" width="750" alt="whatever" />
 <p></p> 
 <p style="text-align: center;"> Fig. 4.1: Results of the experiments.  </p> 
 </figcaption>
</figure>
 <p></p>



## 3) Future Work
Further improvements could be done in the following fields:
- Implementing [PPO](https://arxiv.org/abs/1707.06347) & [TD3](https://arxiv.org/pdf/1802.09477.pdf) (State of the art improvements over DDPG)
- Implementing a multiagent [A2C](https://arxiv.org/pdf/1602.01783.pdf), [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) algorithm to handle the 20 agents. 
- Testing a change from Ornstein-Ulenbeck to Gaussian noise
- Implementing [StableBaseline3](https://stable-baselines3.readthedocs.io/en/master/) to compare differently algorithms easily
