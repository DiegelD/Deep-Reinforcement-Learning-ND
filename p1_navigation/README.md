[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Abstract
In the following you will find the development of an **deep reinforcement learning Agent** that collects just yellow bannanas and leaves the bad black ones.
Its done with a Vaulue Optimzation based learning with DQNs(figure 1). It learns by itself the rules and control movements by just 
given feedback/reward for the collected banna. 

In the first step of the development der Hyper Parameters of the Epsilon gradient and a suitable Batchsizes are figured out. 
And in the following step the agend is compared against the performance of an Double DQN Agent and a 
Prioritzed Experience Replay Agent combined with a Double DQNA Agent. 

 *In the following are some highlights of the project described. For deeper, wider more detailed insights feel free to check the code that speaks for itself*.

<figure>
 <img src="./img/DRL_landscape.png" width="360" alt="BehaviourControl" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 1: Shematic alocation in the reinforcement landscape.  </p> 
 </figcaption>
</figure>
 <p></p>

Overview
---
1. Intro Reinforcement Learning
2. Deep Reinforcement Learning Q-learning
3. Double Q-Learning
4. Prioritzed Experience Replay
5. Hyper Parameter tuning & Agent Comparison
6. Appendix: *Build Instructions & Simulator* ...

## 1) Intro Reinforcement Learning
The idea that we learn by interacting with our environment is probably the first to occur to us when we think about the nature of learning. When an infant plays, waves its arms or looks about, it 
has no explicit teacher, but it does have a direct sensorimotor connection to its environment. Exercising this connection produces a wealth of information about
cause and effect, about the consequences of actions, and about what to do in order to achieve goals. Throughout our lives, such interaction are undoubtedly a major source of knowledge
about our environment and ourself. Whether we are learning to drive a car to hold conversation, we are acutely aware of how our environment responds
to what we do, and we seek to influence what happens through our behavior. Learning from interaction is a foundational idea underlaying nearly all theories of learning and
intelligence.[1]

Reinforcement learning is learning what to do - how to map situation to action --so as to maximize a numerical rewards signal. The learner is not told which actions to take, but instead must discover which actions yield the most rewards by
trying them. In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards.
These two characteristics -trail and error search and delayed reward are the two most important distinguishing features of reinforcement learning. [1]

The problem formalization in reinforcement is using ideas from dynamical systems theory, specifically, as the optimal control of incompletely-kwon
Markov decision (MD) process.[1]

![equation](https://latex.codecogs.com/gif.image?\dpi{150}&space;\left(S,A,r(s_{t},a_{t}),P(s_{t&plus;1}|s_{t},a_{t}),\gamma&space;\right))

At time step *t*, the agent selects the action ![equation](https://latex.codecogs.com/gif.image?\dpi{150}&space;a_{t}\in&space;A) by following a police
![equation](https://latex.codecogs.com/gif.image?\dpi{150}&space;\pi&space;:&space;S\rightarrow&space;\mathbb{R}). After executing *at*, the agent 
is transferred to the next state *st+1* with probabilities ![equation](https://latex.codecogs.com/gif.image?\dpi{150}&space;P(s_{t&plus;1}|s_{t},a_{t}).
Additional, a reward signal [equation](https://latex.codecogs.com/gif.image?\dpi{110}&space;r(s_{t},a_{t})) is received to describe whether the underlying
action *at* is good for reaching the goal or not. For the purpose of brevity, rewrite ![equation](https://latex.codecogs.com/gif.image?\dpi{110}&space;r(s_{t},a_{t})). By repeating 
this process the agent interacts with the environment and obtains trajectory ![equation](https://latex.codecogs.com/gif.image?\dpi{110}&space;\tau&space;=s_{1},a_{1},r_{1},......,s_{T},r_{T})
at the terminal time step T. The discount cumulative reward from time-step *t* can be formulated as <br />
![equation](https://latex.codecogs.com/gif.image?\dpi{110}&space;R_{t}=\sum_{k=t}^{T}\gamma&space;^{k-t}r_{k})<br />
where ![equation](https://latex.codecogs.com/gif.image?\dpi{110}&space;\gamma&space;\in&space;(0,1)) is the discount rate that determines the importance of the
future reward.[2]

## 2) Deep Reinforcement Learning (Deep Q-Networks)
While reinforcement learning agents have achived some succes in a variety of domains, their applicability has previously been limited to domains in 
which useful features can be handcrafted. Here we use recent advances in training deep neural networks to develop a novel artificial agent, termed
a deep Q-network, that can learn sucessful policies directly from high-dimensinal sensory inputs using end to end reinforcement learning. [3]

So in this project an implementation that is close to this [one](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) is used.
However instead of using Convoluional layers, a less camputional network of 3 Neuronal Networks is used. Hence there is on observation space vector 
of 37 deminsions that contains the agents velocit, along with ray-based perception of objects around agents forward direction and 4 discreate action space values.
More infromation about the environoment can be found in the appendix.


<figure>
 <img src="./img/Net.png" width="500" alt="Net" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 2: Shematic illustration of the neural network.  </p> 
 </figcaption>
</figure>
 <p></p>

 Formally the neural network is used to approximate the optimal action value function

 ![equation](https://latex.codecogs.com/gif.image?\dpi{110}&space;Q^{*}(s|a)=&space;max_{\pi}E[r_{t}&plus;\gamma^{2}r_{t&plus;2}&plus;....|s_{t}=s,a_{t}=a,\pi]&space;)

 wich is the maximum sum of rewars *rt* discounted by y at each time step t. Achievable by a behaviour policy pi=P(a|s), after making an observation *(s)*
 and taking an action *(a)*. Reinforment learning is kwon to be unstable or even to diverge when a nonlinear funtion approximator such as a 
 neuronal network is used to represent the action-vale (also known as Q) function. This instability is corrected by using experience replay and 
 Q-fixed target.[3]

## 3) Double Q-Learning (Double DQN)
TThe popular Q-Learning algorithm is known to overestimate action values under certain conditions. Q-Learning by it self is one of the most popular reinforcement 
learning algorithms, but it is known to sometimes learn unrealistic high action values because it includes a maximization step over estimated action values, which tend to 
prefer overestimated to underestimated values. [4]

To focus on this problem, lets look at the updated rule for Q-Learning and focus on the TD target. Here the max operation is necessary to
find the best possible value for the next state.

<figure>
 <img src="./img/DoubleDQN_tdtarget.png" width="350" alt="DoubleDQN" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 3.1: Update equation Q-Learning.  </p> 
 </figcaption>
</figure>
 <p></p>

lets rewrite the target and expend the operation. Its just a more efficient way of saying that we want to obtain the Q-value for the State *S'*
 and the action that results in the maximum Q-value among all possible action from that state. We can see that the arg max operation can easily make an 
 mistake, specially in the early stages when the estimations are not yet sophisticated and the Q-Value is still evolving. The accuracy of
 our Q-values depends a lot of what actions have been tried and which neighboring states have been explored. This results in an overestimate of Q-values
 since we always pick the maximum among a set of noisy numbers.
 
<figure>
 <img src="./img/DoubleDQN_expanded.png" width="350" alt="DoubleDQN" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 3.2: Update equation Q-Learning expanded.  </p> 
 </figcaption>
</figure>
 <p></p>

To make the estimate more robust a double Q-Learning algorithm can be used. Where we select the best action using  on set of parameters *w*, but evaluate it
using a different set of parameters *w'*. It's basically like having two separate function approximations that must agree on the best action. If *w* pick an action that is 
according to *w'*, then the Q-value returned is not that high. In the long run, this prevents the algorithm from propagating incidental high rewards
that may have been obtained by chance and don't reflect long-term returns.<br />
Where do we get the second set of parameters from? In the original formulation of Double Q-Learning, you would basically maintain two value functions
and randomly choose one of them to update at each step and using the other only for evaluating actions. But we using DQNs with fixed Q targets, so
we already have an alternate set of parameters. Remember *w^-*. Since w-minus is kept frozen for a while it is different enough from *w* that it can be
reused for this purpose. And thats it, this simple modification keeps Q-values in check, preventing them from exploding in early stages of learning or
fluctuating later on

<figure>
 <img src="./img/DoubleDQN_update.png" width="350" alt="DoubleDQN" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 3.3: Update equation Double DQN.  </p> 
 </figcaption>
</figure>
 <p></p>

 ## 4) Prioritzed Experience Replay
Online reinforcement learning (RL) agents incrementally update their parameters (of the policy, value function or model) while they observe a stream of experience.
In their simplest from, they discard incomming data immediatly, after a signle update. Two issues with this are a strongly correlated updates that break i.i.d. assumption of many popular stochatic 
gradient-based algorithms and (b) the rapid forgetting of possible rare experience that would be usefull later on.[5]

LEts recall the basic idea behind experience replay. The Agent interact with the envrionoment to collect eyperience tuples <S,A,R....>, save
them in a buffer, and then later reandamly sample a batch to learn from. This helps to break the correlation between  consecutie sxperienaces and
stabilizes our learning algorithm.<br />
Now Prioritzed Experience Replay comes into play, some of these experiences may be more importent for learning than others and may occur infrequently.
Since buffers are practically limited in capacity older may importend experiences get lost. This is where Prioritzed Experience Replay helps.
<figure>
 <img src="./img/experiance_replay.png" width="350" alt="PerDQN" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 4.1: Experiance Replay working principle.  </p> 
 </figcaption>
</figure>
 <p></p>

One approach to assigin the priorities for the tupls, is to use the TD error delta. The bigger the error the more we expect to learn from that tuple.
So lets take the magintude of this error as a measure of priority and stor it along with each vorresponding tuple in the replay buffer.
By using batachs, as we do, we can use sampling probablities. Select any tuple *i* with a probabilty equal to its priority value PI normalize
by the sum of all priority values in the replay buffer. When a tuple is picked we can update its priority with a neqly computed TD error using
the lates q values.This reduce the number of batch updates needed to learn a value function.

<figure>
 <img src="./img/per_buffer.png" width="350" alt="PerDQN" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 4.2: Experiance Replay Buffer.  </p> 
 </figcaption>
</figure>
 <p></p>


## Appendix
### Citation
[1]Reinforcement Learning, Sutton & Barton <br />
[2]Reinforcement Learning and Deep Learning based Lateral Control for Autonomous Driving [Link](https://arxiv.org/abs/1810.12778) <br />
[3]Human-level control through deep reinforcement learning [Link](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) <br />
[4]Deep Reinforcement Learning with Double Q-Learning [Link](https://arxiv.org/abs/1509.06461) <br />
[5]Prioritized Experience Replay [Link](https://arxiv.org/abs/1511.05952) <br />

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 
