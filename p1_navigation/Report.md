[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Abstract
In the following you will find the parameters for algorithms and results.

Overview
---
1. Parameters
2. Algorithm Results
3. Future Work

## 1) Parameters
### NN Architecture
1. Fully-connected layer - input 37, output 64
2. Fully-connected layer - input 64, output 64
3. Fully-connected layer - input 64, output 4

### Parameters
For training:
- Maximum steps per episode 1000
- Starting epsilon: 1.0
- Ending epsilon: 0.01
- Epsilon decay rate: 0.98

## DQNs Parameters
- Replay buffersize: 1e5
- Batch size: 64
- Gamma (discount factor): 0.99
- Tau(or soft update of target parameters): 1e-3
- Learning rate: 5e-4
- Update frequency (how often the second network is updating): 4

## 2) Algorithm Results
### DQN
```
Episode 100	Average Score: 3.92
Episode 200	Average Score: 8.67
Episode 300	Average Score: 12.29
Episode 329	Average Score: 13.02
Environment solved in 229 episodes!	Average Score: 13.02
```
<figure>
 <img src="./img/scores_dqn.png" width="200" alt="PerDQN" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 2.1: DQN Results.  </p> 
 </figcaption>
</figure>
 <p></p>

### Double DQN
```
Episode 100	Average Score: 0.51
Episode 200	Average Score: 4.19
Episode 300	Average Score: 7.63
Episode 400	Average Score: 11.28
Episode 468	Average Score: 13.01
Environment solved in 368 episodes!	Average Score: 13.01
```
<figure>
 <img src="./img/scores_double_dqn.png" width="200" alt="PerDQN" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 2.2: Double DQN Results.  </p> 
 </figcaption>
</figure>
 <p></p>

### PER DQN
```
Episode 100	Average Score: 2.73
Episode 200	Average Score: 5.18
Episode 300	Average Score: 9.66
Episode 377	Average Score: 13.02
Environment solved in 277 episodes!	Average Score: 13.02
```
<figure>
 <img src="./img/scores_per_dqn.png" width="200" alt="PerDQN" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 2.3: PER DQN Results.  </p> 
 </figcaption>
</figure>
 <p></p>

## 3) Future Work
For further improvements could be done in the following fields:
1. **Deep Neuronal Network**. Instead of three neuronal layers that are connected to the observation space vector, a *CNN*
that is connected to the raw pixel display could be used. This would make the agent applicable for a wide field of games. More information you
can find in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).
2. **State of the art**. Implementing the *Rainbow* algorithm. That so fare reaches the best performance of the DQN agents. Here you will find
the *Rainbow* [paper](https://arxiv.org/abs/1710.02298).

