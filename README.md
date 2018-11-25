[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

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

1. Setup Anaconda Environment
    - git clone https://github.com/udacity/deep-reinforcement-learning.git
    - cd deep-reinforcement-learning
    - conda create --name drlnd python=3.6
    - source activate drlnd
2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to train the agent!  

### Learning Algorithm

My implementation uses Deep Q Network implementation along with the
*Double DQN* extension. The overall idea is to use a neural network 
to model the value function. The value function is a function of 
state and action and returns the value for the state, action pair.

The network predicts the value using the current state. 
It also computes the value using the reward and the next state.
The model is then optimized to minimize the MSE (mean square error)
between the two values (using current state, and the next state + reward). 

The network starts with random weights but does an 
exploration-exploitation tradeoff as it continues to learn. 
It uses an epsilon-greedy policy where it chooses between a
random action (exploration) and the best action predicted by the
model (exploitation). Initially as the network is not trained, the
majority of examples are from exploration and as the network continues
to learn, the emphasis on exploitation increases.

 
In addition to using a neural network, a Deep Q Network employs a 
couple of additional techniques to ensure that the network is able 
to learn. These include
1. Replay Buffer - The agent maintains a finite size buffer to store the
examples it has seen in the past. From this buffer, it samples batches
and uses them for learning. This random sampling is crucial for making
the training examples independent of each other.
2. Target Network - As the DQN models a continuous function, small changes
in weights learnt while processing a state, can have an impact on other 
states. This can easily make the learning process unstable. To avoid this,
two networks are used. One is used for learning and the second is used
to compute the value function. Periodically weights from the learnt
network are copied to the other network.
3. Double DQN - DQNs have been proven to overestimate the value function.
To prevent this, an extension of DQN, called Double DQN was implemented
in the project. This is shown to speed up the training process. 
 
 ### Future Work
 In this project I have implemented a DQN with one extension - Double DQN.
 However there are several other extensions which are known to improve
 and speed up the training process. These extensions (called Rainbow)
 have been described in the recent paper on training models for Atari
 Games by Google DeepMind. Future work on this project can include
 trying extensions like
 1. Prioritized Replay buffer - Instead of sampling all examples uniformly,
 assign weights to samples based on the error made by the model in prediction it.
 2. Dueling DQN - Processes the convolution features using two distinct paths.
 One is used for predicting value, and the second for predicting advantage.
 3. Try deeper models, learning rates, buffer size and train longer.
 
  