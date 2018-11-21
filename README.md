An Implementation of MADDPG with Batch normalization that solves the Tennis Environment

![](tennis.gif)

### Introduction 

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of 0.5+ (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

    After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
    This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least 0.5+.

### Installation

Step 1: Clone the repo

The repo includes the Unity Environment for Linux OS

Step 2: Install Dependencies

Easiest way is to create an anaconda environment that contains all the required dependencies to run the project. Other than the modules that come with the Anaconda environment, Pytorch and unityagents are required.

```
conda create --name Tennis python=3.6
source activate Tennis
conda install -y pytorch -c pytorch
pip install unityagents
```

## Training

To train the agent that learns how to solve the environment, simply run **main.py**. This will start the training process with the default hyperparameters given to the model. When the environment is solved, the script saves the model parameters and also outputs a couple of graphs that shows the rewards per episode, and the average rewards last 100 episodes.

Weights for the models that successfully achived an average score of 0.5+ over 100 episodes are included. The one named **episode-2230.pt** is the one trained with the model included in the repo.
