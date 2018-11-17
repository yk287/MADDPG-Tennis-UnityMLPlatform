## Implementation of MADDPG Algorithm

## The Algorithm

The model is an extension of DDPG algorithm for multiple agents. The Actor part of DDPG algorithm only takes in local observations for the agents meaning each agent only gets it's own observation, however the Critics for each agent gets all the observations for all the agents in the environment. By sharing the full observations we can achieve stationairy of the environment and this is the primary motivating factor for the algorithm as mentioned in the ![paper](https://arxiv.org/pdf/1706.02275.pdf)

## Neural Network Model

Unlike the cases with single agent, the environment with multiple agents that require cooporation were rather difficult to solve. I tired many different approaches and the final model that solved the environment had 4 hidden layers with 256, 256, 128, and 64 nodes each for both Actor and Critic. I tried to see if a smaller model can solve the environment but I was not able to do so with the node sizes of [128, 128, 64, 32]. I was also not able to solve the environment with 2 hidden layers with 256 nodes for each although this particular model did not have batch normalization.

After failing to achieve any success for a while, batch normalization was used to help with the model training, but what's interesting is that when I had batch normalization for each layer in the model, I was not able to solve the environment. The only case where I was able to solve the environment was when I had batch normalization for the inputs (states) only, and had no normalization for the rest of the layers. 

The models were updated after every 2 steps the agent took and soft-updates were used to updat the target models.

For measuring the error of value functions, MSE was used. 

## Exploration

Similar to DDPG algorithm, a noise was injected into the acitons selected to promote exploration in the continuous action space. This noise was decayed after every episodes similar to epsilon greedy algorithm


## Hyperparameters

* Number of Hidden Layer: 4 
* Number of Nodes : [256, 256, 128, 64]
* Batch Size : 128
* Discount Factor = 0.99
* Update Frequency = 2
* Soft Update Rate = 0.001
* Loss Function = MSE


## Plots

Looking at the average score over 100 episodes, it appears that the agent solves the environment at around 250 episodes meaning that the agent was able to average 30+ scores from around episodes 150 to 250. This is quite remarkable given that I do not decrease the exploeration rate and give the model constant noise through training. 

In the [paper](https://arxiv.org/pdf/1802.09477.pdf), the authors suggest running the agent without any noise during training to gauge how well it has learned to solve the environments, and during training I gave no noise every 100 episodes

![](RawScore.png)

![](progress.png)

## Ideas For Future Work

PPO is an algorithm that improves on policy gradient methods such as REINFORCE, and it should be able to solve this environment as well. PPO re-uses some past experience to perform policy updates, which helps with the downside of on-policy method which is that it does not learn from past experiences. PPO can also be performed in parallel with multiple agents and given that this environment has a version that has 20 agents, PPO should be very well suited to solve it.  

1) PPO
