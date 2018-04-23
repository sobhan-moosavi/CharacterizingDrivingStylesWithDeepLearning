# Characterizing Driving Styles with Deep Learning

Here, a tensorflow implementation for <a href="https://arxiv.org/abs/1607.03611">Characterizing Driving Styles with Deep Learning</a> is provided. 

## Requirements

1. Python 2.7
2. Tensorflow 1.3.0
3. Cuda 8.0.61

## Experiments
NOTE: training the full pipeline from DQNs to transfer using Progressive architecure may take about two weeks with a fast GPU. 

## Training DQN
You may use the code available <a href="https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner">here</a> to train the DQN and generate t7 torch files. Then, put the t7 files in experts/ directory. 

## Training Progressive network
To train progressive network, run following script:

	$ cd scripts
	$ ./run_progressive [1-based gpuid] ["'SourceGame_1','SourceGame_2','SourceGame_3'"] ["'TargetGame'"] [OutputFile]
 
Here, the first parameter is the GPU id, the second parameter is a list of source games (that is, the ones with frozen DQN networks), the third parameter is the Target game which we want to train a DQN for that, and the last parameter is the output file name. An example is:

	$ ./run_progressive 0 "'pong','breakout'" "'tennis'" tennis_progressive

## Acknowledgments
This implementation is an adapatation of <a href="https://github.com/eparisotto/ActorMimic">Actor-Mimic</a>, which provides code for the <a href="https://arxiv.org/abs/1511.06342">Actor-Mimic deep multitask and transfer reinforcement learning</a>. 

## References 
1. [Human-level control through deep reinforcement learning, Nature 2015](https://www.nature.com/articles/nature14236.pdf)
2. [Progressive Neural Networks, arXiv preprint 2016](https://arxiv.org/pdf/1606.04671.pdf)
3. [Actor-Mimic deep multitask and transfer reinforcement learning, ICLR 2016](https://arxiv.org/pdf/1511.06342.pdf)
4. [A Nested Architecture to Improve Transfer Learning in Deep Reinforcement Lea
