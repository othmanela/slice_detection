# Deep Reinforcement Learning for CT slice detection

This repository contains the code for the paper: Deep Reinforcement Learning for L3 Slice Localization in Sarcopenia Assessment. 
This work was published in [Machine Learning for Medical Imaging at MICCAI 2021.](https://link.springer.com/chapter/10.1007/978-3-030-87589-3_33) 

In short, a Deep Q-Network (DQN) is trained to detect a single slice in a CT volume initially projected using a frontal maximum intensity projection (MIP).
The reinforcement learning agent has been applied to the task of localizing the L3 slice which corresponds to the computation of sarcopenia scores.

## Visualization of the training process
A complete video is provided in the folder `assets`

### Beginning of the training
During the beginning of the training, the agent is exploring the environment.

![](assets/beginning.gif)
### Middle of the training
At this stage, the agent has learned a valid policy and is able to locate the L3 zone. However it is still not very accurate.

![](assets/middle.gif)
### End of the training
The agent is able to locate the L3 slice accurately.

![](assets/end.gif)

## Citation 

If you use this code, please cite our work
```
@InProceedings{10.1007/978-3-030-87589-3_33,
	author="Laousy, Othmane
	and Chassagnon, Guillaume
	and Oyallon, Edouard
	and Paragios, Nikos
	and Revel, Marie-Pierre
	and Vakalopoulou, Maria",
	title="Deep Reinforcement Learning for L3 Slice Localization in Sarcopenia Assessment",
	booktitle="Machine Learning in Medical Imaging @ MICCAI 2021",
	year="2021",
	publisher="Springer International Publishing",
	address="Cham",
	pages="317--326",
	isbn="978-3-030-87589-3"
}
```
