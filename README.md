# Deep Reinforcement Learning for CT slice detection

This repository contains the code for the paper: Deep Reinforcement Learning for L3 Slice Localization in Sarcopenia Assessment (arXiv:2107.12800v1)

A Deep Q-Network is trained to detect a single slice in a CT volume initially projected using MIP.
The reinfocement learning agent has been applied to the task of localizing the L3 slice which corresponds to the computation of sarcopenia scores.

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
@article{laousy2021deep,
      title={Deep Reinforcement Learning for L3 Slice Localization in Sarcopenia Assessment}, 
      author={Othmane Laousy and Guillaume Chassagnon and Edouard Oyallon and Nikos Paragios and Marie-Pierre Revel and Maria Vakalopoulou},
      year={2021},
      eprint={2107.12800},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```