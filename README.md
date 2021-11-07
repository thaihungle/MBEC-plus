# MBEC-plus
source code for ***Model-Based Episodic Memory Induces Dynamic Hybrid Controls***  
arXiv version: https://arxiv.org/abs/2111.02104  
code reference:
- https://github.com/jsrimr/pytorch-rainbow
- https://github.com/Kaixhin/Rainbow

# Setup  
- torch 1.4
- Install gym[atari] https://gym.openai.com/envs/#atari

```
mkdir model
mkdir runs
```

# Atari

- training
``` 
python main_mem.py --env PongNoFrameskip-v4
```
check log training in /runs
- testing  
use **--evaluate**  


