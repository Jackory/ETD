# About

This repository implements **Episodic Novelty Through Temporal Distance(ETD)**, an exploration method for reinforcement learning that has been found to be particularly effective in Contextual MDPs(CMDP). More details can be found in the original paper([ICLR25 Under Review](https://openreview.net/pdf?id=I7DeajDEx7)). In case you are mainly interested in the implementation of ETD, its major components can be found at `src/algo/intrinsic_rewards/tdd.py.`

# Install

### Basic Installation

```bash
conda create -n etd python=3.9
conda activate etd
pip install -r requirements.txt
```

### Miniworld

```bash
git submodule init
git submodule update
cd src/env/gym_miniworld
pip install pyglet==1.5.11
pip install -e .
```

### Usage

### Train ETD on MiniGrid

Run the below command in the root directory of this repository to train a ETD agent in the standard *DoorKey-8x8* (MiniGrid) environment.

```bash
PYTHONPATH=./ python3 src/train.py \\
  --int_rew_source=TDD \\
  --env_source=minigrid \\
  --game_name=DoorKey-8x8 \\
  --int_rew_coef=1e-2
```

We also provide scripts`(*.sh)` to run the experiments of our method. The hyperparameter setup can be found in our paper.



