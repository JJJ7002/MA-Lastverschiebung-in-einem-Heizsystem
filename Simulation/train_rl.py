"""Train one of the two RL controllers (DQN). The training logic lives in learning/training.py.

  python train_rl.py                                            # power-level agent (control.method reinforcement_learning)
  python train_rl.py --config config/rl_training_offset.yaml    # offset agent (control.method reinforcement_learning_offset)
"""
from learning.training import main

if __name__ == '__main__':
    main()
