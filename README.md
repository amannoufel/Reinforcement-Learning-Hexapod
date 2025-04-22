```markdown
# 🐜 Hexapod Gait Learning via Reinforcement Learning (SAC + PyBullet)

This project demonstrates training a **hexapod robot** to walk using **Deep Reinforcement Learning (DRL)**. The hexapod is simulated in **PyBullet**, and trained using the **Soft Actor-Critic (SAC)** algorithm from `stable-baselines3`.

## 📽️ Demo

- The final trained model demonstrates the robot's gait and movement in a PyBullet simulation.
- Videos are saved in the `./videos/` directory.

## 🧠 Algorithm Used

- **Soft Actor-Critic (SAC)**
  - Off-policy actor-critic method
  - Works well for continuous action spaces
  - Stable and sample-efficient

## 🛠️ Tech Stack

- Python 🐍
- [PyBullet](https://github.com/bulletphysics/bullet3)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gym](https://github.com/openai/gym)
- Numpy
- Custom URDF-based hexapod robot

## 📁 Project Structure

```
📦 Hexapod-RL
├── hexapod_reinforcement.py    # Training script with PyBullet environment
├── run_model.py                # Script to load and evaluate the trained model
├── models/                     # Checkpoints and final model saved here
├── videos/                     # Videos recorded during evaluation
└── README.md                   # This file
```

## 📌 Environment Features

- 18-DOF hexapod robot (3 joints per leg)
- Observation space includes joint angles, velocities, base orientation, and linear velocity
- Reward function encourages:
  - Forward walking speed
  - Energy efficiency
  - Postural stability

## 🚀 Training Instructions

1. **Make sure the URDF path is valid:**

   Edit this path in both files:
   ```python
   urdf_file = 'path to the hexa.urdf file'
   ```

2. **Run the training script:**
   ```bash
   python hexapod_reinforcement.py
   ```

   - This will spawn 4 parallel environments for faster training
   - Checkpoints will be saved in `./models/`
   - TensorBoard logs are stored in `./tensorboard/`

3. **Evaluate the trained model and generate video:**
   ```bash
   python run_model.py
   ```

   - Videos will be saved in the `./videos/` folder

## 📊 TensorBoard (Optional)

To visualize training logs:
```bash
tensorboard --logdir=./tensorboard/
```

## 🔒 Dependencies

Install the required Python packages:

```bash
pip install stable-baselines3[extra] pybullet gym numpy
```

## 🧠 Notes

- Make sure your URDF file exists and is correctly formatted.
- You can tweak the reward function, joint limits, or substeps to optimize behavior.
- This setup is ideal for experimenting with continuous action space learning for multi-legged locomotion.

