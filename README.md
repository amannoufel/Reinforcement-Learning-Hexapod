
---

# 🤖 Hexapod Gait Learning with Reinforcement Learning

Welcome to the project where a six-legged robot (hexapod) learns to walk using deep reinforcement learning! 🐜💡  
Leveraging the Soft Actor-Critic (SAC) algorithm and PyBullet simulation, this project trains a hexapod to achieve stable and efficient locomotion.

---

## 🎯 Project Goals

-  Simulate a 3-DOF per leg (18-DOF total) hexapod robot.
-  Train the robot to walk using reinforcement learning.
-  Record and visualize the learned gaits.

---

## 🛠️ Technologies Used

-  Python 3.8+
-  PyBullet (physics simulation)
-  Stable-Baselines3 (reinforcement learning)
-  OpenAI Gym (environment structure)
-  NumPy (math and array operations)

---

## ⚙️ Setup 

1. **📥 Clone the Repository:**

   ```bash
   git clone https://github.com/amannoufel/Reinforcement-Learning-Hexapod.git
   cd Reinforcement-Learning-Hexapod
   ```

2. **📦 Install Required Packages:**

   ```bash
   pip install stable-baselines3[extra] pybullet gym numpy
   ```

3. **🛠️ Set the Path to Your Hexapod URDF File:**

   Edit the following line in both `hexapod_reinforcement.py` and `run_model.py`:

   ```python
   urdf_file = '/absolute/path/to/your/hexa.urdf'
   ```

   Ensure the path is correct and the URDF file exists.

---

## 🚀 Training the Hexapod

Run the training script:

```bash
python hexapod_reinforcement.py
```

This will:

-  Initialize the simulation environment.
-  Train the robot using the SAC algorithm.
-  Save checkpoints in the `./models/` directory.
-  Log training data for TensorBoard in `./tensorboard/`.

---

## 🎥 Evaluating the Trained Model

After training (or if you have a saved model), run:

```bash
python run_model.py
```

This will:

-  Load the trained model.
-  Render the hexapod walking in the GUI.
-  Save a video to the `./videos/` directory.

---

## 📊 Monitoring Training Progress

Track the reward and performance using TensorBoard:

```bash
tensorboard --logdir=./tensorboard/
```

---


## 🧠 Reward Function (Simplified)

The reward encourages:

-  Fast forward velocity.
-  Low energy consumption.
-  Stability (less body tilt).

```python
reward = forward_velocity - 0.01 * energy_used - 0.1 * tilt
```


##  Future Plans

-  Add rough terrain and obstacles.
-  Improve reward tuning.
-  Explore domain randomization for sim-to-real transfer.
-  Add GUI sliders for live control and testing.

---

**Train, Walk, Repeat! 🐜🤖**

---
