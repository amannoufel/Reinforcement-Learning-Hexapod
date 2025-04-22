import os
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import Monitor as GymMonitor

class HexapodEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, urdf_path: str, renders: bool = False, time_step: float = 1./240., num_substeps: int = 8):
        super(HexapodEnv, self).__init__()
        self.urdf_path = urdf_path
        self.renders = renders
        self.time_step = time_step
        self.num_substeps = num_substeps
        self.max_steps = 1000

        self.client = p.connect(p.GUI if self.renders else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(numSolverIterations=150)

        self.n_joints = 18
        # Corrected obs_dim calculation
        self.obs_dim = self.n_joints * 2 + 4 + 3  # angles, velocities, orientation (4), linvel (3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)

        self.robot = None
        self.step_counter = 0

        self.reset()

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        start_pos = [0, 0, 0.2]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found at path: {self.urdf_path}")

        self.robot = p.loadURDF(self.urdf_path, start_pos, start_ori, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.client)

        for j in range(self.n_joints):
            p.resetJointState(self.robot, j, targetValue=0, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, force=0, physicsClientId=self.client)

        self.step_counter = 0
        return self._get_obs()

    def step(self, action):
        for j in range(self.n_joints):
            joint_info = p.getJointInfo(self.robot, j, physicsClientId=self.client)
            lower, upper = joint_info[8], joint_info[9]
            target = lower + (action[j] + 1) * 0.5 * (upper - lower)
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=target, force=5, physicsClientId=self.client)

        for _ in range(self.num_substeps):
            p.stepSimulation(physicsClientId=self.client)

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self.step_counter >= self.max_steps or self._fell_over()
        self.step_counter += 1
        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'rgb_array':
            _, _, px, _, _ = p.getCameraImage(width=640, height=480, physicsClientId=self.client)
            return np.array(px)
        return None

    def close(self):
        p.disconnect(self.client)

    def _get_obs(self):
        angles, velocities = [], []
        for j in range(self.n_joints):
            state = p.getJointState(self.robot, j, physicsClientId=self.client)
            angles.append(state[0])
            velocities.append(state[1])

        pos, ori = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        linvel, _ = p.getBaseVelocity(self.robot, physicsClientId=self.client)
        return np.concatenate([angles, velocities, ori, linvel]).astype(np.float32)

    def _compute_reward(self, obs, action):
        linvel = obs[-3:]
        forward_vel = linvel[0]
        energy = np.square(action).mean()
        ori = obs[self.n_joints*2:self.n_joints*2+4]
        euler = p.getEulerFromQuaternion(ori)
        stability = abs(euler[0]) + abs(euler[1])
        return forward_vel - 0.01 * energy - 0.1 * stability

    def _fell_over(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        return pos[2] < 0.05

def make_env(urdf_path, rank, seed=0):
    def _init():
        env = HexapodEnv(urdf_path, renders=False)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    return _init

if __name__ == '__main__':
    urdf_file = '/home/aman/Desktop/mithi_hexa/hexapod_InverseKinematics/hexa.urdf'

    if not os.path.exists(urdf_file):
        raise FileNotFoundError(f"URDF file not found at {urdf_file}")

    num_cpu = 4
    vec_env = SubprocVecEnv([make_env(urdf_file, i) for i in range(num_cpu)])
    vec_env = VecMonitor(vec_env)

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/', name_prefix='sac_hexapod')

    model = SAC('MlpPolicy', vec_env, verbose=1, tensorboard_log='./tensorboard/')

    # Warm-up rollout to determine observation shape
    obs = vec_env.reset()
    if len(obs.shape) == 2:
        obs_shape = obs.shape[1]
        if obs_shape != vec_env.observation_space.shape[0]:
            raise ValueError(f"Mismatch in observation shape: env gives {obs_shape}, expected {vec_env.observation_space.shape[0]}")

    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
    model.save('models/sac_hexapod_final')

    eval_env = HexapodEnv(urdf_file, renders=True)
    video_env = GymMonitor(eval_env, './videos/', force=True)
    obs = video_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = video_env.step(action)
        if done:
            obs = video_env.reset()
    video_env.close()
    print("Training and gait capture complete. Videos saved in ./videos/")