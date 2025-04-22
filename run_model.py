from stable_baselines3 import SAC
from gym.wrappers import Monitor as GymMonitor
from hexa_reinforcement_working import HexapodEnv
import os

if __name__ == "__main__":
    urdf_file = '/home/aman/Desktop/mithi_hexa/hexapod_InverseKinematics/hexa.urdf'

    if not os.path.exists(urdf_file):
        raise FileNotFoundError(f"URDF file not found at {urdf_file}")

    # Load the latest checkpoint
    model_path = 'models/sac_hexapod_final.zip'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    eval_env = HexapodEnv(urdf_file, renders=True)
    model = SAC.load(model_path, env=eval_env)

    # Optionally, wrap the environment to record videos
    video_env = GymMonitor(eval_env, './videos/', force=True)

    # Run the model
    obs = video_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = video_env.step(action)
        if done:
            obs = video_env.reset()

    video_env.close()
    print("Model evaluation complete. Videos saved in ./videos/")