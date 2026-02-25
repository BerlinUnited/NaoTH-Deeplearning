import gymnasium as gym
import mlflow
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# 1. Setup MLflow Tracking URI
mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
mlflow.set_experiment("CartPole_Example2")

# 2. Define a custom callback to log metrics every few steps
class MLflowLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1000):
        self.log_freq = log_freq
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log metrics from the logger (rollout/ep_rew_mean is a key RL metric)
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                latest_ep = self.model.ep_info_buffer[-1]
                mlflow.log_metric("episode_reward", latest_ep['r'], step=self.num_timesteps)
        return True

# 3. Training Logic
with mlflow.start_run():
    env = gym.make("CartPole-v1")
    
    # Log hyperparameters
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device="cpu")
    mlflow.log_param("user", os.environ.get("MLFLOW_USER"))
    mlflow.log_param("algorithm", "PPO")
    mlflow.log_param("learning_rate", 0.0003)

    # Train with our callback
    callback = MLflowLoggingCallback()
    model.learn(total_timesteps=10000, callback=callback)

    # Save the model
    model.save("ppo_cartpole_model")
    mlflow.log_artifact("ppo_cartpole_model.zip")

    print("Training finished and logged to MLflow!")