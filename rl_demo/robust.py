import gymnasium as gym
import mlflow
from stable_baselines3 import PPO
from requests.exceptions import RequestException
from stable_baselines3.common.callbacks import BaseCallback
import os
# Force a 5-second timeout (Default is 120s)
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"
# Don't spend time retrying a dead server during setup
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"

class MLflowDummy:
    """A dummy object that pretends to be both the mlflow module and a run."""
    def __getattr__(self, name):
        # This catches log_param, log_metric, log_artifact, etc.
        return lambda *args, **kwargs: None
    def __enter__(self): return self
    def __exit__(self, *args): pass
    @property
    def info(self): 
        return type('obj', (object,), {'run_id': 'offline'})

from contextlib import contextmanager

@contextmanager
def safe_mlflow_run(*args, **kwargs):
    try:
        # Try to start a real run
        mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
        mlflow.set_experiment("CartPole_Example2")
        with mlflow.start_run(*args, **kwargs) as run:
            yield mlflow
    except Exception as e:
        # MLflow is down or unreachable
        print(f"⚠️ MLflow server unreachable at start: {e}")
        print("🚀 Proceeding with training anyway (logging disabled)...")
        # Yield 'None' or a dummy object so the 'with' block doesn't crash
        yield MLflowDummy()

# 2. Define a custom callback to log metrics every few steps
class RobustMLflowCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1000):
        self.log_freq = log_freq
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                latest_ep = self.model.ep_info_buffer[-1]
                
                try:
                    # Attempt to log to MLflow
                    mlflow.log_metric("episode_reward", latest_ep['r'], step=self.num_timesteps)
                except (RequestException, Exception) as e:
                    # MLflow is down? Just log to console and keep training!
                    if self.verbose > 0:
                        print(f"⚠️ MLflow Logging Failed at step {self.num_timesteps}. Error: {e}")
        return True


with safe_mlflow_run(run_name="CartPole_Robust_Run") as run:
    
    # Standard training setup
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device="cpu")
    run.log_param("algorithm", "PPO")
    run.log_param("learning_rate", 0.0003)
        
    
    # Your RobustCallback already handles per-step logging errors!
    callback = RobustMLflowCallback(log_freq=1000)
    
    model.learn(total_timesteps=50000, callback=callback)

    # Save the model
    model.save("ppo_cartpole_model")
    run.log_artifact("ppo_cartpole_model.zip")

    print("Training finished and logged to MLflow!")

