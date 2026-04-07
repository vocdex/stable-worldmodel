import warnings                        
import atexit                                                                      
import mujoco.glfw                                        
atexit.register(lambda: None)  # doesn't fix it                                    
                                              
warnings.filterwarnings("ignore")        
import stable_worldmodel as swm
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.utils import record_video_from_dataset
import os
from stable_worldmodel.envs.ogbench import ExpertPolicy

env_type = 'octuple'  # 'single', 'double', or 'quadruple'
world = swm.World('swm/OGBCube-v0', num_envs=1, env_type=env_type,
                    image_shape=(224, 224), mode='data_collection', visualize_info=False)
policy = ExpertPolicy(seed=0, action_noise=0.1, policy_type='plan_oracle')
world.set_policy(policy)

# Record 100 episodes to HDF5
world.record_dataset(
    dataset_name=f"ogbench_{env_type}",
    episodes=10,
    seed=0
)
world.close()

# Load the dataset
dataset=HDF5Dataset(
    name=f'ogbench_{env_type}',
    frameskip=1,
    num_steps=100,
    keys_to_load=['pixels', 'action']
)

out_path = f'./demo_videos/ogbench_{env_type}/'
os.makedirs(out_path, exist_ok=True)
record_video_from_dataset(
    video_path=out_path,
    dataset=dataset,
    episode_idx=[0,1,2,3,4,5,6,7,8,9],  # Record the first 10 episodes
    max_steps=100,
    fps=30,
    viewname='pixels',       # Can also be a list: ['pixels', 'goal']
    fmt='gif'                 # Can be 'mp4' or 'gif'
)

