from minigrid.multiminigrid import MultiMiniGridEnv, MultiMiniGrid10x10
from minigrid.core.mission import MissionSpace
import time

env = MultiMiniGrid10x10(
    mission_space=MissionSpace(lambda :"Junction"),
    render_mode="human",
)

obs, _ = env.reset()
while True:
    state, reward, terminated, truncated, _ = env.step(env.random_actions())
    print(state, reward, terminated, truncated)
    time.sleep(0.1)
    if terminated or truncated:
        break
