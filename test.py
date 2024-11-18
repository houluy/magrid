from minigrid.multiminigrid import MultiMiniGridEnv, MultiMiniGrid10x10
from minigrid.core.mission import MissionSpace
import time

env = MultiMiniGrid10x10(
    mission_space=MissionSpace(lambda :"Junction"),
    render_mode="human",
)

obs, _ = env.reset()
print(obs)
while True:
    env.render()
    time.sleep(1)

