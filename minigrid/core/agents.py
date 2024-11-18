import random
from typing import Any, Iterable, SupportsFloat, TypeVar, Dict, List, Tuple, Self

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Point, WorldObj, Goal, Box

T = TypeVar("T")

class Agent:
    def __init__(
        self,
        name: str,
        idx: int = 0,
        pos: Point = (0, 0),
        drt: int = 0,
        agent_view_size: int = 3,
        see_through_walls: bool = False,
        agent_pov: bool = False,
        color: str | None = None,
    ):
        """
        Agent class, can be indexed by name or index number
        @params: 
            name: name of the agent
            idx: index of the agent
            pos: initial position of the agent
            drt: initial direction of the agent
        """
        self.initial_name = self.name = name
        self.initial_idx = self.idx = idx
        self.initial_pos = self.pos = pos
        self._check_drt(drt)
        self.initial_drt = self.drt = drt

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        self.see_through_walls = see_through_walls
        self.agent_pov = agent_pov
        self.carrying = None
        if color is None:
            self.color = random.choice(COLOR_NAMES)
        else:
            self.color = color

    @staticmethod
    def _check_drt(drt: int):
        assert drt in [0, 1, 2, 3], "Direction must be 0, 1, 2, or 3"

    def reset(self):
        self.name = self.initial_name
        self.idx = self.initial_idx
        self.pos = self.initial_pos
        self.drt = self.initial_drt

    @property
    def dir_vec(self) -> np.ndarray:
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert (self.drt >= 0 and self.drt < 4), f"Invalid agent_dir: {self.drt} is not within range(0, 4)"

        return DIR_TO_VEC[self.drt]

    @property
    def right_vec(self) -> np.ndarray:
        """
        Get the vector pointing to the right of the agent.
        """
        dx, dy = self.dir_vec

        return np.array((-dy, dx))

    @property
    def front_pos(self) -> np.ndarray:
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def get_view_coords(self, i, j) -> np.ndarray:
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """
        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)
        return vx, vy

    def __str__(self):
        return f"{self.name}, pos: "

    def pprint_grid(self, grid: Grid = None):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        if self.pos is None or self.drt is None or grid is None:
            raise ValueError(
                "The environment hasn't been `reset` therefore the `pos`, `drt` or `grid` are unknown."
            )

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""

        # check if self.agent_pos & self.agent_dir is None
        # should not be after env is reset
        if self.pos is None:
            return super().__str__()

        for j in range(grid.height):
            for i in range(grid.width):
                if i == self.pos[0] and j == self.pos[1]:
                    output += 2 * AGENT_DIR_TO_STR[self.drt]
                    continue

                tile = grid.get(i, j)

                if tile is None:
                    output += "  "
                    continue

                if tile.type == "door":
                    if tile.is_open:
                        output += "__"
                    elif tile.is_locked:
                        output += "L" + tile.color[0].upper()
                    else:
                        output += "D" + tile.color[0].upper()
                    continue

                output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

            if j < grid.height - 1:
                output += "\n"

        return output

    def get_view_exts(self, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        """

        agent_view_size = agent_view_size or self.agent_view_size
        agent_pos = self.pos
        agent_dir = self.drt

        # Facing right
        if agent_dir == 0:
            topX = agent_pos[0]
            topY = agent_pos[1] - agent_view_size // 2
        # Facing down
        elif agent_dir == 1:
            topX = agent_pos[0] - agent_view_size // 2
            topY = agent_pos[1]
        # Facing left
        elif agent_dir == 2:
            topX = agent_pos[0] - agent_view_size + 1
            topY = agent_pos[1] - agent_view_size // 2
        # Facing up
        elif agent_dir == 3:
            topX = agent_pos[0] - agent_view_size // 2
            topY = agent_pos[1] - agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    def gen_obs_grid(self, agent_view_size=None, grid: Grid = None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        sub_grid = grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.drt + 1):
            sub_grid = sub_grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = sub_grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(sub_grid.width, sub_grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = sub_grid.width // 2, sub_grid.height - 1
        if self.carrying:
            sub_grid.set(*agent_pos, self.carrying)
        else:
            sub_grid.set(*agent_pos, None)

        return sub_grid, vis_mask

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()

        obs_grid, _ = Grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        assert world_cell is not None

        return obs_cell is not None and obs_cell.type == world_cell.type

    def __eq__(self, other: Point | Self):
        if isinstance(other, tuple):
            return self.pos == other
        elif isinstance(other, Agent):
            return self.pos == other.pos
        else:
            raise TypeError("Can only compare to Point or Agent")


class AgentList:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def __getitem__(self, idx: str | int | Tuple[int, int]):
        if isinstance(idx, str):
            for agent in self.agents:
                if agent.name == idx:
                    return agent
            else:
                raise KeyError(f"Agent {idx} not found")
        elif isinstance(idx, int):
            return self.agents[idx]
        elif isinstance(idx, tuple):
            for agent in self.agents:
                if agent.pos == idx:
                    return agent
            else:
                return None

    def __contains__(self, pos: Point):
        for agent in self.agents:
            if agent == pos:
                return True
        else:
            return False

    @property
    def poses(self):
        # All positions
        return [agent.pos for agent in self.agents]

    @property
    def drts(self):
        # All directions
        return [agent.drt for agent in self.agents]

    def append(self, agent: Agent):
        self.agents.append(agent)
