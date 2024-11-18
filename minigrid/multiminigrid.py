import hashlib
import math
import random
from abc import abstractmethod
from typing import Any, Iterable, SupportsFloat, TypeVar, Dict, List, Tuple, Self

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.core.agents import Agent, AgentList
from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Point, WorldObj, Goal, Box

T = TypeVar("T")


class MultiMiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        mission_space: MissionSpace,
        #agent_nb: int,
        #agent_initial_pos: List[Point],
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 3,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        # Initialize mission
        self.mission = mission_space.sample()

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.agent_view_size = agent_view_size
        self.agent_pov = agent_pov
        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            }
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = width
        self.height = height

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        # NEW: Agent number
        self.agents = AgentList([])
        #self.agent_nb = agent_nb
        #agent_lst = []
        #self.agent_initial_pos = agent_initial_pos

        # Initial Agents
        #for i in range(self.agent_nb):
        #    # Current position and direction of the agent
        #    agent_pos = agent_initial_pos[i]
        #    agent_dir = 0
        #    agent = Agent(name=f"agent_{i}", idx=i, pos=agent_pos, drt=agent_dir, agent_pov=agent_pov, agent_view_size=agent_view_size)
        #    agent_lst.append(agent)
 
        #self.agents = AgentList(agent_lst)

        # Current grid and mission and carrying
        self.grid = Grid(width, height)
        self.carrying = None

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        # Reinitialize episode-specific variables
        self._gen_grid(self.width, self.height)

        # Reintialize agent
        [agent.reset() for agent in self.agents]

        # Generate a new random grid at the start of each episode

        # These fields should be defined by _gen_grid
        assert (
            all([agent.pos >= (0, 0) and agent.drt >= 0 for agent in self.agents])
        )

        # Check that the agent doesn't overlap with an object
        for agent in self.agents:
            start_cell = self.grid.get(*agent.pos)
            assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), *[agent.pos for agent in self.agents], *[agent.drt for agent in self.agents]]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        images = []
        for agent in self.agents:
            grid, vis_mask = agent.gen_obs_grid(grid=self.grid)

            # Encode the partially observable view into a numpy array
            image = grid.encode(vis_mask)

            images.append(image)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": images, "direction": [agent.drt for agent in self.agents], "mission": self.mission}

        return obs

    
    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> List[T]:
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self) -> str:
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int
    ) -> Tuple[int, int]:
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )

    def place_obj(
        self,
        obj: WorldObj | None,
        top: Point | None = None,
        size: Tuple[int, int] | None = None,
        reject_fn=None,
        pos: Point | None = None,
        max_tries=math.inf,
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1
            if pos is None:
                pos = (
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                )

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if pos in self.agents:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)


    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}


    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid(grid=self.grid)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)
        for agent in self.agents:
            # Compute which cells are visible to the agent
            _, vis_mask = agent.gen_obs_grid(grid=self.grid)

            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = agent.dir_vec
            r_vec = agent.right_vec
            top_left = (
                agent.pos
                + f_vec * (self.agent_view_size - 1)
                - r_vec * (self.agent_view_size // 2)
            )

            # For each cell in the visibility mask
            for vis_j in range(0, self.agent_view_size):
                for vis_i in range(0, self.agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agents,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)


    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()


class MultiMiniGrid10x10(MultiMiniGridEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_size=10, *args, **kwargs)

    def _gen_grid(self, width, height):
        self.width = width
        self.height = height

        # Generate the grid
        self.grid = Grid(width, height)

        # Generate the walls first
        #self.grid.wall_rect(0, 0, width, height)
        #self.grid.horz_wall(0, 0)
        #self.grid.horz_wall(0, height - 1)
        #self.grid.vert_wall(0, 0)
        #self.grid.vert_wall(width - 1, 0)

        obstacles = [((0, 0), (3, 3)), ((0, 6), (3, 9)), ((6, 0), (9, 3)), ((6, 6), (9, 9))]
        agent_poses = [(4, 0), (5, 9)]
        agent_dirs = [0, 3]
        for obs in obstacles:
            start = obs[0]
            end = obs[1]
            for i in range(start[0], end[0] + 1):
                for j in range(start[1], end[1] + 1):
                    self.grid.set(i, j, Box("blue"))
        
        for i, agent_pos in enumerate(agent_poses):
            self.place_obj(None, pos=agent_pos)
            agent = Agent(
                    name=f"agent_{i}",
                    idx=i,
                    pos=agent_pos,
                    drt=agent_dirs[i],
                    agent_view_size=3,
                    see_through_walls=True,
                    agent_pov=True,
                )
            self.agents.append(agent)

