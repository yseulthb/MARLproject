from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class DoorKeyEnv(MultiGridEnv):
    """
    Environment with a door and key, sparse reward.
    Similar to DoorKeyEnv in 
        https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/doorkey.py
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(color="green", reward=1), 4, 3) #self.put_obj(Goal(color="green", reward=1), 4, 3)

        # Create a vertical splitting wall
        # plitIdx = self._rand_int(2, width - 2)
        splitIdx = 3
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        # self.place_agents(size=(splitIdx, height))

        # Place a door in the wall
        # doorIdx = self._rand_int(1, width - 2)
        doorIdx = 4 #orig: 1 (vs 4)
        self.put_obj(Door(color="yellow", state=Door.states.open), splitIdx, doorIdx)

        # Place a yellow key on the left side
        #self.place_obj(obj=Key("yellow"), top=(0, 0), size=(width -2, height -3))

        #self.agent_spawn_kwargs = {}
        #self.place_agents(**self.agent_spawn_kwargs)

        #self.agents[0].set_position([1, 1])
        #self.agents[1].set_position([3, 2])
