import numpy as np
from enum import IntEnum


class Map:
    """
    This class represents a map of a building,
    and provides methods for adding obstacles
    to this map and rendering them as a boolean
    occupancy map.
    """

    def __init__(self, width, height):
        """
        Initializes an empty map.

        :param width: the width of the map
        :param height: the height of the map
        """

        self.width = width
        self.height = height

        self._occupancy = np.zeros(shape=(height, width), dtype=int)

    def obstacle(self, x, y, width, height):
        """
        Adds a rectangular obstacle to the map

        :param x: the x coordinate of the top-left corner of the obstacle
        :param y: the y coordinate of the top-left corner of the obstacle
        :param width: the width of the obstacle
        :param height: the height of the obstacle
        """

        for x_pos in range(x, x + width):
            for y_pos in range(y, y + height):
                self._occupancy[y_pos, x_pos] = 1

    def occupancy(self):
        """
        Computes and returns the occupancy array corresponding to this map.

        :return: the occupancy array
        """

        return self._occupancy


class Action(IntEnum):
    """
    An enum describing the set of possible actions.
    """

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Task:
    """
    Represents a single task as a rectangular region.
    """

    def __init__(self, x, y, width, height):
        """
        Initializes the task.

        :param x: the x coordinate of the top left corner of the goal region
        :param y: the y coordinate of the top left corner of the goal region
        :param width: the width of the goal region
        :param height: the height of the goal region
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def goal(self, x, y):
        """
        Tests whether a given location is within the goal region.

        :param x: the x coordinate of the location
        :param y: the y coordinate of the location
        :return: True if the location is within the goal, False otherwise
        """

        dx = x - self.x
        dy = y - self.y

        return (0 <= dx < self.width) and (0 <= dy < self.height)


class Environment:
    """
    Represents a simulation of a 2D navigation task.

    The environment is represented by the occupancy map,
    provided, and tasks are defined by rectangular goal
    regions.  At any given time the agent has a location,
    but it also has a map of the environment that indicates
    whether a particular location has been seen, and whether
    it is occupied.  This map is updated as the agent
    explores the environment.
    """

    def __init__(self, occupancy, tasks={}, radius=10):
        """
        Initializes the simulation.

        :param occupancy: the true occupancy map, as a 2D numpy ndarray
        :param tasks: the set of named tasks, where each task is a Task object
        :param radius: the number of cells in each direction that the agent can sense
        """

        # Initialize occupancy map
        self._occupancy = occupancy

        self.height = occupancy.shape[0]
        self.width = occupancy.shape[1]

        # Initialize tasks
        self.tasks = tasks

        # Initialize agent position
        self.x = None
        self.y = None
        self.reset()

        # Initialize sensor radius
        self._radius = radius

        # Initialize sensor map
        self.map = np.zeros(shape=(self.height, self.width), dtype=int)
        self._sense()

    def reset(self):
        """
        Resets the current state of the environment

        We may eventually want more control over initial state distributions
        """

        while True:
            self.x = np.random.randint(0, self.width)
            self.y = np.random.randint(0, self.height)

            if not self._occupancy[self.y, self.x]:
                break

    def update(self, action):
        """
        Updates the environment based on the action provided.

        :param action: the action taken, a value of the Action enum
        """

        # Update position
        x = self.x
        y = self.y

        if Action.UP == action:
            y = self.y + 1
        elif Action.DOWN == action:
            y = self.y - 1
        elif Action.LEFT == action:
            x = self.x - 1
        elif Action.RIGHT == action:
            x = self.x + 1

        if (0 <= x < self.width) and (0 <= y < self.height) and not self._occupancy[y, x]:
            self.x = x
            self.y = y

        # Update occupancy map
        self.sense()

    def _sense(self):
        """
        Updates the sensor map to allow to include areas
        that have become visible to the agent.
        """

        x_start = max(self.x - self._radius, 0)
        y_start = max(self.y - self._radius, 0)
        x_end = min(self.x + self._radius + 1, self.width)
        y_end = min(self.y + self._radius + 1, self.height)

        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if self._occupancy[y, x]:
                    self.map[y, x] = 2
                else:
                    self.map[y, x] = 1


def interactive_test():
    """
    Starts an graphical, interactive simulation to allow
    for interactive testing of the navigation domain.

    Uses the Pyglet game engine.
    """

    # Construct map
    map = Map(50, 50)
    map.obstacle(10, 10, 4, 30)
    map.obstacle(36, 10, 4, 30)
    map.obstacle(10, 36, 30, 4)

    # Construct domain
    env = Environment(occupancy=map.occupancy(), radius=5)

    # Set the size in pixels of each drawn cell
    scale = 10

    # Set up the Pyglet window
    window = pg.window.Window(env.width * scale, env.height * scale)

    # Define vertex lists
    clear = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (65, 105, 225, 65, 105, 225, 65, 105, 225, 65, 105, 225))
    )

    blocked = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255))
    )

    hidden = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50))
    )

    agent = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (125, 250, 0, 125, 250, 0, 125, 250, 0, 125, 250, 0))
    )

    # Define rendering loop
    def on_draw():
        window.clear()

        # Draw visible map
        for x in range(env.width):
            for y in range(env.height):
                pg.gl.glLoadIdentity()
                pg.gl.glScalef(scale, scale, 1)
                pg.gl.glTranslatef(x, y, 0)

                if 2 == env.map[y, x]:
                    blocked.draw(pg.gl.GL_QUADS)
                elif 1 == env.map[y, x]:
                    clear.draw(pg.gl.GL_QUADS)
                else:
                    hidden.draw(pg.gl.GL_QUADS)

        # Draw agent
        pg.gl.glLoadIdentity()
        pg.gl.glScalef(scale, scale, 1)
        pg.gl.glTranslatef(env.x, env.y, 0)

        agent.draw(pg.gl.GL_QUADS)

    window.on_draw = on_draw

    # Define key handler
    def on_key_press(symbol, modifier):
        if pg.window.key.UP == symbol:
            env.update(Action.UP)
        elif pg.window.key.DOWN == symbol:
            env.update(Action.DOWN)
        elif pg.window.key.LEFT == symbol:
            env.update(Action.LEFT)
        elif pg.window.key.RIGHT == symbol:
            env.update(Action.RIGHT)

    window.on_key_press = on_key_press

    pg.app.run()
