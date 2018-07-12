"""
Defines the core classes representing a driving environment.

Specific environments will extend these classes.

WE MAY WANT TO REORGANIZE THIS API AT SOME POINT
"""

import math
import sensor.Sensor
import collision.Collision


class DriverCar:
    """
    Represents a car that can be controlled by the agent.
    """

    def __init__(self, x, y, theta, speed):
        """
        Initializes the position, direction, and speed of the car

        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the initial direction of the car
        :param speed: the initial speed of the car
        """

        # Define vehicle parameters
        self._min_steering = -math.pi / 10
        self._max_steering = math.pi / 10
        self._max_acceleration = 0.1
        self._min_acceleration = -0.1
        self._max_speed = 1.0
        self._min_speed = 0.0
        self._threshold = 0.05
        self._wheelbase = 0.5

        # Initialize state
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        self.phi = 0.0

    def update(self, acceleration, steering, delta):
        """
        Updates the state of the car.

        :param acceleration: acceleration of the vehicle (negative for braking)
        :param steering: steering rate (positive for left negative for right)
        :param delta: the size of the time step for this update
        """

        # Clip acceleration and steering rate
        acceleration = max(self._min_acceleration, min(self._max_acceleration, acceleration))
        steering = max(self._min_steering, min(self._max_steering, steering))

        # Update speed, steering angle
        self.speed = max(self._min_speed, min(self._max_speed, self.speed + acceleration * delta))
        self.phi = max(self._min_steering, min(self._max_steering, self.phi + steering * delta))

        # Update orientation
        if not (-self._threshold < self.phi < self._threshold):
            self.theta = self.theta - delta * self.speed * math.sin(self.phi) / self._wheelbase

        # Update position
        self.x -= delta * (self.speed * math.sin(self.theta))
        self.y += delta * (self.speed * math.cos(self.theta))


class NPCCar:
    """
    Represents an NPC car, which just moves in a straight line.
    """

    def __init__(self, x, y, theta, speed):
        """
        Initializes the position, direction, and speed of an NPC car.

        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the direction of the car
        :param speed: the speed of the car
        """

        # Initialize state
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed

    def update(self, delta):
        """
        Updates the position of the car.

        :param delta: the size of the time step for the update
        """

        self.x -= delta * math.sin(self.theta) * self.speed
        self.y += delta * math.cos(self.theta) * self.speed


class Wall:
    """
    A wall represented as a line segment
    """

    def __init__(self, x0, y0, x1, y1):
        """
        Initializes the wall

        :param x0: the first x coordinate of the segment
        :param y0: the first y coordinate of the segment
        :param x1: the second x coordinate of the segment
        :param y1: the second y coordinate of the segment
        """

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

class Simulation:
    """
    Represents the state of the driver's car and
    of the NPC cars.
    """

    def __init__(self, car, cars=[]):
        """
        Initializes the simulation object.

        :param car: an object representing the agent's car
        :param cars: a list of objects representing the NPC cars
        """

        self.car = car
        self.cars = cars

    def update(self, acceleration, steering, delta):
        """
        Updates the positions of all the cars.
        """

class Environment:
    """
    Represents a simulation of the driving domain.

    This will include a sensor model as well.
    """

    def __init__(self, width, height, radius, resolution):
        """
        Initializes the environment.

        :param width:
        :param height:
        :param radius:
        :param resolution:
        """

        self.width = width
        self.height = height

        self._radius = radius
        self._resolution = resolution

        self._car = None
        self._cars = []

        self.walls = []

        self.tasks = {}
        self._task = None

        self._sensor_model = None
        self._collision_model = None

        self._sensor_vector = None
        self._is_collision = False

    def _add_task(self, task, name):
        """
        Adds a new task to the environment.

        :param task: the task object
        :param name: the name of the task
        """

        self.tasks[name] = task

    def _add_wall(self, x0, y0, x1, y1):
        """
        Adds a new wall to the environment.

        :param x0: the first x coordinate of the wall
        :param y0: the first y coordinate of the wall
        :param x1: the second x coordinate of the wall
        :param y1: the second y coordinate of the wall
        """

        self.walls.append(Wall(x0, y0, x1, y1))

    def set_task(self, name):
        """
        Sets the current task.

        :param name: the name of the task
        """

        self._task = self.tasks[name]

    def reset(self):
        """
        Resets the environment to a random initial state.

        Only works if there is a current task set, since the current
        task defines how the environment will be initialized.
        """

        if self._task is not None:
            car, cars = self._task.reset()

            self._simulation = Simulation(car, cars)
            self._sensor = Sensor(self, self._simulation, self._radius, self._resolution)
            self._collision = Collision(self, self._simulation, 0.5)

    def update(self, acceleration, steering, delta):
        """
        Updates the state of the environment according to the
        dynamics equations.

        :param acceleration: acceleration of the vehicle (negative for braking)
        :param steering: steering rate (positive for left negative for right)
        :param delta: the size of the time step for this update
        """

        # Update driver car
        self.car.update(acceleration, steering, delta)

        # Update NPC cars
        for car in self.cars:
            car.update()

        # Check for collisions
        if self._collision.update():
            reward = -1.0
            done = True
        elif self._task.complete():
            reward = 1.0
            done = true
        else:
            reward = 0.0
            done = False

        # Return new state, reward, and completion
        return self.car.speed, self.car.phi, self._sensor.update(), reward, done

    def get_speed(self):
        if self._car is not None:
            return self._car.speed

        return 0.0

    def get_steering(self):
        if self._car is not None:
            return self._car.phi

        return 0.0

    def get_sensor(self):
        return self._sensor_vector

    def get_reward(self):
        return self._reward

    def check_complete(self):
        return self._complete
