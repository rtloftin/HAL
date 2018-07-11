"""
Defines the core classes representing a driving environment.

Specific environments will extend these classes.

WE MAY WANT TO REORGANIZE THIS API AT SOME POINT
"""

import pyglet as pg
import math
import numpy as np
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


class Environment:
    """
    Represents a simulation of the driving domain.

    This will include a sensor model as well.
    """

    def __init__(self, width, height):
        """
        Initializes the car's position and orientation.
        """

        self.width = width
        self.height = height

        # Initialize wall list
        self.walls = []

        # Initialize collision model
        self._collision = Collision(env, 0.5)

        # Initialize NPC car list
        self.cars = []

        # Initialize the agent's car
        self.car = None

        # Initialize the sensor model
        self._sensor = None

        # Initialize task dictionary
        self.tasks = {}

        # Initialize the current task
        self._task = None

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

    def _set_driver_car(self, x, y, theta, speed):
        """
        Resets the position of the agent's car.

        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the initial direction of the car
        :param speed: the initial speed of the car
        """

        self.car = Car(x, y, theta, speed)

    def _set_sensor(self, radius, resolution):
        """
        Sets the sensor model for the agent's car.

        :param radius: the range of the sensor
        :param resolution: the angular resolution of the sensor, the number of sensor cones
        """

        self._sensor = Sensor(self, radius, resolution)

    def _add_npc_car(self, x, y, theta, speed):
        """
        Adds a new NPC car.

        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the direction of the car
        :param speed: the speed of the car
        """

        self.cars.append(NPCCar(x, y, theta, speed))

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

        self.car = None
        self.cars = []

        self._task.reset(self)

        return self.car.speed, self.car.phi, self._sensor.update, 0.0, False

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
