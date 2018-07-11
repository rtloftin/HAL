"""
We may get rid of this file.
"""


class Wall:
    """
    A line segment representing a wall in the environment.
    """

    def __init__(self, x0, y0, x1, y1):
        """
        Initializes the fields of the object

        :param x0: the first x coordinate of the segment
        :param y0: the first y coordinate of the segment
        :param x1: the second x coordinate of the segment
        :param y1: the second y coordinate of the segment
        """

        self.x0 = x0
        self.y0 - y0
        self.x1 = x1
        self.y1 = y1


class Car:
    """
    Represents the initial position, direction, and speed of an NPC car
    """

    def __init__(self, x, y, theta, speed):
        """
        Initializes the fields of the object

        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the direction of the car
        :param speed: the speed of the car
        """

        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed


class Map:
    """
    Represents a particular configuration of the driving domain, represents
    the locations of obstacles in the environment, as well as the starting
    locations of all the cars, and the speed and direction of the NPC cars.

    This object is meant to be immutable, in the sense that multiple
    simulations can utilize it at once if needed.
    """

    def __init__(self, width, height, x=0.0, y=0.0, theta=0.0, speed=0.0):
        """
        Initializes the map, to be empty, with no cars or obstacles.

        :param width: the width of the map, in car lengths
        :param height: the height of the map, in car lengths
        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the initial angle of the car
        :param speed: the initial speed of the car
        """

        self.width = width
        self.height = height

        self.start_x = x
        self.start_y = y
        self.start_theta = theta
        self.start_speed = speed

        self.walls = []
        self.cars = []

    def wall(self, x0, y0, x1, y1):
        """
        Adds a new wall to the environment.

        :param x0: the first x coordinate of the wall
        :param y0: the first y coordinate of the wall
        :param x1: the second x coordinate of the wall
        :param y1: the second y coordinate of the wall
        """

        self.walls.append(Wall(x0, y0, x1, y1))

    def car(self, x, y, theta, speed):
        """
        Adds a new NPC car to the environment.

        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the direction of the car
        :param speed: the speed of the car
        """

        self.cars.append(Car(x, y, theta, speed))
