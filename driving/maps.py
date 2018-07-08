class Map:
    """
    Represents a particular configuration of the driving domain, represents
    the locations of obstacles in the environment, as well as the starting
    locations of all the cars, and the speed and direction of the NPC cars.

    This object is meant to be immutable, in the sense that multiple
    simulations can utilize it at once if needed.
    """

    def __init__(self, width, height, x=0.0, y=0.0, angle=0.0, speed=0.0):
        """
        Initializes the map, to be empty, with no cars or obstacles.

        :param width: the width of the map, in car lengths
        :param height: the height of the map, in car lengths
        :param x:
        :param y:
        :param angle:
        :param speed:
        """


        self.width = width
        self.height = height

        self.walls = []
        self.cars = []

    def wall(self, x0, y0, x1, y1):
        """
        Adds a new wall to the environment.

        :param x0:
        :param y0:
        :param x1:
        :param y1:
        :return:
        """
