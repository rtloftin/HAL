import math


def point(cx, cy, t1, t2, r, px, py):
    """
    Computes the distance to a single point within the sensor cone

    :param cx: x coordinate of the car
    :param cy: y coordinate of the car
    :param t1: the start angle of the sensor
    :param t2: the end angle of the sensor
    :param r:  the range of the sensor
    :param px: x coordinate of the target
    :param py: y coordinate of the target
    :return: the scaled distance to the target if it is in range, 1 otherwise
    """

    x = cx - px
    y = py - cy
    d = math.sqrt(x * x + y * y)
    theta = math.atan(x / y)

    if t1 <= theta <= t2:
        return min(d / r, 1.0)

    return 1.0


def intersect(cx, cy, t, p0x, p0y, p1x, p1y):
    """
    Computes the distance to the intersection between a sensor vector
    and the given line segment.

    :param cx: the x coordinate of the vehicle
    :param cy: the y coordinate of the vehicle
    :param t: the angle of the sensor vector
    :param r: the range of the sensor
    :param p0x: the first x coordinate of the line segment
    :param p0y: the first y coordinate of the line segment
    :param p1x: the second x coordinate of the line segment
    :param p1y: the second y coordinate of the line segment
    :return: the scaled distance to the intersection if it exists, 1 otherwise
    """

    a = p1x - p0x
    b = r * math.sin(t)
    c = p1y - p0y
    d = -r * math.cos(t)
    e = cx - p0x
    f = cy - p0y

    det = (a * d) - (b * c)

    if 0.0 == det:
        return 1.0

    t = ((d * e) - (b * f)) / det
    u = ((a * f) - (c * e)) / det

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return u

    return 1.0


def segment(cx, cy, t1, t2, r, p0x, p0y, p1x, p1y):
    """
    Computes the minimum distance to a line segment within a sensor cone.

    :param cx: x coordinate of the car
    :param cy: y coordinate of the car
    :param t1: the start angle of the sensor
    :param t2: the end angle of the sensor
    :param r:  the range of the sensor
    :param p0x: first x coordinate of the line segment
    :param p0y: first y coordinate of the line segment
    :param p1x: second x coordinate of the line segment
    :param p1y: second y coordinate of the line segment
    :return: the minimum scaled distance to the segment if it intersects the sensor cone, 1 otherwise
    """

    return min(
        min(point(cx, cy, t1, t2, r, p0x, p0y), point(x, cy, t1, t2, r, p0x, p0y)),
        min(intersect(cx, cy, t1, r, p0x, p0y, p1x, p1y), intersect(cx, cy, t1, r, p0x, p0y, p1x, p1y))
    )