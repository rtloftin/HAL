import pyglet as pg
import matplotlib.pyplot as pl

from .sensor import Occupancy
from .environment import Action


def visualize(env, sensor, task=None, expert=None):
    """
    Starts an graphical, interactive simulation of the given navigation environment.

    Uses the Pyglet game engine.
    """

    # Select the task if one is not provided
    if task is None:
        task = list(env.tasks)[0][0]

    # Initialize the expert if there is one
    if expert is not None:
        expert.task(task)

    # Initialize environment
    env.reset(task=task)

    # Initialize sensor
    sensor.update()

    # Set the size in pixels of each drawn cell
    scale = 15

    # Set up the Pyglet window
    window = pg.window.Window(env.width * scale, env.height * scale)

    # Define vertex lists
    clear = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (65, 105, 225, 65, 105, 225, 65, 105, 225, 65, 105, 225))
    )

    obstacle = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255))
    )

    unknown_clear = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (30, 50, 110, 30, 50, 110, 30, 50, 110, 30, 50, 110))
    )

    unknown_obstacle = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120))
    )

    agent = pg.graphics.vertex_list(
        4,
        ('v2i', (0, 0, 0, 1, 1, 1, 1, 0)),
        ('c3B', (125, 250, 0, 125, 250, 0, 125, 250, 0, 125, 250, 0))
    )

    # Define rendering loop
    def on_draw():
        window.clear()

        # Draw map
        for x in range(env.width):
            for y in range(env.height):
                pg.gl.glLoadIdentity()
                pg.gl.glScalef(scale, scale, 1)
                pg.gl.glTranslatef(x, y, 0)

                if Occupancy.CLEAR == sensor.map[x, y]:
                    clear.draw(pg.gl.GL_QUADS)
                elif Occupancy.OCCUPIED == sensor.map[x, y]:
                    obstacle.draw(pg.gl.GL_QUADS)
                else:
                    if env.occupied[x, y]:
                        unknown_obstacle.draw(pg.gl.GL_QUADS)
                    else:
                        unknown_clear.draw(pg.gl.GL_QUADS)

        # Draw goal
        goal = env.task.goal

        pg.gl.glLoadIdentity()
        pg.gl.glScalef(scale, scale, 1)
        pg.gl.glTranslatef(goal.x, goal.y, 0)

        pg.graphics.draw(4, pg.gl.GL_QUADS,
                         ('v2f', (0, 0, 0, goal.height, goal.width, goal.height, goal.width, 0)),
                         ('c3B', (255, 140, 0, 255, 140, 0, 255, 140, 0, 255, 140, 0)))

        # Draw agent
        pg.gl.glLoadIdentity()
        pg.gl.glScalef(scale, scale, 1)
        pg.gl.glTranslatef(env.x, env.y, 0)

        agent.draw(pg.gl.GL_QUADS)

    window.on_draw = on_draw

    # Define the screenshot method
    capture_index = 0

    def capture():
        nonlocal capture_index
        capture_index += 1

        pg.image.get_buffer_manager().get_color_buffer().save("navigation_" + str(capture_index) + ".png")

    # Decide whether we are doing manual or agent control
    if expert is None:
        def on_key_press(symbol, modifier):
            if pg.window.key.UP == symbol:
                env.update(Action.UP)
                sensor.update()
            elif pg.window.key.DOWN == symbol:
                env.update(Action.DOWN)
                sensor.update()
            elif pg.window.key.LEFT == symbol:
                env.update(Action.LEFT)
                sensor.update()
            elif pg.window.key.RIGHT == symbol:
                env.update(Action.RIGHT)
                sensor.update()
            elif pg.window.key.SPACE == symbol:
                env.reset()
            elif pg.window.key.ENTER == symbol:
                capture()

        window.on_key_press = on_key_press
    else:
        def on_key_press(symbol, modifier):
            if pg.window.key.SPACE == symbol:
                env.reset()
            elif pg.window.key.ENTER == symbol:
                capture()

        window.on_key_press = on_key_press

        def update(dt):
            env.update(expert.act(env.x, env.y))
            sensor.update()

        pg.clock.schedule_interval(update, 0.7)

    pg.app.run()


def render(values):
    fig, ax = pl.subplots()
    ax.pcolormesh(values)
    pl.show()
