import pyglet as pg
from .sensor import SensorState
from .environment import Action


def visualize(env, task, manual=True):
    """
    Starts an graphical, interactive simulation of the given navigation environment.

    Uses the Pyglet game engine.
    """

    # Initialize environment
    env.set_task(task)
    env.reset()

    # Set the size in pixels of each drawn cell
    scale = 20

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

                if SensorState.CLEAR == env.map[x, y]:
                    clear.draw(pg.gl.GL_QUADS)
                elif SensorState.OCCUPIED == env.map[x, y]:
                    obstacle.draw(pg.gl.GL_QUADS)
                else:
                    if env.occupancy[x, y]:
                        unknown_obstacle.draw(pg.gl.GL_QUADS)
                    else:
                        unknown_clear.draw(pg.gl.GL_QUADS)

        # Draw goal
        pg.gl.glLoadIdentity()
        pg.gl.glScalef(scale, scale, 1)
        pg.gl.glTranslatef(env.goal_x, env.goal_y, 0)

        pg.graphics.draw(4, pg.gl.GL_QUADS,
                         ('v2f', (0, 0, 0, env.goal_height, env.goal_width, env.goal_height, env.goal_width, 0)),
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
    if manual:
        def on_key_press(symbol, modifier):
            if pg.window.key.UP == symbol:
                env.update(Action.UP)
            elif pg.window.key.DOWN == symbol:
                env.update(Action.DOWN)
            elif pg.window.key.LEFT == symbol:
                env.update(Action.LEFT)
            elif pg.window.key.RIGHT == symbol:
                env.update(Action.RIGHT)
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

        def update():
            env.update(env.expert())

        pg.clock.schedule_interval(update, 0.7)

    pg.app.run()
