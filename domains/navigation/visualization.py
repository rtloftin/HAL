import pyglet as pg


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
