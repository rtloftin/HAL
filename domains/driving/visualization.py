import pyglet as pg
import math


def interactive_test():
    """
    Starts an graphical, interactive simulation to allow
    for interactive testing of the driving domain.

    Uses the Pyglet game engine.
    """

    # Construct domain
    env = Environment()

    control = {
        'acceleration': 0.0,
        'steering': 0.0
    }

    delta = 0.05

    # Set up the Pyglet window
    width = 600
    height = 600
    window = pg.window.Window(width, height)

    # Load the car sprite
    car_image = pg.image.load('car_orange.png')
    car_sprite = pg.sprite.Sprite(car_image, car_image.width / -2, car_image.height / -2)
    car_scale = 1 / car_sprite.width

    # Define vertex lists
    car = pg.graphics.vertex_list(
        4,
        ('v2f', (-.5, -.5, -.5, .5, .5, .5, .5, -.5)),
        ('c3B', (125, 250, 0, 125, 250, 0, 125, 250, 0, 125, 250, 0))
    )

    background = pg.graphics.vertex_list(
        4,
        ('v2f', (0, 0, 0, 10, 10, 10, 10, 0)),
        ('c3B', (65, 105, 225, 65, 105, 225, 65, 105, 225, 65, 105, 225))
    )

    # Define update loop
    def update(dt):
        env.update(control['acceleration'], control['steering'], dt)

    # Define rendering loop
    def on_draw():
        window.clear()

        pg.gl.glLoadIdentity()
        pg.gl.glScalef(60, 60, 1)

        # Draw background
        background.draw(pg.gl.GL_QUADS)

        # Draw agent
        pg.gl.glPushMatrix()
        pg.gl.glTranslatef(env.x, env.y, 0)
        pg.gl.glRotatef(90 + 180 * env.theta / math.pi, 0, 0, 1)

        pg.gl.glScalef(car_scale, car_scale, 1)
        car_sprite.draw()
        # car.draw(pg.gl.GL_QUADS)

        pg.gl.glPopMatrix()

    window.on_draw = on_draw

    # Define key handler
    def on_key_press(symbol, modifier):
        if pg.window.key.UP == symbol:
            if control['acceleration'] < 0.1:
                control['acceleration'] += 0.05
        elif pg.window.key.DOWN == symbol:
            if control['acceleration'] > -0.1:
                control['acceleration'] -= 0.05
        elif pg.window.key.LEFT == symbol:
            if control['steering'] > -0.4:
                control['steering'] -= 0.2
        elif pg.window.key.RIGHT == symbol:
            if control['steering'] < 0.4:
                control['steering'] += 0.2

    window.on_key_press = on_key_press

    # Start simulation
    pg.clock.schedule_interval(update, delta)

    # Start interface
    pg.app.run()