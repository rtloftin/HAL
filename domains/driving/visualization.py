import pyglet as pg
import math


def visualize(env, task):
    """
    Starts an graphical, interactive simulation to allow
    for interactive testing of the driving domain.

    Uses the Pyglet game engine.

    :param env: the environment to render
    :param task: the specific task used to initialize the environment
    """

    # Initialize the environment
    env.set_Task(task)
    env.reset()

    # Simulation parameters
    control = {
        'acceleration': 0.0,
        'steering': 0.0
    }

    delta = 0.05

    # Set up the Pyglet window
    width = 600
    height = 600
    window = pg.window.Window(width, height)

    # Load the car sprites
    orange_car = pg.image.load('car_orange.png')
    white_car = pg.image.load('car_white.png')

    agent_sprite = pg.sprite.Sprite(orange_car, orange_car.width / -2, orange_car.height / -2)
    npc_sprite = pg.sprite.Sprite(white_car, white_car.width / -2, white_car.height / -2)

    car_scale = 1 / car_sprite.width

    # Define the map vertex batch
    background = pg.graphics.batch()
    background.add(4, pg.gl.GL_QUADS, None,
                   ('v2f', (0, 0, 0, 10, 10, 10, 10, 0)),
                   ('c3B', (65, 105, 225, 65, 105, 225, 65, 105, 225, 65, 105, 225)))

    for wall in env.walls:
        background.add(2, pg.gl.GL_LINES, None,
                       ('v2f', (wall.x0, wall.y0, wall.x1, wall.y1))
                       ('c3b', (255, 255, 255, 255, 255, 255)))

    # Define background vertices
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