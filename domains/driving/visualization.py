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
    env.set_task(task)
    env.reset()

    # Simulation parameters
    control = {
        'acceleration': 0.0,
        'steering': 0.0
    }

    delta = 0.05

    is_paused = False

    # Set up the Pyglet window
    width = 600
    height = 600
    window = pg.window.Window(width, height)

    # Load the car sprites
    orange_car = pg.image.load('domains/driving/car_orange.png')
    white_car = pg.image.load('domains/driving/car_white.png')

    agent_sprite = pg.sprite.Sprite(orange_car, orange_car.width / -2, orange_car.height / -2)
    npc_sprite = pg.sprite.Sprite(white_car, white_car.width / -2, white_car.height / -2)

    agent_scale = 1 / agent_sprite.width
    npc_scale = 1 / npc_sprite.width

    # Define the map vertex batch
    background = pg.graphics.Batch()

    background.add(4, pg.gl.GL_QUADS, None,
                   ('v2f', (0, 0, 0, env.height, env.width, env.height, env.width, 0)),
                   ('c3B', (65, 105, 225, 65, 105, 225, 65, 105, 225, 65, 105, 225)))


    for wall in env.walls:
        # print("Wall added - (", wall.x0, ",", wall.y0, "),(", wall.x1, ",", wall.y1, ")")
        background.add(2, pg.gl.GL_LINES, None,
                       ('v2f', (wall.x0, wall.y0, wall.x1, wall.y1)))

    # Define update loop
    def update(dt):
        nonlocal is_paused

        if not is_paused:
            env.update(control['acceleration'], control['steering'], dt)

            if env.complete:
                is_paused = True

    # Define rendering loop
    def on_draw():
        window.clear()

        pg.gl.glLineWidth(5)
        # pg.gl.glColor3f(1.0, 1.0, 1.0)

        pg.gl.glLoadIdentity()
        pg.gl.glScalef(width / env.width, height / env.height, 1)

        # Draw background
        background.draw()

        # Draw NPC cars
        for car in env.npc:
            pg.gl.glPushMatrix()
            pg.gl.glTranslatef(car.x, car.y, 0)
            pg.gl.glRotatef(90 + 180 * car.theta / math.pi, 0, 0, 1)
            pg.gl.glScalef(npc_scale, npc_scale, 1)
            npc_sprite.draw()
            pg.gl.glPopMatrix()

        # Draw agent
        pg.gl.glPushMatrix()
        pg.gl.glTranslatef(env.x, env.y, 0)
        pg.gl.glRotatef(90 + 180 * env.direction / math.pi, 0, 0, 1)
        pg.gl.glScalef(agent_scale, agent_scale, 1)
        agent_sprite.draw()
        pg.gl.glPopMatrix()

    window.on_draw = on_draw

    # Define key handler
    def on_key_press(symbol, modifier):
        nonlocal is_paused

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
        elif pg.window.key.ENTER == symbol:
            env.reset()
            is_paused = False

    window.on_key_press = on_key_press

    # Start simulation
    pg.clock.schedule_interval(update, delta)

    # Start interface
    pg.app.run()
