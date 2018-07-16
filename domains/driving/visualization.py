import pyglet as pg
import math


def visualize(env, task, manual=True):
    """
    Starts an graphical, interactive simulation of the given driving environment.

    Uses the Pyglet game engine.  Can be run in either manual or expert mode. In
    manual model, the car is controlled using the keyboard, while in expert mode,
    the car is controlled by the predefined expert policy for the given task.

    :param env: the environment to render
    :param task: the specific task used to initialize the environment
    :param manual: whether to allow manual control, or simulate according to the expert's policy
    """

    # Initialize the environment
    env.set_task(task)
    env.reset()

    # Control parameters
    acceleration = 0.0
    steering = 0.0

    # Simulation parameters
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

    # Define background rectangle
    background = pg.graphics.Batch()
    background.add(4, pg.gl.GL_QUADS, None,
                   ('v2f', (0, 0, 0, env.height, env.width, env.height, env.width, 0)),
                   ('c3B', (65, 105, 225, 65, 105, 225, 65, 105, 225, 65, 105, 225)))

    # Define walls
    map = pg.graphics.Batch()

    for wall in env.walls:
        # print("Wall added - (", wall.x0, ",", wall.y0, "),(", wall.x1, ",", wall.y1, ")")
        map.add(2, pg.gl.GL_LINES, None, ('v2f', (wall.x0, wall.y0, wall.x1, wall.y1)))

    pg.gl.glLineWidth(5)
    pg.gl.glColor3f(1.0, 1.0, 1.0)

    # Define update loop
    def update(dt):
        nonlocal is_paused

        if not is_paused:
            if manual:
                env.update(acceleration, steering, dt)
            else:
                acc, steer = env.expert()
                env.update(acc, steer)

            if env.complete:
                is_paused = True

    # Define rendering loop
    def on_draw():
        window.clear()

        pg.gl.glLoadIdentity()
        pg.gl.glScalef(width / env.width, height / env.height, 1)

        # Draw background
        background.draw()

        # Draw map
        map.draw()

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
        nonlocal is_paused, steering, acceleration

        if pg.window.key.UP == symbol:
            if acceleration < 0.1:
                acceleration += 0.05
        elif pg.window.key.DOWN == symbol:
            if acceleration > -0.1:
                acceleration -= 0.05
        elif pg.window.key.LEFT == symbol:
            if steering > -0.2:
                steering -= 0.05
        elif pg.window.key.RIGHT == symbol:
            if steering < 0.2:
                steering += 0.05
        elif pg.window.key.SPACE == symbol:
            env.reset()
            acceleration = 0.0
            steering = 0.0
            is_paused = False

    window.on_key_press = on_key_press

    # Start simulation
    pg.clock.schedule_interval(update, delta)

    # Start interface
    pg.app.run()
