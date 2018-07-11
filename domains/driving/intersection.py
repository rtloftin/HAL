"""
Defines the 'intersection' environment, along with the associated tasks.

This environment is a simple four-way intersection, with the agent's car
coming from one direction, and random NPC cars coming from the other directions.

Tasks include turning left and right, and going straight.  Each road has two lanes,
and reward is given for reaching the desired lane without hitting anything.
"""