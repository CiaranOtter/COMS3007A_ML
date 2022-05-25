# import the need libraries including the local helpers.py
import helpers
import numpy as np;

# creating a new instance of Domain aka the world in which the agent exists
length = breadth = 4
seed = np.random.seed(np.random.randint(0,10000000));
world = helpers.Domain(length, breadth, seed);


def detAction():
    