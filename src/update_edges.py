'''
This file includes the function which updates the edge weights of the DFA in
number of ways. These different methods include:

  1. Scaling the edge weights due to obstacles according to a vector potential field
     which acts to "repel" an agent from an obstacle.
  2. Decreasing the edge weights approaching the goal region as to attract the
     agent to the goal postiion.
  3. Perturb the edge weights if deadlock is detected according to the type of
     deadlock detected. This follows schemes similar to those of traffic rules.
'''
