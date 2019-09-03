# This file includes the function which updates the edge weights of the DFA in
# number of ways. These different methods include:
#
#  1. Scaling the edge weights due to obstacles according to a vector potential field
#     which acts to "repel" an agent from an obstacle.
#  2. Decreasing the edge weights approaching the goal region as to attract the
#     agent to the goal postiion.
#  3. Perturb the edge weights if deadlock is detected according to the type of
#     deadlock detected. This follows schemes similar to those of traffic rules.


import logging
import copy






def update_edge_weights(edge_list):
    '''Updates edge weights in order to find a new control policy.'''

    # copy attributes
    edge_list_new = copy.deepcopy(edge_list)

    # edge_list_new.weight = edge_list.weight
    # edge_list_new.label = edge_list.label

    for edge, label in edge_list:
        if label == 'update':
            new_weight = potential_field(edge)
            edge = new_weight
    return edge_list


def potential_field(edge):
    '''Update edge weight using a potential field function.'''
    pass
    return weight
