'''
.. module:: geometric_funcs.py
   :synopsis: Functions to perform geometric checks and operations.

.. moduleauthor:: Ryan Peterson <pete9936@umn.edu.edu>

'''

import logging, sys
import copy, math, pdb
import operator
import networkx as nx


def get_discretization(ts):
    ''' Finds the discretization in the xy plane and the z discretization
    to be used by diagonal feasibility checks later. '''
    node_set = nx.get_node_attributes(ts.g,"position")
    for key, vals in node_set.items():
        check_3d = vals
        break
    try:
        check_3d[2]
        flag_3d = True
    except IndexError:
        flag_3d = False
    # find the x-y discretization
    iter = 0
    for key, vals in node_set.items():
        if iter == 0:
            set1 = (vals[0],vals[1])
        elif iter == 1:
            set2 = (vals[0],vals[1])
            break
        iter = iter + 1
    if abs(set1[0] - set2[0]) > 0.0:
        disc = abs(set1[0] - set2[0])
    else:
        disc = abs(set1[1] - set2[1])
    # find the z discretization
    if flag_3d == True:
        iter = 0
        disc_z = 1000 # sufficiently high number
        for key, vals in node_set.items():
            if iter == 0:
                z_init = vals[2]
                iter = iter + 1
            else:
                z_change = abs(z_init - vals[2])
                if z_change > 0.0:
                    if z_change < disc_z:
                        disc_z = z_change
    else:
        disc_z = 0
    return disc, disc_z

def check_cross(cur_ind, ts, ts_prev_states, ts_next_states, priority, key_list, disc, disc_z):
    ''' Check if moving in diagonal directions causes any conflict. Look into only performing
    this if local neighborhood is true, and only with the agents in the local neighborhood.
    Also consider adding additional weight to the node directly below (downwash).'''
    weighted_nodes = []
    node_set = nx.get_node_attributes(ts.g,"position")
    cur_prev_pose = node_set[ts_prev_states[cur_ind]]
    cur_next_pose = node_set[ts_next_states[cur_ind]]
    x_cur_prev = cur_prev_pose[0]
    x_cur_next = cur_next_pose[0]
    x_cur_dist = cur_prev_pose[0] - cur_next_pose[0]
    y_cur_prev = cur_prev_pose[1]
    y_cur_next = cur_next_pose[1]
    y_cur_dist = cur_prev_pose[1] - cur_next_pose[1]
    if disc_z > 0:
        z_cur_prev = cur_prev_pose[2]
        z_cur_next = cur_next_pose[2]
        z_cur_dist = cur_prev_pose[2] - cur_next_pose[2]
    # run through higher priority set
    for p_ind, p_val in enumerate(priority):
        for k, key in enumerate(key_list):
            if p_val == key:
                comp_prev_pose = node_set[ts_prev_states[k]]
                comp_next_pose = node_set[ts_next_states[k]]
                x_comp_prev = comp_prev_pose[0]
                x_comp_next = comp_next_pose[0]
                x_comp_dist = comp_prev_pose[0] - comp_next_pose[0]
                y_comp_prev = comp_prev_pose[1]
                y_comp_next = comp_next_pose[1]
                y_comp_dist = comp_prev_pose[1] - comp_next_pose[1]
                if disc_z > 0:
                    z_comp_prev = comp_prev_pose[2]
                    z_comp_next = comp_next_pose[2]
                    z_comp_dist = comp_prev_pose[2] - comp_next_pose[2]
                # Now perform the checks
                if disc_z == 0:
                    if x_cur_dist == disc and y_cur_dist == disc and ((x_comp_dist == -1*disc and y_comp_dist == disc \
                                   and x_cur_prev == x_comp_next and y_cur_prev == y_comp_prev) or (x_comp_dist == disc \
                                   and y_comp_dist == -1*disc and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == -1*disc and ((x_comp_dist == -1*disc and y_comp_dist == disc \
                                   and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next) or (x_comp_dist == disc \
                                   and y_comp_dist == -1*disc and y_cur_prev == y_comp_prev and x_cur_prev == x_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == disc and ((x_comp_dist == disc and y_comp_dist == disc \
                                   and x_cur_prev == x_comp_next and y_cur_prev == y_comp_prev) or (x_comp_dist == -1*disc \
                                   and y_comp_dist == -1*disc and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == disc and y_cur_dist == -1*disc and ((x_comp_dist == disc and y_comp_dist == disc \
                                   and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next) or (x_comp_dist == -1*disc \
                                   and y_comp_dist == -1*disc and x_cur_prev == x_comp_next and y_cur_prev == y_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                else:
                    # add downwash check here ***
                    if x_cur_dist == disc and y_cur_dist == disc and ((x_comp_dist == -1*disc and y_comp_dist == disc \
                        and x_cur_prev == x_comp_next and y_cur_prev == y_comp_prev) or (x_comp_dist == disc and y_comp_dist == -1*disc \
                        and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next)) and ((z_cur_dist == z_comp_dist and z_cur_prev == z_comp_prev) \
                        or (z_cur_dist == disc_z and z_comp_dist == -1*disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next) \
                        or (z_cur_dist == -1*disc_z and z_comp_dist == disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == -1*disc and ((x_comp_dist == -1*disc and y_comp_dist == disc \
                        and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next) or (x_comp_dist == disc and y_comp_dist == -1*disc \
                        and y_cur_prev == y_comp_prev and x_cur_prev == x_comp_next)) and ((z_cur_dist == z_comp_dist and z_cur_prev == z_comp_prev) \
                        or (z_cur_dist == disc_z and z_comp_dist == -1*disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next) \
                        or (z_cur_dist == -1*disc_z and z_comp_dist == disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == disc and ((x_comp_dist == disc and y_comp_dist == disc \
                        and x_cur_prev == x_comp_next and y_cur_prev == y_comp_prev) or (x_comp_dist == -1*disc and y_comp_dist == -1*disc \
                        and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next)) and ((z_cur_dist == z_comp_dist and z_cur_prev == z_comp_prev) \
                        or (z_cur_dist == disc_z and z_comp_dist == -1*disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next) \
                        or (z_cur_dist == -1*disc_z and z_comp_dist == disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == disc and y_cur_dist == -1*disc and ((x_comp_dist == disc and y_comp_dist == disc \
                        and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next) or (x_comp_dist == -1*disc and y_comp_dist == -1*disc \
                        and x_cur_prev == x_comp_next and y_cur_prev == y_comp_next)) and ((z_cur_dist == z_comp_dist and z_cur_prev == z_comp_prev) \
                        or (z_cur_dist == disc_z and z_comp_dist == -1*disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next) \
                        or (z_cur_dist == -1*disc_z and z_comp_dist == disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    # in plane movement (xz plane)
                    elif x_cur_dist == disc and y_cur_dist == 0 and z_cur_dist == disc_z and y_comp_dist == 0 and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == 0 and z_cur_dist == -1*disc_z and y_comp_dist == 0 and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_node = ts_next_states[k]
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == 0 and z_cur_dist == disc_z and y_comp_dist == 0 and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_node = ts_next_states[k]
                        break
                    elif x_cur_dist == disc and y_cur_dist == 0 and z_cur_dist == -1*disc_z and y_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_prev)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    # in plane movement (yz plane)
                    elif y_cur_dist == disc and x_cur_dist == 0 and z_cur_dist == disc_z and x_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif y_cur_dist == -1*disc and x_cur_dist == 0 and z_cur_dist == -1*disc_z and x_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif y_cur_dist == -1*disc and x_cur_dist == 0 and z_cur_dist == disc_z and x_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif y_cur_dist == disc and x_cur_dist == 0 and z_cur_dist == -1*disc_z and x_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    # in plane movement (diagonal plane 1)
                    elif x_cur_dist == disc and y_cur_dist == disc and z_cur_dist == disc_z and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == -1*disc and z_cur_dist == -1*disc_z and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == -1*disc and z_cur_dist == disc_z and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == disc and y_cur_dist == disc and z_cur_dist == -1*disc_z and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    # in plane movement (diagonal plane 2)
                    elif x_cur_dist == disc and y_cur_dist == -1*disc and z_cur_dist == disc_z and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == disc and z_cur_dist == -1*disc_z and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == -1*disc and y_cur_dist == disc and z_cur_dist == disc_z and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next)):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif x_cur_dist == disc and y_cur_dist == -1*disc and z_cur_dist == -1*disc_z and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev)):
                        weighted_nodes.append(ts_next_states[k])
                        break
    return weighted_nodes


if __name__ == '__main__':
    pass
