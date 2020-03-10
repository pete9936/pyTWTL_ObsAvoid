'''
.. module:: geometric_funcs.py
   :synopsis: Functions to perform geometric checks and operations.

.. moduleauthor:: Ryan Peterson <pete9936@umn.edu.edu>

'''

import logging, sys
import copy, math, pdb
import operator
import networkx as nx
import numpy as np


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
        disc_z = 10000 # sufficiently high number
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

def downwash_check(cur_ind, ts, ts_next_states, priority, key_list, radius):
    ''' Find nodes in transition directly above or below agent of higher priority in order
    to avoid downwash instability for quadrotors. '''
    downwash_nodes = []
    downwash_zone = 7*radius  # (3.5 diameters should be sufficient)
    node_set = nx.get_node_attributes(ts.g,"position")
    cur_next_pose = node_set[ts_next_states[cur_ind]]
    x_cur_next = cur_next_pose[0]
    y_cur_next = cur_next_pose[1]
    # determine if grid is defined in 2D or 3D
    try:
        cur_next_pose[2]
        flag_3d = True
    except IndexError:
        flag_3d = False
    # run through higher priority set
    if flag_3d == True:
        for p_ind, p_val in enumerate(priority):
            for k, key in enumerate(key_list):
                if p_val == key:
                    comp_next_pose = node_set[ts_next_states[k]]
                    x_comp_next = comp_next_pose[0]
                    y_comp_next = comp_next_pose[1]
                    z_comp_next = comp_next_pose[2]
                    z_cur_next = cur_next_pose[2]
                    a0 = np.array([x_cur_next, y_cur_next])
                    b0 = np.array([x_comp_next, y_comp_next])
                    dist = np.linalg.norm(a0-b0)
                    if dist < 2*radius:
                        if abs(z_cur_next-z_comp_next) < downwash_zone:
                            downwash_nodes.append(ts_next_states[k])
                    break
    return downwash_nodes

def check_intersect(cur_ind, ts, ts_prev_states, ts_next_states, priority, key_list, radius, seg_time):
    ''' Check if moving in diagonal directions causes any conflict. Look into only performing
    this if local neighborhood is true, and only with the agents in the local neighborhood.
    This method uses intersection of lines to test which should be more robust in environments
    where discretization is non-uniform. '''
    # first give some discretization of segments based on given agent dynamics
    num_steps = 10
    weighted_nodes = []
    node_set = nx.get_node_attributes(ts.g,"position")
    cur_prev_pose = node_set[ts_prev_states[cur_ind]]
    cur_next_pose = node_set[ts_next_states[cur_ind]]
    x_cur_prev = cur_prev_pose[0]
    x_cur_next = cur_next_pose[0]
    y_cur_prev = cur_prev_pose[1]
    y_cur_next = cur_next_pose[1]
    # determine if grid is defined in 2D or 3D
    try:
        cur_prev_pose[2]
        flag_3d = True
    except IndexError:
        flag_3d = False
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
                # Now perform the checks
                if flag_3d == True:
                    z_cur_prev = cur_prev_pose[2]
                    z_cur_next = cur_next_pose[2]
                    z_comp_prev = comp_prev_pose[2]
                    z_comp_next = comp_next_pose[2]
                    a0 = np.array([x_cur_prev, y_cur_prev, z_cur_prev])
                    a1 = np.array([x_cur_next, y_cur_next, z_cur_next])
                    b0 = np.array([x_comp_prev, y_comp_prev, z_comp_prev])
                    b1 = np.array([x_comp_next, y_comp_next, z_comp_next])
                else:
                    a0 = np.array([x_cur_prev, y_cur_prev])
                    a1 = np.array([x_cur_next, y_cur_next])
                    b0 = np.array([x_comp_prev, y_comp_prev])
                    b1 = np.array([x_comp_next, y_comp_next])
                if flag_3d == False:
                    Xa, Ya, append = check_intersect_2d(x_cur_prev, y_cur_prev, x_cur_next, y_cur_next, \
                                                        x_comp_prev, y_comp_prev, x_comp_next, y_comp_next)
                    if append == True:
                        A = a1 - a0
                        B = b1 - b0
                        p_int = np.array([Xa, Ya])
                        pA_vec = a1 - p_int
                        pB_vec = b1 - p_int
                        pA_vec_percent = np.linalg.norm(pA_vec)/np.linalg.norm(A)
                        pB_vec_percent = np.linalg.norm(pB_vec)/np.linalg.norm(B)
                        if abs(pA_vec_percent - pB_vec_percent) < 1.0/num_steps:
                            weighted_nodes.append(ts_next_states[k])
                    break
                else:
                    pA, pB, distance = closestDistanceBetweenLines(a0,a1,b0,b1)
                    print 'Distance between both segments:', distance
                    if distance == None:
                        break
                    elif pA is None:
                        if distance <= radius:
                            weighted_nodes.append(ts_next_states[k])
                        break
                    else:
                        A = a1 - a0
                        B = b1 - b0
                        pA_vec = a1 - pA
                        pB_vec = b1 - pB
                        pA_vec_percent = np.linalg.norm(pA_vec)/np.linalg.norm(A)
                        pB_vec_percent = np.linalg.norm(pB_vec)/np.linalg.norm(B)
                        if abs(pA_vec_percent - pB_vec_percent) < 1.0/num_steps:
                            if distance <= radius:
                                weighted_nodes.append(ts_next_states[k])
                        break
                        # if np.linalg.norm(a0 - b1) < 0.01 or np.linalg.norm(a1 - b0) < 0.01: # improve later ***
                        #    distance = None
                        # else:
                        # if distance == None:
                        #    break
                        # elif distance <= radius:
                        #    weighted_nodes.append(ts_next_states[k])
                        # break
    return weighted_nodes

def closestDistanceBetweenLines(a0,a1,b0,b1):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each finite vector and their distance '''
    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    if magA > 0 and magB > 0:
        _A = A / magA
        _B = B / magB
    else:
        return None, None, None
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        d1 = np.dot(_A,(b1-a0))
        # dist = min(d0,d1)
        # Segments overlap, return distance between parallel segments
        return None,None, np.linalg.norm(((d0*_A)+a0)-b0)

    # Lines cross at some point (skew): Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])
    t0 = detA/denom
    t1 = detB/denom
    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B
    # Projections onto finite vectors A and B
    if t0 < 0:
        pA = a0
    elif t0 > magA:
        pA = a1
    if t1 < 0:
        pB = b0
    elif t1 > magB:
        pB = b1
    # Projection on A
    if t0 < 0 or t0 > magA:
        dot = np.dot(_B,(pA-b0))
        if dot < 0:
            dot = 0
        elif dot > magB:
            dot = magB
        pB = b0 + (_B * dot)
    # Projection on B
    if t1 < 0 or t1 > magB:
        dot = np.dot(_A,(pB-a0))
        if dot < 0:
            dot = 0
        elif dot > magA:
            dot = magA
        pA = a0 + (_A * dot)

    return pA,pB,np.linalg.norm(pA-pB)


def check_intersect_2d(x_cur_prev, y_cur_prev, x_cur_next, y_cur_next, \
                        x_comp_prev, y_comp_prev, x_comp_next, y_comp_next):
    ''' Check if two lines in 2D space intersect within current bounds '''
    x_cur_dist = x_cur_prev - x_cur_next
    y_cur_dist = y_cur_prev - y_cur_next
    x_comp_dist = x_comp_prev - x_comp_next
    y_comp_dist = y_comp_prev - y_comp_next
    I_x1 = [min(x_cur_prev, x_cur_next), max(x_cur_prev, x_cur_next)]
    I_x2 = [min(x_comp_prev, x_comp_next), max(x_comp_prev, x_comp_next)]
    I_y1 = [min(y_cur_prev, y_cur_next), max(y_cur_prev, y_cur_next)]
    I_y2 = [min(y_comp_prev, y_comp_next), max(y_comp_prev, y_comp_next)]
    if ((I_x1[1] <= I_x2[0] or I_x1[0] >= I_x2[1] or I_y1[1] <= I_y2[0] or I_y1[0] >= I_y2[1])):
        return None, None, False
    A1 = 0
    A2 = 0
    if x_cur_dist != 0:
        A1 = y_cur_dist/x_cur_dist
        b1 = y_cur_prev - A1*x_cur_prev
    if x_comp_dist != 0:
        A2 = y_comp_dist/x_comp_dist
        b2 = y_comp_prev - A2*x_comp_prev
    if A1 == 0 or A2 == 0 or A1 == A2:
        # checks if lines are parallel, or at least have a slope in our case
        return None, None, False
    if A1-A2 != 0:
        # A1*Xa + b1 = A2*Xa + b2
        Xa = (b2-b1)/(A1-A2)
        Ya = A1*Xa + b1
        if ((Xa <= max(I_x1[0],I_x2[0])) or (Xa >= min(I_x1[1],I_x2[1])) \
            or (Ya <= max(I_y1[0],I_y2[0])) or (Ya >= min(I_y1[1],I_y2[1]))):
            return None, None, False
        else:
            return Xa, Ya, True
    # default return
    return None, None, False


if __name__ == '__main__':
    pass
