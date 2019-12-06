'''
.. module:: example_neighbors_RP8.py
   :synopsis: Case studies for TWTL package and lab testing.

.. moduleauthor:: Ryan Peterson <pete9936@umn.edu.edu>

'''

import logging, sys
import StringIO
import pdb, os, copy, math
import time, timeit
import operator
import csv

import networkx as nx
import matplotlib.pyplot as plt

import twtl
import write_files

from dfa import DFAType
from synthesis import expand_duration_ts, compute_control_policy, ts_times_fsa,\
                      verify, compute_control_policy2, compute_energy
from geometric_funcs import get_discretization, check_cross, check_intersect
from write_files import write_to_land_file, write_to_csv_iter, write_to_csv,\
                        write_to_iter_file, write_to_control_policy_file
from learning import learn_deadlines
from lomap import Ts


def case1_synthesis(formulas, ts_files, radius, time_wp, lab_testing):
    startFull = timeit.default_timer()
    startOff = timeit.default_timer()
    dfa_dict = {}
    for ind, f in enumerate(formulas):
        _, dfa_inf, bdd = twtl.translate(f, kind=DFAType.Infinity, norm=True)

        logging.debug('\nEnd of translate\n\n')
        logging.info('The bound of formula "%s" is (%d, %d)!', f, *bdd)
        logging.info('Translated formula "%s" to infinity DFA of size (%d, %d)!',
                     f, *dfa_inf.size())
        dfa_dict[ind+1] = copy.deepcopy(dfa_inf) # Note that the key is set to the agent number

    logging.debug('\n\nStart policy computation\n')

    ts_dict = {}
    ets_dict = {}
    for ind, ts_f in enumerate(ts_files):
        ts_dict[ind+1] = Ts(directed=True, multi=False)
        ts_dict[ind+1].read_from_file(ts_f)
        ets_dict[ind+1] = expand_duration_ts(ts_dict[ind+1])
    for ind in ts_dict:
        print 'Size of TS:', ets_dict[ind].size()
    # Get the nominal PA for each agent
    pa_nom_dict = {}
    for key in dfa_dict:
        logging.info('Constructing product automaton with infinity DFA!')
        pa = ts_times_fsa(ets_dict[key], dfa_dict[key])
        # Give initial weight attribute to all edges in pa
        nx.set_edge_attributes(pa.g,"weight",1)
        logging.info('Product automaton size is: (%d, %d)', *pa.size())
        # Make a copy of the nominal PA to change
        pa_nom_dict[key] = copy.deepcopy(pa)

    for key in pa_nom_dict:
        print 'Size of PA:', pa_nom_dict[key].size()

    # Compute optimal path in Pa_Prime and project onto the TS, initial policy
    ts_policy_dict_nom = {}
    pa_policy_dict_nom = {}
    output_dict_nom = {}
    tau_dict_nom = {}
    for key in pa_nom_dict:
        ts_policy_dict_nom[key], pa_policy_dict_nom[key], output_dict_nom[key], tau_dict_nom[key] = \
                    compute_control_policy(pa_nom_dict[key], dfa_dict[key], dfa_dict[key].kind)
    # Perform initial check on nominal control policies
    for key in ts_policy_dict_nom:
        if ts_policy_dict_nom[key] is None:
            logging.info('No control policy found!')

    # Compute the energy for each agent's PA at every node to use in offline instance
    startEnergy = timeit.default_timer()
    energy_dict = {}
    for key in ts_policy_dict_nom:
        compute_energy(pa_nom_dict[key], dfa_dict[key])
    stopEnergy = timeit.default_timer()
    print 'Run Time (s) to get the energy function for all three PA: ', stopEnergy - startEnergy

    # set empty control policies that will be iteratively updated
    ts_control_policy_dict = {}
    pa_control_policy_dict = {}
    # Set a Boolean vector to indicate if a path needs to be recomputed
    policy_flag = [1]*len(ts_files)

    # Initialize policy variables
    for key in ts_policy_dict_nom:
        ts_control_policy_dict[key] = []
        pa_control_policy_dict[key] = []

    # Concatenate nominal policies for searching
    policy_match, key_list, policy_match_index = update_policy_match(ts_policy_dict_nom)

    # get the discretization(s) from the transition system
    disc, disc_z = get_discretization(ets_dict[key])

    # Initialize vars, give nominal policies
    iter_step = 0
    append_flag = True
    final_flag = False
    final_count = 0
    running = True
    traj_length = 0
    ts_policy = copy.deepcopy(ts_policy_dict_nom)
    pa_policy = copy.deepcopy(pa_policy_dict_nom)
    tau_dict = tau_dict_nom
    local_flag = {}
    compute_local = False
    switch_flag = False
    for key in ts_policy:
        local_flag[key] = False
    num_hops = 1 # parameter for trade study
    # Get agent priority based on lowest energy
    priority = get_priority(pa_nom_dict, pa_policy_dict_nom, key_list)

    # Print time statistics
    stopOff = timeit.default_timer()
    print 'Offline run time for all initial setup: ', stopOff - startOff
    startOnline = timeit.default_timer()

    # Execute takeoff caommand for all crazyflies in lab testing
    if lab_testing:
        startTakeoff = timeit.default_timer()
        os.chdir("/home/ryan/crazyswarm/ros_ws/src/crazyswarm/scripts")
        os.system("/home/ryan/crazyswarm/ros_ws/src/crazyswarm/scripts/twtl_takeoff.py") # make sure file is an executable
        os.chdir("/home/ryan/Desktop/pyTWTL/src")
        stopTakeoff = timeit.default_timer()
        print 'Takeoff time, should be ~2.7sec: ', stopTakeoff - startTakeoff

    # Iterate through all policies sequentially
    while running:
        while policy_match:
            for p_ind, p_val in enumerate(priority):
                if p_ind < 1:
                    append_flag = True
                else:
                    # Get local neighborhood (two-hop) of nodes to search for a conflict
                    for k, key in enumerate(key_list):
                        if p_val == key:
                            node = policy_match[0][k]
                            break
                    local_set = get_neighborhood(node, ts_dict[p_val])
                    prev_nodes = []
                    for pty in priority[0:p_ind]:
                        for k, key in enumerate(key_list):
                            if pty == key:
                                prev_node = policy_match[0][k]
                                if prev_node in local_set:
                                    prev_nodes.append(prev_node)
                                break
                    # Get extra nodes from sharing extended trajectory
                    soft_nodes = {}
                    for pty in priority[0:p_ind]:
                        for k, key in enumerate(key_list):
                            if pty == key:
                                ts_length = len(ts_policy[key])
                                if ts_length > num_hops:
                                    for i in range(num_hops):
                                        if ts_policy[key][i+1] in local_set:
                                            try:
                                                soft_nodes[i+1] = [soft_nodes[i+1], ts_policy[key][i+1]]
                                            except KeyError:
                                                soft_nodes[i+1] = ts_policy[key][i+1]
                                else:
                                    for i in range(ts_length-1):
                                        if ts_policy[key][i+1] in local_set:
                                            try:
                                                soft_nodes[i+1] = [soft_nodes[i+1], ts_policy[key][i+1]]
                                            except KeyError:
                                                soft_nodes[i+1] = ts_policy[key][i+1]
                    # check if local flags should be set or if one is switching off
                    if prev_nodes:
                        local_flag[p_val] = True
                    elif local_flag[p_val] == True and len(ts_policy[p_val])==1:
                        local_flag[p_val] = False
                        switch_flag = True
                        policy_flag[p_val-1] = 0
                        break
                    else:
                        local_flag[p_val] = False
                    if node in prev_nodes:
                        policy_flag[p_val-1] = 0
                        append_flag = False
                        compute_local = True
                        break
                    else:
                        policy_flag[p_val-1] = 1
                        append_flag = True

            if append_flag == False:
                weighted_nodes = prev_nodes
                weighted_soft_nodes = soft_nodes
            else:
                weighted_nodes = []
                weighted_soft_nodes = []
            # Update weights if transitioning between same two nodes
            ts_prev_states = []
            ts_index = []
            if len(policy_match[0]) > 1 and traj_length >= 1:
                for key in ts_control_policy_dict:
                    if len(ts_control_policy_dict[key]) == traj_length:
                        ts_prev_states.append(ts_control_policy_dict[key][-1])
            if ts_prev_states:
                for p_ind2, p_val2 in enumerate(priority):
                    if p_ind2 > 0:
                        for k_c, key in enumerate(key_list):
                            if p_val2 == key:
                                node = policy_match[0][k_c]
                                break
                        # Check if the trajectories will cross each other in transition
                        # cross_weight = check_cross(k_c, ets_dict[key], ts_prev_states, policy_match[0], \
                        #                                    priority[0:p_ind2], key_list, disc, disc_z)
                        cross_weight = check_intersect(k_c, ets_dict[key], ts_prev_states, policy_match[0], \
                                                            priority[0:p_ind2], key_list, radius)
                        if cross_weight:
                            weighted_nodes = prev_nodes
                            weighted_soft_nodes = soft_nodes
                            for cross_node in cross_weight:
                                if cross_node not in weighted_nodes:
                                    weighted_nodes.append(cross_node)
                            policy_flag[p_val2-1] = 0
                            compute_local = True
                            append_flag = False
                            # Check if agents using same transition
                            for p_ind3, p_val3 in enumerate(priority[0:p_ind2]):
                                for k, key in enumerate(key_list):
                                    if p_val3 == key:
                                        if ts_prev_states[k] == node:
                                            if policy_match[0][k] == ts_prev_states[k_c]:
                                                temp_node = policy_match[0][k]
                                                if temp_node not in weighted_nodes:
                                                    weighted_nodes.append(temp_node)
                                                if node not in weighted_nodes:
                                                    weighted_nodes.append(node)
                                                break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            # Check if agents using same transition
                            for p_ind3, p_val3 in enumerate(priority[0:p_ind2]):
                                for k, key in enumerate(key_list):
                                    if p_val3 == key:
                                        if ts_prev_states[k] == node:
                                            if policy_match[0][k] == ts_prev_states[k_c]:
                                                weighted_nodes = prev_nodes
                                                weighted_soft_nodes = soft_nodes
                                                temp_node = policy_match[0][k]
                                                if temp_node not in weighted_nodes:
                                                    weighted_nodes.append(temp_node)
                                                if node not in weighted_nodes:
                                                    weighted_nodes.append(node)
                                                policy_flag[p_val2-1] = 0
                                                compute_local = True
                                                append_flag = False
                                                break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
            else:
                append_flag = True

            # Account for final_state exception issues
            if len(policy_match) == 1 and final_flag == True:
                weighted_nodes = []
                append_flag = False
            if final_flag == True:
                final_count += 1
                policy_flag[p_val-1] = 0
            if final_count > 2:
                final_count = 0
            # Account for receding horizon trajectories
            if switch_flag:
                compute_local = False
                append_flag = False
                switch_flag = False
            elif len(policy_match) == 1:
                for pty in priority:
                    if local_flag[pty] == True and append_flag == True: # update local trajectory and move on
                        init_loc = pa_control_policy_dict[pty][-1]
                        ts_temp = ts_policy[pty]
                        pa_temp = pa_policy[pty]
                        ts_policy[pty], pa_policy[pty], ignore = extend_horizon(pa_nom_dict[pty], pa_policy[pty][0])
                        if ignore:
                            # This accounts for termination criteria
                            ts_policy[pty] = ts_temp
                            pa_policy[pty] = pa_temp
                        else:
                            policy_match, key_list, policy_match_index = update_policy_match(ts_policy)
                    elif local_flag[pty] == True and append_flag == False: # update local trajectory later
                        compute_local = True
                        break
                    elif local_flag[pty] == False and append_flag == False: # update trajectory w/ Dijkstra's later
                        compute_local = False
                    else:
                        continue
            # Append trajectories
            if append_flag and final_count <= 1:
                for key in ts_policy:
                    ts_control_policy_dict[key].append(ts_policy[key].pop(0))
                    pa_policy_temp = list(pa_policy[key])
                    pa_control_policy_dict[key].append(pa_policy_temp.pop(0))
                    pa_policy[key] = tuple(pa_policy_temp)
                ts_write = policy_match.pop(0)
                traj_length += 1
                # publish this waypoint to a csv file
                write_to_csv_iter(ts_dict, ts_write, key_list, time_wp)
                # Execute waypoint in crazyswarm lab testing
                if lab_testing:
                    startWaypoint = timeit.default_timer()
                    os.chdir("/home/ryan/crazyswarm/ros_ws/src/crazyswarm/scripts")
                    os.system("/home/ryan/crazyswarm/ros_ws/src/crazyswarm/scripts/twtl_waypoint.py") # make sure executable
                    os.chdir("/home/ryan/Desktop/pyTWTL/src")
                    stopWaypoint = timeit.default_timer()
                    print 'Waypoint time, should be ~2.0sec: ', stopWaypoint - startWaypoint
                break
            else:
                # Update PA with new weights and policies to match
                iter_step += 1
                # This for loop simply finds index of the 0
                for key, pflag in enumerate(policy_flag):
                    if pflag == 0:
                        break
                key = key+1
                # Now recompute the control policy with updated edge weights
                init_loc = pa_control_policy_dict[key][-1]
                # Either compute receding horizon or dijkstra's shortest path
                if compute_local:
                    local_flag[key] = True
                    ts_policy[key], pa_policy[key] = \
                            two_hop_horizon(pa_nom_dict[key], weighted_nodes, weighted_soft_nodes, num_hops, init_loc)
                else:
                    # Check if pa_prime.final is in the set of weighted nodes
                    final_state_count = 0
                    for p in pa_nom_dict[key].final:
                        for node in weighted_nodes:
                            if node in p:
                                final_state_count += 1
                                break
                    # If all final nodes occupied, update final_set to be lowest energy feasible node(s)
                    if final_state_count == len(pa_nom_dict[key].final):
                        final_flag = True
                    else:
                        final_flag = False
                    if final_flag:
                        for prev in ts_prev_states:
                            if prev not in weighted_nodes:
                                weighted_nodes.append(prev)
                        for node in weighted_nodes:
                            if init_loc[0] == node:
                                weighted_nodes.remove(node)
                        pa_prime = update_weight(pa_nom_dict[key], weighted_nodes)
                        # update final accepting state(s)
                        update_final_state(pa_nom_dict[key], pa_prime, weighted_nodes, init_loc)
                    else:
                        pa_prime = update_weight(pa_nom_dict[key], weighted_nodes)

                    # Get control policy from current location
                    ts_policy[key], pa_policy[key], tau_dict[key] = \
                            compute_control_policy2(pa_prime, dfa_dict[key], init_loc) # Look at tau later ***
                    # Get rid of the duplicate node
                    ts_policy[key].pop(0)
                    pa_policy_temp = list(pa_policy[key])
                    pa_policy_temp.pop(0)
                    # account for final state issue
                    if final_flag == True:
                        ts_policy[key].append(ts_policy[key][0])
                        pa_policy_temp.append(pa_policy_temp[0])
                    pa_policy[key] = tuple(pa_policy_temp)
                    # Determine if last policy
                    if len(ts_policy) == 1:
                        break
                # Write updates to file
                write_to_iter_file(ts_policy[key], ts_dict[key], ets_dict[key], key, iter_step)
                # Update policy match
                policy_match, key_list, policy_match_index = update_policy_match(ts_policy)

        # Update policy_match now that a trajectory has finalized and policy_match is empty
        if ts_policy:
            # Remove keys from policies that have terminated
            land_keys = []
            for key, val in ts_policy.items():
                if len(val) == 0:
                    land_keys.append(key)
                    del ts_policy[key]
                    del pa_policy[key]
            # publish to the land csv file for lab testing
            if land_keys:
                if lab_testing:
                    write_to_land_file(land_keys)
                    os.chdir("/home/ryan/crazyswarm/ros_ws/src/crazyswarm/scripts")
                    os.system("/home/ryan/crazyswarm/ros_ws/src/crazyswarm/scripts/twtl_land.py") # make sure executable
                    os.chdir("/home/ryan/Desktop/pyTWTL/src")
            if not ts_policy:
                running = False
                break
            # Update policy match
            policy_match, key_list, policy_match_index = update_policy_match(ts_policy)
            # Get agent priority based on lowest energy
            priority = get_priority(pa_nom_dict, pa_policy, key_list)
        else:
            running = False

    # Print run time statistics
    stopOnline = timeit.default_timer()
    print 'Online run time for safe algorithm: ', stopOnline - startOnline
    stopFull = timeit.default_timer()
    print 'Full run time for safe algorithm: ', stopFull - startFull

    # Possibly just set the relaxation to the nominal + additional nodes added *** Change (10/28)
    for key in pa_nom_dict:
        tau_dict[key] = tau_dict_nom[key] + len(ts_control_policy_dict[key])-len(ts_policy_dict_nom[key])

    # Write the nominal and final control policies to a file
    for key in pa_nom_dict:
        write_to_control_policy_file(ts_policy_dict_nom[key], pa_policy_dict_nom[key], \
                output_dict_nom[key], tau_dict_nom[key], dfa_dict[key],ts_dict[key],ets_dict[key],\
                ts_control_policy_dict[key], pa_control_policy_dict[key], tau_dict[key], key)
    # Write the CSV files for experiments
    for key in pa_nom_dict:
        write_to_csv(ts_dict[key], ts_control_policy_dict[key], key, time_wp)


def get_priority(pa_nom_dict, pa_policy, key_list):
    ''' Computes the agent priority based on lowest energy. '''
    priority = []
    temp_energy = {}
    for key in key_list:
        temp_energy[key] = pa_nom_dict[key].g.node[pa_policy[key][0]]['energy']
    # Sort the energy values found for all agents
    sorted_energy = sorted(temp_energy.items(), key=operator.itemgetter(1))
    for key, energy_val in sorted_energy:
        priority.append(key)
    return priority

def two_hop_horizon(pa, weighted_nodes, soft_nodes, num_hops, init_loc):
    ''' Compute the two hop local horizon when imminenet collision detected
    or still within the local neighborhood. '''
    ts_policy = []
    pa_policy = []
    # Create local one-hop set
    local_set = pa.g.neighbors(init_loc)
    keep_nodes = {}
    for neighbor in local_set:
        for node in pa.g.nodes(data='true'):
            if neighbor == node[0] and node[0][0] not in weighted_nodes:
                keep_nodes[node[0]] = node[1]['energy']
                break
    if not keep_nodes:
        ts_policy.append([init_loc[0],init_loc[0]])
        pa_policy.append(init_loc, init_loc)
        print 'No feasible location to move, therefore stay in current position'
        return ts_policy, pa_policy
    # iterate through all hops
    if num_hops < 1:
        one_hop_node = min(keep_nodes, key=keep_nodes.get)
    else:
        for keep_node in keep_nodes:
            temp_node = keep_node
            energy_temp = keep_nodes[keep_node]
            for i in range(num_hops):
                local_set = pa.g.neighbors(temp_node)
                energy_low = float('inf')
                for neighbor in local_set:
                    for node in pa.g.nodes(data='true'):
                        try:
                            soft_nodes[i+1]
                        except KeyError:
                            soft_nodes[i+1] = []
                        if neighbor == node[0] and node[0][0] not in soft_nodes[i+1]:
                            energy_temp = node[1]['energy']
                            if energy_temp < energy_low:
                                temp_node = node[0]
                                energy_low = energy_temp
                                break
                if energy_low == float('inf'):
                    ts_policy.append([keep_node[0],keep_node[0]])
                    pa_policy.append(keep_node, keep_node)
                    print 'No feasible location to move, therefore stay in current position'
                    return ts_policy, pa_policy
                energy_temp = energy_temp + energy_low
            keep_nodes[keep_node] = energy_temp
        one_hop_node = min(keep_nodes, key=keep_nodes.get)
    # Append the policies
    ts_policy.append(one_hop_node[0])
    pa_policy.append(one_hop_node)
    # Use the energy function to get the second hop
    local_set2 = pa.g.neighbors(one_hop_node)
    keep_nodes2 = {}
    if num_hops > 0:
        for neighbor in local_set2:
            for node in pa.g.nodes(data='true'):
                if neighbor == node[0] and node[0][0] not in soft_nodes[1]:
                    keep_nodes2[node[0]] = node[1]['energy']
                    break
    else:
        for neighbor in local_set2:
            for node in pa.g.nodes(data='true'):
                if neighbor == node[0] and node[0][0] not in weighted_nodes:
                    keep_nodes2[node[0]] = node[1]['energy']
                    break
    if not keep_nodes2:
        ts_policy.append(one_hop_node[0])
        pa_policy.append(one_hop_node)
        print 'No feasible location to move, therefore stay in current position'
        return ts_policy, pa_policy
    if num_hops < 2:
        two_hop_node = min(keep_nodes2, key=keep_nodes2.get)
    else:
        for keep_node2 in local_set2:
            temp_node = keep_node2
            energy_temp = keep_nodes[keep_node]
            for i in range(num_hops-1):
                local_set = pa.g.neighbors(temp_node)
                energy_low = float('inf')
                for neighbor in local_set2:
                    for node in pa.g.nodes(data='true'):
                        try:
                            soft_nodes[i+2]
                        except KeyError:
                            soft_nodes[i+2] = []
                        if neighbor == node[0] and node[0][0] not in soft_nodes[i+2]:
                            energy_temp = node[1]['energy']
                            if energy_temp < energy_low:
                                temp_node = node[0]
                                energy_low = energy_temp
                                break
                if energy_low == float('inf'):
                    ts_policy.append([keep_node2[0],keep_node2[0]])
                    pa_policy.append(keep_node2, keep_node2)
                    print 'No feasible location to move, therefore stay in current position'
                    return ts_policy, pa_policy
                energy_temp = energy_temp + energy_low
            keep_nodes2[keep_node2] = energy_temp
        two_hop_node = min(keep_nodes2, key=keep_nodes2.get)
    # Append policies returned
    ts_policy.append(two_hop_node[0])
    pa_policy.append(two_hop_node)
    return ts_policy, pa_policy

def extend_horizon(pa, pa_node):
    ''' This extends the receding horizon trajectory when immediate conflict
    not seen but still in the local neighborhood of another agent. '''
    ignore_flag = False
    ts_policy = []
    pa_policy = []
    ts_policy.append(pa_node[0])
    pa_policy.append(pa_node)
    # Get local neighborhood
    local_set = pa.g.neighbors(pa_node)
    energy_low = float('inf')
    for neighbor in local_set:
        for node in pa.g.nodes(data='true'):
            if neighbor == node[0]:
                energy_temp = node[1]['energy']
                if energy_temp < energy_low:
                    energy_low = energy_temp
                    next_node = node[0]
                    break
    if energy_low == float('inf'):
        ignore_flag = True
        next_node = pa_node
        print 'No feasible location to move, therefore stay in current position'
    if energy_low < 1:
        ignore_flag = True
    ts_policy.append(next_node[0])
    pa_policy.append(next_node)
    return ts_policy, pa_policy, ignore_flag

def update_final_state(pa, pa_prime, weighted_nodes, init_loc):
    ''' Use of the energy function to get ideal final state to move to in
    the case where all final accepting states are occupied. '''
    # Create local one-hop set and remove current node
    local_set = pa.g.neighbors(init_loc)
    local_set.remove(init_loc)
    energy_fin = float('inf')
    for node in local_set:
        if node not in weighted_nodes:
            for i in pa.g.nodes(data='true'):
                if i[0] == node:
                    temp_energy = i[1]['energy']
                    break
            if temp_energy < energy_fin:
                energy_fin = temp_energy
                temp_node = node
    pa_prime.final.add(temp_node)

def get_neighborhood(node, ts):
    ''' function to get the two-hop neighborhood of nodes to compare for
    collision avoidance '''
    local_set = ts.g.neighbors(node)
    two_hop_set = []
    for local_node in local_set:
        two_hop_temp = ts.g.neighbors(local_node)
        for two_hop_node in two_hop_temp:
            if two_hop_node not in two_hop_set:
                two_hop_set.append(two_hop_node)
    # Merge one hop and two hop sets
    for i in two_hop_set:
        if i not in local_set:
            local_set.append(i)
    return local_set

def update_policy_match(ts_policy):
    ''' This takes care of updating the policies which are compared
    throughout the run. '''
    ts_shortest = float('inf')
    for match_key in ts_policy:
        if len(ts_policy[match_key]) < ts_shortest:
            ts_shortest = len(ts_policy[match_key])
    temp_match = [[] for i in range(ts_shortest)]
    # Add all previous control policies to temp_match
    # Gives a way to account for what sets policy_match is taking from
    key_list = ts_policy.keys()
    policy_match_index = ts_policy.keys()
    for match_key in ts_policy:
        for ind, item in enumerate(ts_policy[match_key]):
            if ind >= ts_shortest:
                break
            else:
                temp_match[ind].append(item)
    # Set policy_match
    return temp_match, policy_match_index, key_list

def update_weight(pa_prime, s_token):
    ''' Update edge weights of PA when a collision between nodes is detected.
    This searches the edges and if an edge is connected to the obstacle node
    then we assign updated weight. '''
    for s in s_token:
        for i in pa_prime.g.edges():
            for item in i:
                if s in item:
                    weight_new = pa_prime.g.number_of_edges()/2 + 1
                    temp = list(i)
                    temp.append(weight_new)
                    pa_prime.g.add_weighted_edges_from([tuple(temp)])
                    break
    return pa_prime

def setup_logging():
    fs, dfs = '%(asctime)s %(levelname)s %(message)s', '%m/%d/%Y %I:%M:%S %p'
    loglevel = logging.DEBUG
    logging.basicConfig(filename='../output/examples_RP8.log', level=loglevel,
                        format=fs, datefmt=dfs)
    root = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    ch.setFormatter(logging.Formatter(fs, dfs))
    root.addHandler(ch)

if __name__ == '__main__':
    setup_logging()
    # case study 1: Synthesis
    phi1 = '[H^2 r21]^[0, 7] * [H^1 r12]^[0, 7]'
    # Add a second agent
    phi2 = '[H^2 r21]^[0, 8] * [H^1 r23]^[0, 7]'
    # Add a third agent
    phi3 = '[H^2 r31]^[0, 8] * [H^1 r10]^[0, 10]'
    # Currently set to use the same transition system
    phi = [phi1, phi2, phi3]
    ts_files = ['../data/ts_synth_6x6_3D1.txt', '../data/ts_synth_6x6_3D2.txt', '../data/ts_synth_6x6_3D3.txt']
    # ts_files = ['../data/ts_synth_6x6_diag1.txt', '../data/ts_synth_6x6_diag2.txt', '../data/ts_synth_6x6_diag3.txt']
    # Set the time to go from one waypoint to the next (seconds)
    time_wp = 1.5
    # Define the radius of agents considering, this is for diagonal collision avoidance
    radius = 0.1
    # Set to True if running on Crazyflies in the lab
    lab_testing = False
    # consider more than one hop out, not here but just a note 12/5 ***
    case1_synthesis(phi, ts_files, radius, time_wp, lab_testing)
