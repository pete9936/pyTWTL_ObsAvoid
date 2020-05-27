'''
.. module:: example_multicost_RP9.py
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
import numpy as np

import twtl
import write_files

from dfa import DFAType
from synthesis import expand_duration_ts, compute_control_policy, ts_times_fsa,\
                      verify, compute_energy
from geometric_funcs import check_intersectDP, downwash_checkDP
from write_files import write_to_land_file, write_to_csv_iter, write_to_csv,\
                        write_to_iter_file, write_to_control_policy_file
from learning import learn_deadlines
from lomap import Ts


def case1_synthesis(formulas, ts_files, alpha, radius, time_wp, lab_testing, always_active):
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
    norm_factor = {}
    startPA = timeit.default_timer()
    for key in dfa_dict:
        logging.info('Constructing product automaton with infinity DFA!')
        pa = ts_times_fsa(ets_dict[key], dfa_dict[key])
        # Give length and weight attributes to all edges in pa
        nom_weight_dict = {}
        edges_all = nx.get_edge_attributes(ts_dict[key].g,'edge_weight')
        max_edge = max(edges_all, key=edges_all.get)
        norm_factor[key] = edges_all[max_edge]
        for pa_edge in pa.g.edges():
            edge = (pa_edge[0][0], pa_edge[1][0], 0)
            nom_weight_dict[pa_edge] = edges_all[edge]/norm_factor[key]
        nx.set_edge_attributes(pa.g, 'edge_weight', nom_weight_dict)
        nx.set_edge_attributes(pa.g, 'weight', 1)
        logging.info('Product automaton size is: (%d, %d)', *pa.size())
        # Make a copy of the nominal PA to change
        pa_nom_dict[key] = copy.deepcopy(pa)
    stopPA = timeit.default_timer()
    print 'Run Time (s) to get all three PAs is: ', stopPA - startPA

    for key in pa_nom_dict:
        print 'Size of PA:', pa_nom_dict[key].size()

    # Use alpha to perform weighted optimization of time and edge_weight and make this a
    # new edge attribute to find "shortest path" over
    for key in pa_nom_dict:
        weight_dict = {}
        time_weight = nx.get_edge_attributes(pa_nom_dict[key].g,'weight')
        edge_weight = nx.get_edge_attributes(pa_nom_dict[key].g,'edge_weight')
        for pa_edge in pa_nom_dict[key].g.edges():
            weight_dict[pa_edge] = alpha*time_weight[pa_edge] + (1-alpha)*edge_weight[pa_edge]
        # Append the multi-objective cost to the edge attribtues of the PA
        nx.set_edge_attributes(pa_nom_dict[key].g, 'new_weight', weight_dict)

    # Compute the energy (multi-objective cost function) for each agent's PA at every node
    startEnergy = timeit.default_timer()
    for key in pa_nom_dict:
        compute_energy(pa_nom_dict[key])
    stopEnergy = timeit.default_timer()
    print 'Run Time (s) to get the moc energy function for all three PA: ', stopEnergy - startEnergy

    # Compute optimal path in PA and project onto the TS
    ts_policy_dict_nom = {}
    pa_policy_dict_nom = {}
    tau_dict_nom = {}
    for key in pa_nom_dict:
        ts_policy_dict_nom[key], pa_policy_dict_nom[key], tau_dict_nom[key] = \
                    compute_control_policy(pa_nom_dict[key], dfa_dict[key], dfa_dict[key].kind)
    # Perform initial check on nominal control policies
    for key in ts_policy_dict_nom:
        if ts_policy_dict_nom[key] is None:
            logging.info('No control policy found!')

    # set empty control policies that will be iteratively updated
    ts_control_policy_dict = {}
    pa_control_policy_dict = {}

    # Initialize policy variables
    for key in ts_policy_dict_nom:
        ts_control_policy_dict[key] = []
        pa_control_policy_dict[key] = []

    # Concatenate nominal policies for searching
    policy_match, key_list, policy_match_index = update_policy_match(ts_policy_dict_nom)

    # Initialize vars, give nominal policies
    iter_step = 0
    running = True
    traj_length = 0
    ts_policy = copy.deepcopy(ts_policy_dict_nom)
    pa_policy = copy.deepcopy(pa_policy_dict_nom)
    tau_dict = tau_dict_nom
    # Choose parameter for n-horizon local trajectory and information sharing,
    # must be at least 2
    num_hops = 2
    # Get agent priority based on lowest energy
    prev_states = {}
    for key in ts_policy_dict_nom:
        prev_states[key] = pa_policy_dict_nom[key][0]
    priority = get_priority(pa_nom_dict, pa_policy_dict_nom, prev_states, key_list)
    # Create Agent energy dictionary for post-processing, and deadlock flags
    # Create Termination indicator to assign terminated agents lowest priority
    # and some temporary ts_policy
    D_flags = {}
    F_indicator = {}
    tsf_policy = {}
    agent_energy_dict = {}
    for key in ts_policy_dict_nom:
        agent_energy_dict[key] = []
        # D_flags[key] = False
        F_indicator[key] = False
        tsf_policy[key] = []

    # Print time statistics
    stopOff = timeit.default_timer()
    print 'Offline run time for all initial setup: ', stopOff - startOff
    startOnline = timeit.default_timer()

    # Execute takeoff command for all crazyflies in lab testing
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
                # D_flag_tot = False
                if p_ind < 1:
                    weighted_nodes = {}
                    for i in range(num_hops):
                        weighted_nodes[i] = []
                else:
                    # Get local neighborhood (n-hop) of nodes to search for a conflict
                    for k_c, key_c in enumerate(key_list):
                        if p_val == key_c:
                            node = policy_match[0][k_c]
                            break
                    # Note that communication range needs to be 2*H, the receding horizon length
                    local_set = get_neighborhood(node, ts_dict[p_val], 2*num_hops)
                    # Get constraints for each transition
                    weighted_nodes = {}
                    for pty in priority[0:p_ind]:
                        for key in key_list:
                            if pty == key:
                                ts_length = len(ts_policy[key])
                                if ts_length >= num_hops:
                                    for i in range(num_hops):
                                        if ts_policy[key][i] in local_set:
                                            try:
                                                weighted_nodes[i].append(ts_policy[key][i])
                                                # Add downwash nodes to constraint, mostly for physical testing
                                                downwash_nodes = downwash_checkDP(ets_dict[key], ts_policy[key][i], radius)
                                                if downwash_nodes:
                                                    for downwash_node in downwash_nodes:
                                                        if downwash_node not in weighted_nodes[i]:
                                                            weighted_nodes[i].append(downwash_node)
                                            except KeyError:
                                                weighted_nodes[i] = [ts_policy[key][i]]
                                                downwash_nodes = downwash_checkDP(ets_dict[key], ts_policy[key][i], radius)
                                                if downwash_nodes:
                                                    for downwash_node in downwash_nodes:
                                                        if downwash_node not in weighted_nodes[i]:
                                                            weighted_nodes[i].append(downwash_node)
                                else:
                                    for i in range(ts_length):
                                        if ts_policy[key][i] in local_set:
                                            try:
                                                weighted_nodes[i].append(ts_policy[key][i])
                                                # Add downwash nodes to constraint, mostly for physical testing
                                                downwash_nodes = downwash_checkDP(ets_dict[key], ts_policy[key][i], radius)
                                                if downwash_nodes:
                                                    for downwash_node in downwash_nodes:
                                                        if downwash_node not in weighted_nodes[i]:
                                                            weighted_nodes[i].append(downwash_node)
                                            except KeyError:
                                                weighted_nodes[i] = [ts_policy[key][i]]
                                                downwash_nodes = downwash_checkDP(ets_dict[key], ts_policy[key][i], radius)
                                                if downwash_nodes:
                                                    for downwash_node in downwash_nodes:
                                                        if downwash_node not in weighted_nodes[i]:
                                                            weighted_nodes[i].append(downwash_node)
                                for i in range(num_hops):
                                    try:
                                        weighted_nodes[i]
                                    except KeyError:
                                        weighted_nodes[i] = []
                    # Update constraint set with intersecting transitions
                    if traj_length >= 1:
                        for p_ind2, p_val2 in enumerate(priority[0:p_ind]):
                            for k, key in enumerate(key_list):
                                if p_val2 == key:
                                    # initialize previous state
                                    comp_prev_state = ts_control_policy_dict[key][-1]
                                    cur_prev_state = ts_control_policy_dict[key_c][-1]
                                    cur_ts_policy_length = len(ts_policy[key_c])
                                    ts_length = len(ts_policy[key])
                                    if ts_length >= num_hops:
                                        for i in range(num_hops):
                                            comp_next_state = ts_policy[key][i]
                                            if i < cur_ts_policy_length:
                                                cur_next_state = ts_policy[key_c][i]
                                                if comp_next_state in local_set:
                                                    # Check if the trajectories cross during transition (or use same transition)
                                                    cross_weight = check_intersectDP(ets_dict[key], cur_prev_state, cur_next_state, \
                                                                                comp_prev_state, comp_next_state, radius, time_wp)
                                                    if cross_weight:
                                                        for cross_node in cross_weight:
                                                            if cross_node not in weighted_nodes[i]:
                                                                weighted_nodes[i].append(cross_node)
                                                    # Check if using same transition in updated case
                                                    if comp_next_state == cur_prev_state:
                                                        if comp_prev_state not in weighted_nodes[i]:
                                                            weighted_nodes[i].append(comp_prev_state)
                                                    # Set previous state for next iteration
                                                    comp_prev_state = ts_policy[key][i]
                                                    cur_prev_state = ts_policy[key_c][i]
                                            else:
                                                break
                                    else:
                                        for i in range(ts_length):
                                            comp_next_state = ts_policy[key][i]
                                            if i < cur_ts_policy_length:
                                                cur_next_state = ts_policy[key_c][i]
                                                if comp_next_state in local_set:
                                                    # Check if the trajectories cross during transition (or use same transition)
                                                    cross_weight = check_intersectDP(ets_dict[key], cur_prev_state, cur_next_state, \
                                                                                comp_prev_state, comp_next_state, radius, time_wp)
                                                    if cross_weight:
                                                        for cross_node in cross_weight:
                                                            if cross_node not in weighted_nodes[i]:
                                                                weighted_nodes[i].append(cross_node)
                                                    # Check if using same transition in updated case
                                                    if comp_next_state == cur_prev_state:
                                                        if comp_prev_state not in weighted_nodes[i]:
                                                            weighted_nodes[i].append(comp_prev_state)
                                                    # Set previous state for next iteration
                                                    comp_prev_state = ts_policy[key][i]
                                                    cur_prev_state = ts_policy[key_c][i]
                                            else:
                                                break
                # Generate receding horizon all the time while checking for termination
                if traj_length >= 1:
                    init_loc = pa_control_policy_dict[p_val][-1]
                    # Compute receding horizon shortest path
                    ts_temp = ts_policy[p_val]
                    pa_temp = pa_policy[p_val]
                    ts_policy[p_val], pa_policy[p_val], D_flag = local_horizonDP(pa_nom_dict[p_val], weighted_nodes, num_hops, init_loc)
                    # Check for deadlock, and if so resolve deadlock
                    if p_ind > 0:
                        if D_flag == True:
                            # Agent in deadlock is to remain stationary
                            ts_policy[p_val] = [ts_control_policy_dict[p_val][-1],ts_control_policy_dict[p_val][-1]]
                            pa_policy[p_val] = [pa_control_policy_dict[p_val][-1],pa_control_policy_dict[p_val][-1]]
                            # Assign deadlock node
                            x_d = ts_control_policy_dict[p_val][-1]
                            x_d_val = p_val
                            x_d_flag = True
                            hp_set = priority[0:p_ind]
                            while x_d_flag == True and hp_set:
                                x_d_flag = False
                                for hp in hp_set:
                                    if ts_policy[hp][0] == x_d:
                                        if hp == priority[0]:
                                            # Make all agents stationary and perform Dijkstra's shortest path
                                            for j in priority[1:p_ind]:
                                                ts_policy[j] = [ts_control_policy_dict[j][-1],ts_control_policy_dict[j][-1]]
                                                pa_policy[j] = [pa_control_policy_dict[j][-1],pa_control_policy_dict[j][-1]]
                                            occupied_nodes = [ts_control_policy_dict[x_d_val][-1]]
                                            for j in priority[0:p_ind]:
                                                occupied_nodes.append(ts_control_policy_dict[j][-1])
                                            init_loc = pa_control_policy_dict[x_d_val][-1]
                                            ts_policy[x_d_val], pa_policy[x_d_val] = deadlock_path(pa_nom_dict[x_d_val], occupied_nodes, init_loc)
                                            for j in priority[1:p_ind]:
                                                for ind, node in enumerate(ts_policy[x_d_val][:-1]):
                                                    if ts_policy[j][0] == node:
                                                        ts_policy[j] = [ts_policy[x_d_val][ind+1],ts_policy[x_d_val][ind+1]]
                                                        # Find the actual state on PA that corresponds to this
                                                        neighbors = pa_nom_dict[j].g.neighbors(pa_policy[j][0])
                                                        for node in neighbors:
                                                            if node[0] == ts_policy[j][0]:
                                                                pa_policy[j] = [node, node]
                                            break
                                        else:
                                            ts_policy[hp] = [ts_control_policy_dict[hp][-1],ts_control_policy_dict[hp][-1]]
                                            pa_policy[hp] = [pa_control_policy_dict[hp][-1],pa_control_policy_dict[hp][-1]]
                                            x_d = ts_control_policy_dict[hp][-1]
                                            x_d_val = hp
                                            x_d_flag = True
                                            hp_set.remove(hp)
                                            break
                    # Increase iteration step (for statistics at end)
                    iter_step += 1

            # Update policy match
            policy_match, key_list, policy_match_index = update_policy_match(ts_policy)

            # Account for agents which have finished, also accouns for other finished agents through agent ID ordering
            if always_active == True:
                finished_ID = []
                for key in F_indicator:
                    if F_indicator[key] == True:
                        finished_ID.append(key)
                        current_node = ts_control_policy_dict[key][-1]
                        hp_nodes_avoid = []
                        for k in key_list:
                            hp_nodes_avoid.append(ts_policy[k][0])
                            hp_nodes_avoid.append(ts_control_policy_dict[k][-1])
                        for fID in finished_ID[:-1]:
                            hp_nodes_avoid.append(ts_control_policy_dict[fID][-1])
                        if current_node in hp_nodes_avoid:
                            local_set = ts_dict[key].g.neighbors(current_node)
                            for node in local_set:
                                if node not in hp_nodes_avoid:
                                    ts_control_policy_dict[key].append(node)
                                    break
                        else:
                            ts_control_policy_dict[key].append(current_node)

            # Append trajectories
            for key in ts_policy:
                D_flags[key] = False
                agent_energy_dict[key].append(pa_nom_dict[key].g.node[pa_policy[key][0]]['energy'])
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

            # Update policy_match now that a trajectory has finalized and policy_match is empty
            if ts_policy:
                # Remove keys from policies that have terminated
                land_keys = []
                for key, val in ts_policy.items():
                    if len(val) == 0:
                        F_indicator[key] = True
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
                for key in key_list:
                    prev_states[key] = pa_control_policy_dict[key][-1]
                priority = get_priority(pa_nom_dict, pa_policy, prev_states, key_list)
            else:
                running = False

    # Print run time statistics
    stopOnline = timeit.default_timer()
    print 'Online run time for safe algorithm: ', stopOnline - startOnline
    stopFull = timeit.default_timer()
    print 'Full run time for safe algorithm: ', stopFull - startFull
    # Print other statistics from simulation
    print 'Number of iterations for run: ', iter_step
    print 'Average time for itertion is: ', (stopOnline - startOnline)/iter_step
    print 'Number of full updates in run: ', traj_length
    print 'Average update time for single step: ', (stopOnline - startOnline)/traj_length

    # Print energy statistics from run
    plot_energy(agent_energy_dict)

    # Possibly just set the relaxation to the nominal + additional nodes added *** Change (10/28)
    for key in pa_nom_dict:
        tau_dict[key] = tau_dict_nom[key] + len(ts_control_policy_dict[key])-len(ts_policy_dict_nom[key])

    # Write the nominal and final control policies to a file
    for key in pa_nom_dict:
        write_to_control_policy_file(ts_policy_dict_nom[key], pa_policy_dict_nom[key], \
                tau_dict_nom[key], dfa_dict[key],ts_dict[key],ets_dict[key],\
                ts_control_policy_dict[key], pa_control_policy_dict[key], tau_dict[key], key)
    # Write the CSV files for experiments
    for key in pa_nom_dict:
        write_to_csv(ts_dict[key], ts_control_policy_dict[key], key, time_wp)


def get_priority(pa_nom_dict, pa_policy, prev_states, key_list):
    ''' Computes the agent priority based on lowest energy. '''
    priority = []
    temp_energy = {}
    progress_check = {}
    for key in key_list:
        temp_energy[key] = pa_nom_dict[key].g.node[pa_policy[key][0]]['energy']
    # Sort the energy values found for all agents in descending energy order
    sorted_energy = sorted(temp_energy.items(), key=operator.itemgetter(1))
    # Generate set of priorities with lwoest energy given highest priority
    for key, energy_val in sorted_energy:
        priority.append(key)
    return priority

def deadlock_path(pa, occupied_nodes, init_loc):
    ''' Find the sortest path to the nearest unoccupied node for
    deadlock resolution. '''
    ts_path = []
    pa_path = []
    conflict_nodes = [occupied_nodes[0], occupied_nodes[1]]
    # Initialize local set
    local_set = init_loc
    old_local_set = {}
    open_flag = False
    target_set = []
    i = 0
    # Expand subgraph of nodes out until unoccupied node is found
    while open_flag == False:
        # Expand subgraph for next time step
        old_local_set[i] = local_set
        local_set_temp = []
        local_set = []
        if i == 0:
            temp_set = pa.g.neighbors(old_local_set[i])
            for node in temp_set:
                if node not in local_set_temp:
                    local_set_temp.append(node)
        else:
            for loc_node in old_local_set[i]:
                temp_set = pa.g.neighbors(loc_node)
                for node in temp_set:
                    if node not in local_set_temp:
                        local_set_temp.append(node)
        # Remove conflicting nodes in the set
        for neighbor in local_set_temp:
            if neighbor[0] not in conflict_nodes:
                local_set.append(neighbor)
        # Check if unoccupied nodes
        for node in local_set:
            if node not in occupied_nodes:
                target_set.append(node)
                open_flag = True
        # Perform another hop
        if not target_set:
            local_set = local_set_temp
        i = i+1
    # Search for lowest energy value in remaining set of nodes and use for shortest path
    all_node_energy = nx.get_node_attributes(pa.g,'energy')
    target_set_energy = []
    for node in target_set:
        node_energy = all_node_energy[node]
        target_set_energy.append(node_energy)
    # Get index of the minimum energy path
    index_min = min(xrange(len(target_set_energy)), key=target_set_energy.__getitem__)
    target_node = target_set[index_min]
    # Perform the targeted DP method
    num_trans = i
    edges_all = nx.get_edge_attributes(pa.g,'new_weight')
    all_node_energy = nx.get_node_attributes(pa.g,'energy')
    node_dict = {}
    node_costs = {}
    if num_trans > 1:
        for j in range(num_trans-1):
            advance_flag = False
            node_list_temp = []
            trans_cost_temp = []
            if j == 0:
                check_nodes = pa.g.neighbors(target_node)
                for check_node in check_nodes:
                    if check_node in old_local_set[num_trans-1]:
                        advance_flag = True
                        edge = (check_node, target_node)
                        try:
                            trans_cost = all_node_energy[check_node] + edges_all[edge]
                            node_list_temp.append(check_node)
                            trans_cost_temp.append(trans_cost)
                        except KeyError:
                            # If we are traversing on the same state (i.e. ('r21',1)->('r21',2))
                            new_edge = pa.g.in_edges(target_node)[0]
                            trans_cost = all_node_energy[new_edge[0]] + edges_all[new_edge]
                            node_list_temp = [new_edge[0]]
                            trans_cost_temp = [trans_cost]
                            break
                if advance_flag == False:
                    try:
                        new_edge = ((target_node[0],target_node[1]-1), target_node)
                        trans_cost = all_node_energy[new_edge[0]] + edges_all[new_edge]
                        node_dict[num_trans-1] = [new_edge[0]]
                        node_costs[num_trans-1] = [trans_cost]
                    except KeyError:
                        edges = pa.g.in_edges(target_node)
                        for edge in edges:
                            if edge[0] in old_local_set[num_trans-1]:
                                trans_cost = all_node_energy[edge[0]] + edges_all[edge]
                                node_list_temp.append(edge[0])
                                trans_cost_temp.append(trans_cost)
                        node_dict[num_trans-1] = node_list_temp
                        node_costs[num_trans-1] = trans_cost_temp
                else:
                    node_dict[num_trans-1] = node_list_temp
                    node_costs[num_trans-1] = trans_cost_temp
            else:
                for old_node in node_dict[num_trans-j]:
                    old_node_index = node_dict[num_trans-j].index(old_node)
                    old_node_cost = node_costs[num_trans-j][old_node_index]
                    check_nodes = pa.g.neighbors(old_node)
                    for check_node in check_nodes:
                        if check_node in old_local_set[num_trans-j-1]:
                            advance_flag = True
                            edge = (check_node, old_node)
                            try:
                                trans_cost = old_node_cost + all_node_energy[check_node] + edges_all[edge]
                                # Need a check on trans_cost to see if it's the lowest (replace if needed)
                                if check_node in node_list_temp:
                                    check_index = node_list_temp.index(check_node)
                                    if trans_cost_temp[check_index] > trans_cost:
                                        trans_cost_temp[check_index] = trans_cost
                                else:
                                    node_list_temp.append(check_node)
                                    trans_cost_temp.append(trans_cost)
                            except KeyError:
                                # If we are traversing on the same state (i.e. ('r21',1)->('r21',2))
                                new_edge = pa.g.in_edges(old_node)[0]
                                trans_cost = old_node_cost + all_node_energy[new_edge[0]] + edges_all[new_edge]
                                node_list_temp = [new_edge[0]]
                                trans_cost_temp = [trans_cost]
                                break
                    # Save updated dictionary values for next step
                    if advance_flag == False:
                        try:
                            new_edge = ((old_node[0],old_node[1]-1), old_node)
                            trans_cost = old_node_cost + all_node_energy[new_edge[0]] + edges_all[new_edge]
                            node_dict[num_trans-j-1] = [new_edge[0]]
                            node_costs[num_trans-j-1] = [trans_cost]
                            break
                        except KeyError:
                            edges = pa.g.in_edges(old_node)
                            for edge in edges:
                                if edge[0] in old_local_set[num_trans-j-1]:
                                    trans_cost = old_node_cost + all_node_energy[edge[0]] + edges_all[edge]
                                    node_list_temp.append(edge[0])
                                    trans_cost_temp.append(trans_cost)
                            node_dict[num_trans-j-1] = node_list_temp
                            node_costs[num_trans-j-1] = trans_cost_temp
                            break
                if advance_flag == True:
                    node_dict[num_trans-j-1] = node_list_temp
                    node_costs[num_trans-j-1] = trans_cost_temp
        # Handle the last transition differently by just adding in the edge_weight
        for ind, old_node in enumerate(node_dict[1]):
            edge = (init_loc, old_node)
            node_costs[1][ind] = node_costs[1][ind] + edges_all[edge]

        # Construct lowest cost feasible path based on DP calculations above
        path = []
        for key in node_dict:
            if key == 1:
                # Get index of the minimum cost reachable node
                index_min = min(xrange(len(node_costs[1])), key=node_costs[1].__getitem__)
                path_node_prior = node_dict[1][index_min]
                path.append(path_node_prior)
            else:
                check_nodes = pa.g.neighbors(path_node_prior)
                path_node_cost = float('inf')
                for check_node in check_nodes:
                    if check_node in node_dict[key]:
                        temp_cost = node_costs[key][node_dict[key].index(check_node)]
                        if temp_cost < path_node_cost:
                            path_node_cost = temp_cost
                            path_node_prior = check_node
                path.append(path_node_prior)
        # Append the final target node to path
        path.append(target_node)
        # generate policy based on the generated path
        for p in path:
            ts_path.append(p[0])
            pa_path.append(p)
    else:
        ts_path.append(target_node[0])
        ts_path.append(target_node[0])
        pa_path.append(target_node)
        pa_path.append(target_node)

    return ts_path, pa_path

def local_horizonDP(pa, weighted_nodes, num_hops, init_loc):
    ''' Compute the n-hop lowest energy horizon without conflicts using
    a targeted Dynamic Programming (DP) method. '''
    # Compute the n-hop trajectory, ensures a minimum 2-hop trajectory
    # Performs check on deadlock as well
    ts_policy = []
    pa_policy = []
    D_flag = False
    # Initialize local set
    local_set = init_loc
    final_flag = False
    old_local_set = {}
    # Expand subgraph of nodes out to num_hops for search
    for i in range(num_hops):
        # Expand subgraph for next time step
        old_local_set[i] = local_set
        local_set_temp = []
        local_set = []
        if i == 0:
            temp_set = pa.g.neighbors(old_local_set[i])
            for node in temp_set:
                if node not in local_set_temp:
                    local_set_temp.append(node)
        else:
            for loc_node in old_local_set[i]:
                temp_set = pa.g.neighbors(loc_node)
                for node in temp_set:
                    if node not in local_set_temp:
                        local_set_temp.append(node)
        # Remove conflicting nodes in the set
        for neighbor in local_set_temp:
            if neighbor[0] not in weighted_nodes[i]:
                local_set.append(neighbor)
        # Check if the agent is in a deadlock situation, or there are no feasible transitions
        if not local_set:
            if i==0:
                D_flag = True
                return None, None, D_flag
            else:
                i = i-1
                local_set = old_local_set[i]
                break
        # Check if any of the nodes are in the final set and if so break and use node
        for node in local_set:
            if node in pa.final:
                final_flag = True
                target_node = node
                break
        else:
            continue
        break
    # Search for lowest energy value in remaining set of nodes and use for shortest path
    if final_flag == False:
        all_node_energy = nx.get_node_attributes(pa.g,'energy')
        local_set_energy = []
        for node in local_set:
            node_energy = all_node_energy[node]
            local_set_energy.append(node_energy)
        # Get index of the minimum energy path
        index_min = min(xrange(len(local_set_energy)), key=local_set_energy.__getitem__)
        target_node = local_set[index_min]

    # Perform the targeted DP method
    num_trans = i+1
    edges_all = nx.get_edge_attributes(pa.g,'new_weight')
    all_node_energy = nx.get_node_attributes(pa.g,'energy')
    node_dict = {}
    node_costs = {}
    if num_trans > 1:
        for j in range(num_trans-1):
            advance_flag = False
            node_list_temp = []
            trans_cost_temp = []
            if j == 0:
                check_nodes = pa.g.neighbors(target_node)
                for check_node in check_nodes:
                    if check_node in old_local_set[num_trans-1]:
                        advance_flag = True
                        edge = (check_node, target_node)
                        try:
                            trans_cost = all_node_energy[check_node] + edges_all[edge]
                            node_list_temp.append(check_node)
                            trans_cost_temp.append(trans_cost)
                        except KeyError:
                            # If we are traversing on the same state (i.e. ('r21',1)->('r21',2))
                            new_edge = pa.g.in_edges(target_node)[0]
                            trans_cost = all_node_energy[new_edge[0]] + edges_all[new_edge]
                            node_list_temp = [new_edge[0]]
                            trans_cost_temp = [trans_cost]
                            break
                if advance_flag == False:
                    try:
                        new_edge = ((target_node[0],target_node[1]-1), target_node)
                        trans_cost = all_node_energy[new_edge[0]] + edges_all[new_edge]
                        node_dict[num_trans-1] = [new_edge[0]]
                        node_costs[num_trans-1] = [trans_cost]
                    except KeyError:
                        edges = pa.g.in_edges(target_node)
                        for edge in edges:
                            if edge[0] in old_local_set[num_trans-1]:
                                trans_cost = all_node_energy[edge[0]] + edges_all[edge]
                                node_list_temp.append(edge[0])
                                trans_cost_temp.append(trans_cost)
                        node_dict[num_trans-1] = node_list_temp
                        node_costs[num_trans-1] = trans_cost_temp
                else:
                    node_dict[num_trans-1] = node_list_temp
                    node_costs[num_trans-1] = trans_cost_temp
            else:
                for old_node in node_dict[num_trans-j]:
                    old_node_index = node_dict[num_trans-j].index(old_node)
                    old_node_cost = node_costs[num_trans-j][old_node_index]
                    check_nodes = pa.g.neighbors(old_node)
                    for check_node in check_nodes:
                        if check_node in old_local_set[num_trans-j-1]:
                            advance_flag = True
                            edge = (check_node, old_node)
                            try:
                                trans_cost = old_node_cost + all_node_energy[check_node] + edges_all[edge]
                                # Need a check on trans_cost to see if it's the lowest (replace if needed)
                                if check_node in node_list_temp:
                                    check_index = node_list_temp.index(check_node)
                                    if trans_cost_temp[check_index] > trans_cost:
                                        trans_cost_temp[check_index] = trans_cost
                                else:
                                    node_list_temp.append(check_node)
                                    trans_cost_temp.append(trans_cost)
                            except KeyError:
                                # If we are traversing on the same state (i.e. ('r21',1)->('r21',2))
                                new_edge = pa.g.in_edges(old_node)[0]
                                trans_cost = old_node_cost + all_node_energy[new_edge[0]] + edges_all[new_edge]
                                node_list_temp = [new_edge[0]]
                                trans_cost_temp = [trans_cost]
                                break
                    # Save updated dictionary values for next step
                    if advance_flag == False:
                        try:
                            new_edge = ((old_node[0],old_node[1]-1), old_node)
                            trans_cost = old_node_cost + all_node_energy[new_edge[0]] + edges_all[new_edge]
                            node_dict[num_trans-j-1] = [new_edge[0]]
                            node_costs[num_trans-j-1] = [trans_cost]
                            break
                        except KeyError:
                            edges = pa.g.in_edges(old_node)
                            for edge in edges:
                                if edge[0] in old_local_set[num_trans-j-1]:
                                    trans_cost = old_node_cost + all_node_energy[edge[0]] + edges_all[edge]
                                    node_list_temp.append(edge[0])
                                    trans_cost_temp.append(trans_cost)
                            node_dict[num_trans-j-1] = node_list_temp
                            node_costs[num_trans-j-1] = trans_cost_temp
                            break
                if advance_flag == True:
                    node_dict[num_trans-j-1] = node_list_temp
                    node_costs[num_trans-j-1] = trans_cost_temp
        # Handle the last transition differently by just adding in the edge_weight
        for ind, old_node in enumerate(node_dict[1]):
            edge = (init_loc, old_node)
            node_costs[1][ind] = node_costs[1][ind] + edges_all[edge]

        # Construct lowest cost feasible path based on DP calculations above
        path = []
        for key in node_dict:
            if key == 1:
                # Get index of the minimum cost reachable node
                index_min = min(xrange(len(node_costs[1])), key=node_costs[1].__getitem__)
                path_node_prior = node_dict[1][index_min]
                path.append(path_node_prior)
            else:
                check_nodes = pa.g.neighbors(path_node_prior)
                path_node_cost = float('inf')
                for check_node in check_nodes:
                    if check_node in node_dict[key]:
                        temp_cost = node_costs[key][node_dict[key].index(check_node)]
                        if temp_cost < path_node_cost:
                            path_node_cost = temp_cost
                            path_node_prior = check_node
                path.append(path_node_prior)
        # Append the final target node to path
        path.append(target_node)
        # generate policy based on the generated path
        for p in path:
            ts_policy.append(p[0])
            pa_policy.append(p)
    else:
        ts_policy.append(target_node[0])
        pa_policy.append(target_node)

    return ts_policy, pa_policy, D_flag

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

def get_neighborhood(node, ts, num_hops):
    ''' Function to get the n-hop neighborhood of nodes to compare for
    collision avoidance. Where num_hops is 2*H (the receding horizon length) '''
    local_set = ts.g.neighbors(node)
    n_hop_set = []
    for i in range(num_hops):
        for local_node in local_set:
            n_hop_temp = ts.g.neighbors(local_node)
            for n_hop_node in n_hop_temp:
                if n_hop_node not in n_hop_set:
                    n_hop_set.append(n_hop_node)
        # Merge i and i+1 node sets
        for node in n_hop_set:
            if node not in local_set:
                local_set.append(node)
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

def setup_logging():
    fs, dfs = '%(asctime)s %(levelname)s %(message)s', '%m/%d/%Y %I:%M:%S %p'
    loglevel = logging.DEBUG
    logging.basicConfig(filename='../output/example_DP.log', level=loglevel,
                        format=fs, datefmt=dfs)
    root = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    ch.setFormatter(logging.Formatter(fs, dfs))
    root.addHandler(ch)

def plot_energy(agent_energy):
    '''Generate plots for energy at each state along the trajectory. Plot for each
    agent and another for the total energy of the system.'''
    plt.figure(figsize=(8,5))
    plt.subplot(211)
    sys_energy = []
    for key in agent_energy:
        datay = np.asarray(agent_energy[key])
        datax = np.arange(len(datay))
        plt.plot(datax, datay, 'o--', label='Agent %s' %key, linewidth=2.0)
        if len(sys_energy) < len(datay):
            temp = datay.copy()
            temp[:len(sys_energy)] += sys_energy
            sys_energy = temp
        else:
            sys_energy[:len(datay)] += datay
    plt.ylabel('Agent Energy', fontsize=14)
    # plt.grid(axis='y')
    # plt.tick_params(labelsize=12)
    # plt.axis([0, 12, 0, 13])
    plt.legend()
    plt.subplot(212)
    plt.ylabel('System Energy', fontsize=14)
    plt.xlabel('time-steps', fontsize=14)
    datax = np.arange(len(sys_energy))
    plt.plot(datax, sys_energy,'bo:', linewidth=4.5)
    # plt.ytick_params(labelsize=12)
    # plt.xticks(datax)
    plt.show()

if __name__ == '__main__':
    setup_logging()
    # Define TWTL Specifications for each agent
    # Scenario 1, standard environment
    # phi1 = '[H^1 r29]^[0, 5] * [H^1 r105]^[0, 4] * [H^0 Base1]^[0, 4]' # B, F
    # phi2 = '[H^2 r21]^[0, 4] * [H^1 r55]^[0, 3] * [H^0 Base2]^[0, 3]' # A, E
    # phi3 = '[H^2 r21]^[0, 4] * [H^1 r55]^[0, 3] * [H^0 Base3]^[0, 3]' # A, E
    # phi4 = '[H^1 r9]^[0, 4] * [H^1 r12]^[0, 4] * [H^0 Base4]^[0, 3]'  # C, D
    # phi5 = '[H^1 r9]^[0, 4] * [H^1 r12]^[0, 4] * [H^0 Base5]^[0, 3]'  #C, D
    # # Set to use the same transition system
    # phi = [phi1, phi2, phi3, phi4, phi5]
    # ts_files = ['../data/scenario1ths/ts_6x6x3_5Ag_1.txt', '../data/scenario1ths/ts_6x6x3_5Ag_2.txt', \
    #       '../data/scenario1ths/ts_6x6x3_5Ag_3.txt', '../data/scenario1ths/ts_6x6x3_5Ag_4.txt', '../data/scenario1ths/ts_6x6x3_5Ag_5.txt']

    # Scenario 2, large enviroment
    # phi1 = '[H^1 r54]^[0, 4] * [H^2 r46]^[0, 7] * [H^0 Base1]^[0, 5]' # C, E
    # phi2 = '[H^1 r54]^[0, 4] * [H^2 r105]^[0, 7] * [H^0 Base2]^[0, 5]' # C, F
    # phi3 = '[H^1 r50]^[0, 8] * [H^1 r105]^[0, 8] * [H^0 Base3]^[0, 5]' # B, F
    # phi4 = '[H^1 r23]^[0, 8] * [H^1 r8]^[0, 5] * [H^0 Base4]^[0, 7]'  # D, H
    # phi5 = '[H^1 r13]^[0, 5] * [H^1 r8]^[0, 9] * [H^0 Base5]^[0, 8]'  # A, H
    # phi6 = '[H^1 r50]^[0, 5] * [H^1 r105]^[0, 8] * [H^0 Base6]^[0, 9]' # B, F
    # phi7 = '[H^1 r50]^[0, 5] * [H^2 r208]^[0, 5] * [H^0 Base7]^[0, 5]' # B, G
    # phi8 = '[H^1 r13]^[0, 4] * [H^1 r8]^[0, 8] * [H^0 Base8]^[0, 7]' # A, H
    # phi9 = '[H^1 r13]^[0, 4] * [H^1 r46]^[0, 9] * [H^0 Base9]^[0, 8]'  # A, E
    # phi10 = '[H^2 r54]^[0, 6] * [H^2 r208]^[0, 4] * [H^0 Base10]^[0, 7]' # C, G
    # phi = [phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10]
    # ts_files = ['../data/big_env/ts_6x12x4_10Ag_1.txt', '../data/big_env/ts_6x12x4_10Ag_2.txt', '../data/big_env/ts_6x12x4_10Ag_3.txt', \
    #             '../data/big_env/ts_6x12x4_10Ag_4.txt', '../data/big_env/ts_6x12x4_10Ag_5.txt', '../data/big_env/ts_6x12x4_10Ag_6.txt', \
    #             '../data/big_env/ts_6x12x4_10Ag_7.txt', '../data/big_env/ts_6x12x4_10Ag_8.txt', '../data/big_env/ts_6x12x4_10Ag_9.txt', \
    #             '../data/big_env/ts_6x12x4_10Ag_10.txt']

    # Scenario 3, tight corridor
    # phi1 = '[H^1 r5]^[0, 6]' # B
    # phi2 = '[H^1 r12]^[0, 7]' # A
    # phi3 = '[H^3 r6]^[0, 7]' # C
    # phi4 = '[H^4 Base1]^[0, 8]'
    # phi = [phi1, phi2, phi3, phi4]
    # ts_files = ['../data/corr/ts_3x6x1_4Ag_1.txt', '../data/corr/ts_3x6x1_4Ag_2.txt', '../data/corr/ts_3x6x1_4Ag_3.txt', \
    #             '../data/corr/ts_3x6x1_4Ag_4.txt']

    # Scenario 4, different drop-off points
    # phi1 = '[H^1 r2]^[0, 4] * ([H^1 r10]^[0,4] | [H^1 r11]^[0,4] | [H^1 r5]^[0,4])' # P1, D1 or D2 or D3
    # phi2 = '[H^1 r14]^[0, 4] * ([H^1 r10]^[0,4] | [H^1 r11]^[0,4] | [H^1 r5]^[0,4])' # P2, D1 or D2 or D3
    # phi1 = '[H^1 r2]^[0, 4] * [H^1 r10]^[0,4]' # P1, D1 or D2 or D3
    # phi2 = '[H^1 r14]^[0, 4] * [H^1 r10]^[0,4]' # P2, D1 or D2 or D3
    # phi = [phi1, phi2]
    # ts_files = ['../data/scenario4/ts_3x6x1_2Ag_1.txt', '../data/scenario4/ts_3x6x1_2Ag_2.txt']

    # Scenario 1J, Journal complete example
    phi1 = '[H^2 r16]^[0, 3] * [H^1 r27]^[0, 6] * [H^0 Base1]^[0, 5]' # A, D
    phi2 = '([H^2 r12]^[0, 6] | [H^2 r13]^[0, 6] | [H^2 r20]^[0, 6]) * ([H^2 r7]^[0, 7] | [H^2 r14]^[0, 7])  * [H^1 Base2]^[0, 3]' # B or C, E
    phi3 = '([H^2 r12]^[0, 6] | [H^2 r13]^[0, 6] | [H^2 r20]^[0, 6]) * ([H^2 r7]^[0, 7] | [H^2 r14]^[0, 7]) * [H^1 Base3]^[0, 3]' # B or C, E
    phi4 = '([H^2 r12]^[0, 6] | [H^2 r13]^[0, 6] | [H^2 r20]^[0, 6]) * ([H^2 r7]^[0, 7] | [H^2 r14]^[0, 7]) * [H^1 Base4]^[0, 3]'  # B or C, E
    phi5 = '[H^2 r16]^[0, 5] * [H^2 r0]^[0, 5] * [H^1 Base5]^[0, 6]'  # A, F
    # Set to use the same transition system
    phi = [phi1, phi2, phi3, phi4, phi5]
    ts_files = ['../data/scenario1J/ts_4x7x1_5Ag_1.txt', '../data/scenario1J/ts_4x7x1_5Ag_2.txt', '../data/scenario1J/ts_4x7x1_5Ag_3.txt', \
                '../data/scenario1J/ts_4x7x1_5Ag_4.txt', '../data/scenario1J/ts_4x7x1_5Ag_5.txt']

    ''' Define alpha [0:1] for weighted average function: w' = min[alpha*time_weight + (1-alpha)*edge_weight]
        Note: For alpha=0 we only account for the weighted transition system (edge_weight),
              for alpha=1 we only account for minimizing time (time_weight)
              and thus becomes a single-objective optimization problem.
              Otherwise it is a multi-objective cost minimization of the two factors. '''
    alpha = 0.5
    # Set the time to go from one waypoint to the next (seconds), accounts for agent dynamics
    time_wp = 2.0
    # Define the radius (m) of agents considered, used for diagonal collision avoidance and to avoid downwash
    radius = 0.1
    # Set to True if all agents are active until the last agent terminates its task
    always_active = False
    # Set to True if running on Crazyflies in the lab
    lab_testing = False
    case1_synthesis(phi, ts_files, alpha, radius, time_wp, lab_testing, always_active)
