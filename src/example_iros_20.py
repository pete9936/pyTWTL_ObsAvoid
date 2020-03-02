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
                      verify, compute_control_policy2, compute_control_policy3,\
                      compute_energy
from geometric_funcs import check_intersect, downwash_check
from write_files import write_to_land_file, write_to_csv_iter, write_to_csv,\
                        write_to_iter_file, write_to_control_policy_file
from learning import learn_deadlines
from lomap import Ts


def case1_synthesis(formulas, ts_files, alpha, radius, time_wp, lab_testing):
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

    # Compute optimal path in Pa_Prime and project onto the TS, and initial policy based on moc_weight
    ts_policy_dict_nom = {}
    pa_policy_dict_nom = {}
    tau_dict_nom = {}
    for key in pa_nom_dict:
        ts_policy_dict_nom[key], pa_policy_dict_nom[key], tau_dict_nom[key] = \
                    compute_control_policy(pa_nom_dict[key], dfa_dict[key], dfa_dict[key].kind)
    for key in pa_nom_dict:
        ts_policy_dict_nom[key], pa_policy_dict_nom[key] = \
                    compute_control_policy3(pa_nom_dict[key], dfa_dict[key], pa_policy_dict_nom[key][0])
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
    # Create Agent energy dictionary for post-processing
    agent_energy_dict = {}
    for key in ts_policy_dict_nom:
        agent_energy_dict[key] = []

    # Print time statistics
    stopOff = timeit.default_timer()
    print 'Offline run time for all initial setup: ', stopOff - startOff
    startOnline = timeit.default_timer()
    # pdb.set_trace()

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
                if p_ind < 1:
                    weighted_nodes = []
                    weighted_soft_nodes = {}
                    for i in range(num_hops-1):
                        weighted_soft_nodes[i+1] = []
                else:
                    # Get local neighborhood (n-hop) of nodes to search for a conflict
                    for k, key in enumerate(key_list):
                        if p_val == key:
                            node = policy_match[0][k]
                            break
                    # Note that communication range needs to be 2*H, the receding horizon length
                    local_set = get_neighborhood(node, ts_dict[p_val], 2*num_hops)
                    one_hop_set = ts_dict[p_val].g.neighbors(node)
                    # Assign constraints for immediate transition
                    weighted_nodes = []
                    for pty in priority[0:p_ind]:
                        for k, key in enumerate(key_list):
                            if pty == key:
                                prev_node = policy_match[0][k]
                                if prev_node in one_hop_set:
                                    weighted_nodes.append(prev_node)
                                # Check if downwash constraint needs to be added, mostly for physical testing
                                downwash_weight = downwash_check(k, ets_dict[key], policy_match[0], \
                                                                priority[0:k], key_list, radius)
                                if downwash_weight:
                                    for downwash_node in downwash_weight:
                                        if downwash_node not in weighted_nodes:
                                            weighted_nodes.append(downwash_node)
                                break
                    # Get constraints for later transitions
                    soft_nodes = {}
                    for pty in priority[0:p_ind]:
                        for k, key in enumerate(key_list):
                            if pty == key:
                                ts_length = len(ts_policy[key])
                                if ts_length >= num_hops:
                                    for i in range(num_hops-1):
                                        if ts_policy[key][i+1] in local_set:
                                            try:
                                                soft_nodes[i+1]
                                                soft_nodes[i+1].append(ts_policy[key][i+1])
                                            except KeyError:
                                                soft_nodes[i+1] = [ts_policy[key][i+1]]
                                else:
                                    for i in range(ts_length-1):
                                        if ts_policy[key][i+1] in local_set:
                                            try:
                                                soft_nodes[i+1]
                                                soft_nodes[i+1].append(ts_policy[key][i+1])
                                            except KeyError:
                                                soft_nodes[i+1] = [ts_policy[key][i+1]]
                                for i in range(num_hops-1):
                                    try:
                                        soft_nodes[i+1]
                                    except KeyError:
                                        soft_nodes[i+1] = []
                    # Assign later constraint nodes
                    weighted_soft_nodes = soft_nodes
                    # Update weights if transitioning between same two nodes
                    ts_prev_states = []
                    ts_index = []
                    if len(policy_match[0]) > 1 and traj_length >= 1:
                        for key in ts_control_policy_dict:
                            if len(ts_control_policy_dict[key]) == traj_length:
                                ts_prev_states.append(ts_control_policy_dict[key][-1])
                    if ts_prev_states:
                        for p_ind2, p_val2 in enumerate(priority[0:p_ind+1]):
                            if p_ind2 > 0:
                                for k_c, key in enumerate(key_list):
                                    if p_val2 == key:
                                        node = policy_match[0][k_c]
                                        break
                                # Check if the trajectories will cross each other in transition
                                cross_weight = check_intersect(k_c, ets_dict[key], ts_prev_states, policy_match[0], \
                                                                    priority[0:p_ind2], key_list, radius, time_wp)
                                # if traj_length > 4:
                                #     pdb.set_trace()
                                if cross_weight:
                                    for cross_node in cross_weight:
                                        if cross_node not in weighted_nodes:
                                            weighted_nodes.append(cross_node)
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
                # Compute local horizon function to account for receding horizon all the time
                # while checking for termination
                if traj_length >= 1:
                    init_loc = pa_control_policy_dict[p_val][-1]
                    # Compute receding horizon shortest path
                    ts_policy[p_val], pa_policy[p_val] = \
                        local_horizon(pa_nom_dict[p_val], weighted_nodes, weighted_soft_nodes, num_hops, init_loc)
                    # Write updates to file
                    iter_step += 1
                    # write_to_iter_file(ts_policy[p_val], ts_dict[p_val], ets_dict[p_val], p_val, iter_step)

                # Update policy match
                policy_match, key_list, policy_match_index = update_policy_match(ts_policy)

            # Append trajectories
            for key in ts_policy:
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
    # plot_energy(agent_energy_dict)

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

def findPaths(pa, init, n):
    ''' This takes a network pa, a node init, and a length n. It recursively finds
    all paths of length n-1 starting from neighbors of init. '''
    if n==0:
        return [[init]]
    if init in pa.final:
        return [[init]]
    paths = []
    for neighbor in pa.g.neighbors(init):
        for path in findPaths(pa,neighbor,n-1):
            if init not in path:
                paths.append([init]+path)
    return paths

def local_horizon(pa, weighted_nodes, soft_nodes, num_hops, init_loc):
    ''' Compute the n-hop lowest energy horizon without imminent conflict and
    incorporating penalty for soft constraints based on number of hops from
    the current state. '''
    ts_policy = []
    pa_policy = []
    epsilon = 0.001

    # Compute the n-hop trajectory, ensures a minimum 2-hop trajectory
    soft_nodes[0] = weighted_nodes
    index = 0
    paths_temp = findPaths(pa, init_loc, num_hops)
    paths = copy.deepcopy(paths_temp)
    # Get rid of all infeasible paths due to constraints
    for path in paths_temp:
        for ind, node in enumerate(path[1::]):
            if node[0] in soft_nodes[ind]:
                paths.remove(path)
                break

    all_node_energy = nx.get_node_attributes(pa.g,'energy')
    path_energy = []
    for path in paths:
        temp_energy = 0
        for node in path[1::]:
            node_energy = all_node_energy[node]
            temp_energy = temp_energy + node_energy
        path_energy.append(temp_energy)
    # Get index of the minimum energy path
    index_min = min(xrange(len(path_energy)), key=path_energy.__getitem__)

    for node in paths[index_min][1::]:
        ts_policy.append(node[0])
        pa_policy.append(node)

    return ts_policy, pa_policy

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
    logging.basicConfig(filename='../output/examples_iros_20.log', level=loglevel,
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
    # case study 1: Synthesis
    phi1 = '[H^2 r21]^[0, 6] * [H^1 r12]^[0, 5]'
    phi2 = '[H^2 r21]^[0, 5] * [H^1 r23]^[0, 4]'
    phi3 = '[H^1 r86]^[0, 4] * [H^2 r97]^[0, 4]'
    phi4 = '[H^2 r89]^[0, 5] * [H^1 Base4]^[0, 3]'
    phi5 = '[H^1 r105]^[0, 6] * [H^1 Base5]^[0, 6]'
    # Set to use the same transition system
    phi = [phi1, phi2, phi3, phi4, phi5]
    ts_files = ['../data/ts_6x6x3_5Ag_1.txt', '../data/ts_6x6x3_5Ag_2.txt', '../data/ts_6x6x3_5Ag_3.txt', \
                '../data/ts_6x6x3_5Ag_4.txt', '../data/ts_6x6x3_5Ag_5.txt']

    ''' Define alpha [0:1] for weighted average function: w' = min[alpha*time_weight + (1-alpha)*edge_weight]
        Note: For alpha=0 we only account for the weighted transition system (edge_weight),
              for alpha=1 we only account for minimizing time (time_weight)
              and thus becomes a single-objective optimization problem.
              Otherwise it is a multi-objective cost minimization of the two factors. '''
    alpha = 0.5
    # Set the time to go from one waypoint to the next (seconds), accounts for agent dynamics
    time_wp = 1.9
    # Define the radius (m) of agents considered, used for diagonal collision avoidance and to avoid downwash
    radius = 0.1
    # Set to True if running on Crazyflies in the lab
    lab_testing = True
    case1_synthesis(phi, ts_files, alpha, radius, time_wp, lab_testing)
