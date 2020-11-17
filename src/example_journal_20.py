'''
.. module:: example_journal_20.py
   :synopsis: Case studies for TWTL package and lab testing for journal 2020
                submission.

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
                        write_to_iter_file, write_to_control_policy_file, write_to_priority
from DP_paths import local_horizonDP, deadlock_path
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
        dfa_dict[ind+1] = copy.deepcopy(dfa_inf) # The key is set to the agent number

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
    # Choose parameter for n-horizon local trajectory, must be at least 2
    num_hops = 2
    # Get agent priority based on lowest energy
    prev_states = {}
    for key in ts_policy_dict_nom:
        prev_states[key] = pa_policy_dict_nom[key][0]
    priority = get_priority(pa_nom_dict, pa_policy_dict_nom, prev_states, key_list)
    # Create Agent energy dictionary for post-processing
    # Create Termination indicator to assign terminated agents lowest priority
    F_indicator = {}
    agent_energy_dict = {}
    for key in ts_policy_dict_nom:
        agent_energy_dict[key] = []
        F_indicator[key] = False

    # Print time statistics
    stopOff = timeit.default_timer()
    print 'Offline run time for all initial setup: ', stopOff - startOff
    startOnline = timeit.default_timer()

    # Execute takeoff command for all crazyflies in lab testing
    if lab_testing:
        startTakeoff = timeit.default_timer()
        os.chdir("/home/ryan/crazyswarm/ros_ws/src/crazyswarm/scripts")
        os.system("/home/ryan/crazyswarm/ros_ws/src/crazyswarm/scripts/twtl_takeoff.py") # make sure executable
        os.chdir("/home/ryan/Desktop/pyTWTL/src")
        stopTakeoff = timeit.default_timer()
        print 'Takeoff time, should be ~2.7sec: ', stopTakeoff - startTakeoff

    ############################################################################

    # Iterate through all policies sequentially
    while running:
        while policy_match:
            for p_ind, p_val in enumerate(priority):
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
                    # Receive path information from 2*H neighborhood
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
                                                # Add downwash nodes to constraint, for drone experiments
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
                                                # Add downwash nodes to constraint, for drone experiments
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
                # Generate receding horizon path and check for termination
                if traj_length >= 1:
                    init_loc = pa_control_policy_dict[p_val][-1]
                    ts_temp = ts_policy[p_val]
                    pa_temp = pa_policy[p_val]
                    # Compute receding horizon shortest path
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
                                            for j in priority[0:p_ind+1]:
                                                occupied_nodes.append(ts_control_policy_dict[j][-1])
                                            init_loc = pa_control_policy_dict[x_d_val][-1]
                                            ts_policy[x_d_val], pa_policy[x_d_val] = deadlock_path(pa_nom_dict[x_d_val], occupied_nodes, init_loc)
                                            for j in priority[1:p_ind]:
                                                for ind, node in enumerate(ts_policy[x_d_val][:-1]):
                                                    if ts_policy[j][0] == node:
                                                        ts_policy[j] = [ts_policy[x_d_val][ind+1],ts_policy[x_d_val][ind+1]]
                                                        # Find the actual state on agent's PA that corresponds to this node
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

            # Account for agents which have finished, also accounts for other finished agents through agent ID ordering
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
                agent_energy_dict[key].append(pa_nom_dict[key].g.node[pa_policy[key][0]]['energy'])
                ts_control_policy_dict[key].append(ts_policy[key].pop(0))
                pa_policy_temp = list(pa_policy[key])
                pa_control_policy_dict[key].append(pa_policy_temp.pop(0))
                pa_policy[key] = tuple(pa_policy_temp)
            ts_write = policy_match.pop(0)
            traj_length += 1

            # publish waypoint to a csv file
            write_to_csv_iter(ts_dict, ts_write, key_list, time_wp)
            # write_to_priority(priority)
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
                # publish to the land csv file when finished (for experiments)
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

    # Print energy graph for each agent and the system from run
    plot_energy(agent_energy_dict)

    # Not exact, but gives insight
    for key in pa_nom_dict:
        tau_dict[key] = tau_dict_nom[key]+len(ts_control_policy_dict[key])-len(ts_policy_dict_nom[key])

    # Write the nominal and final control policies to a file
    for key in pa_nom_dict:
        write_to_control_policy_file(ts_policy_dict_nom[key], pa_policy_dict_nom[key], \
                tau_dict_nom[key], dfa_dict[key],ts_dict[key],ets_dict[key],\
                ts_control_policy_dict[key], pa_control_policy_dict[key], tau_dict[key], key)

    # Write the CSV files for experiments
    for key in pa_nom_dict:
        write_to_csv(ts_dict[key], ts_control_policy_dict[key], key, time_wp)

################################################################################

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
    logging.basicConfig(filename='../output/example_S2J.log', level=loglevel,
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
    plt.legend()
    plt.subplot(212)
    plt.ylabel('System Energy', fontsize=14)
    plt.xlabel('time-steps', fontsize=14)
    datax = np.arange(len(sys_energy))
    plt.plot(datax, sys_energy,'bo:', linewidth=4.5)
    plt.show()

if __name__ == '__main__':
    setup_logging()
    # Define TWTL Specifications for each agent
    # Scenario 1J, Journal complete example
    # phi1 = '[H^2 r16]^[0, 3] * [H^1 r27]^[0, 5] * [H^0 Base1]^[0, 5]' # A, D
    # phi2 = '([H^3 r12]^[0, 6] | [H^3 r13]^[0, 6] | [H^3 r20]^[0, 6]) * ([H^2 r7]^[0, 7] | [H^2 r14]^[0, 7])  * [H^0 Base2]^[0, 3]' # B or C, E
    # phi3 = '([H^2 r12]^[0, 6] | [H^2 r13]^[0, 6] | [H^2 r20]^[0, 6]) * ([H^2 r7]^[0, 7] | [H^2 r14]^[0, 7]) * [H^0 Base3]^[0, 3]' # B or C, E
    # phi4 = '([H^2 r12]^[0, 6] | [H^2 r13]^[0, 6] | [H^2 r20]^[0, 6]) * ([H^2 r7]^[0, 7] | [H^2 r14]^[0, 7]) * [H^0 Base4]^[0, 3]'  # B or C, E
    # phi5 = '[H^2 r16]^[0, 5] * [H^2 r0]^[0, 5] * [H^0 Base5]^[0, 5]'  # A, F
    phi1 = '[H^1 r2]^[0, 5] * ([H^3 r10]^[0, 7] | [H^3 r11]^[0, 7] | [H^3 r5]^[0, 7])' # P1, D1 or D2 or D3
    phi2 = '[H^1 r14]^[0, 5] * ([H^3 r10]^[0, 7] | [H^3 r11]^[0, 7] | [H^3 r5]^[0, 7])' # P2, D1 or D2 or D3
    phi3 = '[H^1 r14]^[0, 5] * ([H^3 r10]^[0, 7] | [H^3 r11]^[0, 7] | [H^3 r5]^[0, 7])' # P2, D1 or D2 or D3
    phi = [phi1, phi2, phi3]
    ts_files = ['../data/scenario2J/ts_3x6x1_3Ag_1.txt', '../data/scenario2J/ts_3x6x1_3Ag_2.txt', '../data/scenario2J/ts_3x6x1_3Ag_3.txt']

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
    always_active = True
    # Set to True if running on Crazyflies in the lab
    lab_testing = False
    case1_synthesis(phi, ts_files, alpha, radius, time_wp, lab_testing, always_active)
