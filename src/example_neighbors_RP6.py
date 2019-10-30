'''
.. module:: example_neighbors_RP6.py
   :synopsis: Case studies for American Control Conference (2020).

.. moduleauthor:: Ryan Peterson <pete9936@umn.edu.edu>

'''

import logging, sys
import StringIO
import pdb, os, copy, math
import timeit

import networkx as nx
import matplotlib.pyplot as plt

import twtl
from dfa import DFAType
from synthesis import expand_duration_ts, compute_control_policy, ts_times_fsa,\
                      verify, compute_control_policy2, compute_energy
from learning import learn_deadlines
from lomap import Ts


def case1_synthesis(formulas, ts_files):
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
        print 'Size of TS:', pa_nom_dict[key].size()

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
    energy_dict = {}
    for key in ts_policy_dict_nom:
        compute_energy(pa_nom_dict[key], dfa_dict[key])

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

    # Iterate through all policies sequentially
    while running:
        while policy_match:
            for ind, node in enumerate(policy_match[0]):
                if ind < 1:
                    append_flag = True
                else:
                    # Get local neighborhood (two-hop) of nodes to search for a conflict
                    local_set = get_neighborhood(node, ts_dict[ind+1])
                    prev_nodes = []
                    for prev_node in policy_match[0][0:ind]:
                        if prev_node in local_set:
                            prev_nodes.append(prev_node)
                    # check if local flags should be set or if one is switching off
                    if prev_nodes:
                        local_flag[key_list[ind]] = True
                    elif local_flag[key_list[ind]] == True and len(ts_policy[key_list[ind]])==1:
                        local_flag[key_list[ind]] = False
                        switch_flag = True
                        policy_flag[key_list[ind]-1] = 0
                        break
                    else:
                        local_flag[key_list[ind]] = False
                    if node in prev_nodes:
                        policy_flag[key_list[ind]-1] = 0
                        append_flag = False
                        compute_local = True
                        break
                    else:
                        policy_flag[key_list[ind]-1] = 1
                        append_flag = True
            temp = prev_nodes
            weighted_nodes = temp
            # Update weights if transitioning between same two nodes
            ts_prev_states = []
            ts_index = []
            if len(policy_match[0]) > 1 and traj_length >= 1:
                for key in ts_control_policy_dict:
                    if len(ts_control_policy_dict[key]) == traj_length:
                        ts_prev_states.append(ts_control_policy_dict[key][-1])
            if ts_prev_states:
                for ind_cur, node in enumerate(policy_match[0]):
                    if ind_cur > 0:
                        for i in range(ind_cur):
                            if policy_match[0][i] == node:
                                temp_node = ts_control_policy_dict[policy_match_index[i]][-1]
                                if temp_node not in weighted_nodes:
                                    weighted_nodes.append(temp_node)
                                append_flag = False
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
                policy_flag[key_list[ind]-1] = 0
            if final_count > 2:
                final_count = 0
            # Account for receding horizon trajectories
            if switch_flag:
                compute_local = False
                append_flag = False
                switch_flag = False
            elif len(policy_match) == 1:
                for key in key_list:
                    if local_flag[key] == True and append_flag == True: # update local trajectory and move on
                        init_loc = pa_control_policy_dict[key][-1]
                        ts_temp = ts_policy[key]
                        pa_temp = pa_policy[key]
                        ts_policy[key], pa_policy[key], ignore = extend_horizon(pa_nom_dict[key], weighted_nodes, pa_policy[key][0])
                        if ignore:
                            # This accounts for termination criteria
                            ts_policy[key] = ts_temp
                            pa_policy[key] = pa_temp
                            break
                        else:
                            policy_match, key_list, policy_match_index = update_policy_match(ts_policy)
                            break
                    elif local_flag[key] == True and append_flag == False: # update local trajectory later
                        compute_local = True
                        break
                    elif local_flag[key] == False and append_flag == False: # update trajectory w/ Dijkstra's later
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
                policy_match.pop(0)
                traj_length += 1
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
                    ts_policy[key], pa_policy[key] = two_hop_horizon(pa_nom_dict[key], weighted_nodes, init_loc)
                else:
                    # local_flag[key] = False
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
            for key, val in ts_policy.items():
                if len(val) == 0:
                    del ts_policy[key]
                    del pa_policy[key]
            if not ts_policy:
                running = False
                break
            # Update policy match
            policy_match, key_list, policy_match_index = update_policy_match(ts_policy)
        else:
            running = False

    # Possibly just set the relaxation to the nominal + additional nodes added *** Change (10/28)
    for key in pa_nom_dict:
        tau_dict[key] = tau_dict_nom[key] + len(ts_control_policy_dict[key])-len(ts_policy_dict_nom[key])

    # Write the nominal and final control policies to a file
    for key in pa_nom_dict:
        write_to_control_policy_file(ts_policy_dict_nom[key], pa_policy_dict_nom[key], \
                output_dict_nom[key], tau_dict_nom[key], dfa_dict[key],ts_dict[key],ets_dict[key],\
                ts_control_policy_dict[key], pa_control_policy_dict[key], tau_dict[key], key)


def two_hop_horizon(pa, weighted_nodes, init_loc):
    ''' Compute the two hop local horizon when imminenet collision detected
    or still within the local neighborhood. '''
    ts_policy = []
    pa_policy = []
    # Create local one-hop set and remove current node
    local_set = pa.g.neighbors(init_loc)
    # Use the energy function to get the first hop
    energy_low = float('inf')
    for neighbor in local_set:
        for node in pa.g.nodes(data='true'):
            if neighbor == node[0] and node[0][0] not in weighted_nodes:
                if node[1]['energy'] < energy_low:
                    energy_low = node[1]['energy']
                    one_hop_node = node[0]
                    break
    if energy_low == float('inf'):
        one_hop_node = init_loc
        print 'No feasible location to move, therefore stay in current position'

    ts_policy.append(one_hop_node[0])
    pa_policy.append(one_hop_node)
    # Create local second-hop set and remove current node
    two_hop_temp = pa.g.neighbors(one_hop_node)
    # Use the energy function to get the second hop
    energy_low = float('inf')
    for neighbor in two_hop_temp:
        for node in pa.g.nodes(data='true'):
            if neighbor == node[0] and node[0][0] not in weighted_nodes:
                if node[1]['energy'] < energy_low:
                    energy_low = node[1]['energy']
                    two_hop_node = node[0]
                    break
    if energy_low == float('inf'):
        two_hop_node = one_hop_node
        print 'No feasible location to move, therefore stay in current position'
    # Append policies returned
    ts_policy.append(two_hop_node[0])
    pa_policy.append(two_hop_node)
    return ts_policy, pa_policy

def extend_horizon(pa, weighted_nodes, pa_node):
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
            if neighbor == node[0] and node[0][0] not in weighted_nodes:
                if node[1]['energy'] < energy_low:
                    energy_low = node[1]['energy']
                    next_node = node[0]
                    break
    if energy_low == float('inf'):
        ignore_flag = True
        next_node = pa_node
        print 'No feasible location to move, therefore stay in current position'
    if energy_low < 3:
        ignore_flag = True
    ts_policy.append(next_node[0])
    pa_policy.append(next_node)
    return ts_policy, pa_policy, ignore_flag

def update_final_state(pa, pa_prime, weighted_nodes, init_loc):
    ''' Use of the energy function to get ideal final state to move to in
    the case where all final accpeting states are occupied. '''
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
    # Need a way to account for what sets policy_match is taking from
    key_list = []
    policy_match_index = []
    for match_key in ts_policy:
        key_list.append(match_key)
        policy_match_index.append(match_key)
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

def write_to_iter_file(policy, ts, ets, key, iter_step):
    ''' Writes each iteration of the control policy to an output file
    to keep track of the changes and updates being made. '''
    policy = [x for x in policy if x not in ets.state_map]
    out = StringIO.StringIO()
    for u, v in zip(policy[:-1], policy[1:]):
        print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
    print>>out, policy[-1],
    logging.info('Generated control policy is: %s', out.getvalue())
    if os.path.isfile('../output/control_policy_updates_RP6.txt'):
        with open('../output/control_policy_updates_RP6.txt', 'a+') as f1:
            f1.write('Control Policy for agent %s at step ' % key)
            f1.write('%s:  ' % iter_step)
            f1.write('%s\n\n' % out.getvalue())
    else:
        with open('../output/control_policy_updates_RP6.txt', 'w+') as f1:
            f1.write('Control Policy for agent %s at step ' % key)
            f1.write('%s:  ' % iter_step)
            f1.write('%s\n\n' % out.getvalue())
    f1.close()
    out.close()

def write_to_control_policy_file(ts_nom_policy, pa_nom_policy, output, tau, dfa, ts, ets, ts_policy, pa_policy, tau_new, key):
    ''' This writes the nominal and final control policy for each agent to
    an output file. '''
    logging.info('Max deadline: %s', tau)
    if ts_nom_policy is not None:
        logging.info('Generated output word is: %s', [tuple(o) for o in output])
        policy = [x for x in ts_nom_policy if x not in ets.state_map]
        out = StringIO.StringIO()
        for u, v in zip(policy[:-1], policy[1:]):
            print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
        print>>out, policy[-1],
        logging.info('Generated control policy is: %s', out.getvalue())
        if os.path.isfile('../output/control_policy_RP6.txt'):
            with open('../output/control_policy_RP6.txt', 'a+') as f2:
                f2.write('Nominal Control Policy for agent %s.\n' % key)
                f2.write('Optimal relaxation is: %s \n' % tau)
                f2.write('Generated PA control policy is: (')
                f2.write(') -> ('.join('%s %s' % x for x in pa_nom_policy))
                f2.write(') \nGenerated TS control policy is: %s \n\n' % ts_nom_policy)
                f2.write('Final Control policy for agent %s.\n' % key)
                f2.write('Optimal relaxation is: %s \n' % tau_new)
                f2.write('Generated PA control policy is: (')
                f2.write(') -> ('.join('%s %s' % x for x in pa_policy))
                f2.write(') \nGenerated TS control policy is:  %s \n\n' % ts_policy)
        else:
            with open('../output/control_policy_RP6.txt', 'w+') as f2:
                f2.write('Nominal Control Policy for agent %s.\n' % key)
                f2.write('Optimal relaxation is: %s \n' % tau)
                f2.write('Generated PA control policy is: (')
                f2.write(') -> ('.join('%s %s' % x for x in pa_nom_policy))
                f2.write(') \nGenerated control policy is: %s \n\n' % ts_nom_policy)
                f2.write('Final Control policy for agent %s.\n' % key)
                f2.write('Generated PA control policy is: (')
                f2.write(') -> ('.join('%s %s' % x for x in pa_policy))
                f2.write(') \nGenerated TS control policy is:  %s \n\n' % ts_policy)
        f2.close()
        out.close()
    else:
        logging.info('No control policy found!')

def setup_logging():
    fs, dfs = '%(asctime)s %(levelname)s %(message)s', '%m/%d/%Y %I:%M:%S %p'
    loglevel = logging.DEBUG
    logging.basicConfig(filename='../output/examples_RP6.log', level=loglevel,
                        format=fs, datefmt=dfs)
    root = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    ch.setFormatter(logging.Formatter(fs, dfs))
    root.addHandler(ch)

if __name__ == '__main__':
    setup_logging()
    # case study 1: Synthesis
    # phi1 = '[H^2 V]^[0, 7] * [H^2 M]^[0, 7]'
    phi1 = '[H^2 r21]^[0, 7] * [H^2 r12]^[0, 7]'
    # Add a second agent
    # phi2 = '[H^2 N]^[0, 8] * [H^2 X]^[0, 7]'
    phi2 = '[H^2 r21]^[0, 8] * [H^2 r23]^[0, 7]'
    # Add a third agent
    # phi3 = '[H^2 f]^[0, 8] * [H^3 K]^[0, 10]'
    phi3 = '[H^2 r31]^[0, 8] * [H^3 r10]^[0, 10]'
    # Currently set to use the same transition system
    phi = [phi1, phi2, phi3]
    ts_files = ['../data/ts_synth_6x6_test1.txt', '../data/ts_synth_6x6_test2.txt', '../data/ts_synth_6x6_test3.txt']
    # ts_files = ['../data/ts_synthesis_6x6_obs1.txt', '../data/ts_synthesis_6x6_obs2.txt', '../data/ts_synthesis_6x6_obs3.txt']
    case1_synthesis(phi, ts_files)
