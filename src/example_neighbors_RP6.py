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
    ts_shortest = float('inf')
    for init_key in ts_policy_dict_nom:
        if len(ts_policy_dict_nom[init_key]) < ts_shortest:
            ts_shortest = len(ts_policy_dict_nom[init_key])
    temp_match = [[] for i in range(ts_shortest)]

    # match up all policies into a list of lists
    policy_match_index = []
    for init_key in ts_policy_dict_nom:
        policy_match_index.append(init_key)
        for ind, item in enumerate(ts_policy_dict_nom[init_key]):
            if ind >= ts_shortest:
                break
            else:
                temp_match[ind].append(item)
    # Create the policy match list of tuples
    policy_match = temp_match

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
    key_list = []
    for key in ts_policy:
        key_list.append(key)

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
                    if node in prev_nodes:
                        policy_flag[key_list[ind]-1] = 0
                        append_flag = False
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
                        ts_index.append(key)
            if ts_prev_states:
                for ind1, ts_state in enumerate(ts_prev_states):
                    for ind2, node in enumerate(policy_match[0]):
                        if ts_state == node:
                            ts_comp_ind = ts_index[ind1]
                            pol_comp_ind = policy_match_index[ind2]
                            if ts_comp_ind != pol_comp_ind:
                                if policy_match_index[ind2] in ts_index and ts_index[ind1] in policy_match_index:
                                    if policy_match[0][policy_match_index.index(ts_comp_ind)] == ts_state and \
                                        ts_prev_states[ts_index.index(pol_comp_ind)] == node:
                                        if ind > ind2:
                                            weighted_nodes = weighted_nodes[0:ind2+1]
                                            policy_flag[key_list[ind]-1] = 1
                                            policy_flag[key_list[ind2]-1] = 0
                                            append_flag = False
                                            break
                                        elif ind == ind2:
                                            weighted_nodes.append(node)
                                            append_flag = False
                                            break
                    else:
                        continue
                    break
            else:
                append_flag = True
            if len(ts_policy) == 1 and final_flag == True:
                weighted_nodes = []
                append_flag = False
            if final_flag == True:
                final_count += 1
                policy_flag[key_list[ind]-1] = 0
            if final_count > 2:
                final_count = 0
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

                # Create local set and remove current node
                local_set = pa_nom_dict[key].g.neighbors(init_loc)
                local_set.remove(init_loc)


                # Create method to use local energy update instead of computing full path ***
                # Possibly make this dependent on overall trajectory length
                # Get a finite horizon based on lowest energy (gradient descent) that is non-violating
                # if local energy is below some threshold (e.g. 10) then compute full path based
                # on Dijkstra's
                # Need a flag that says we can not terminate but now need to compute receding horizon
                # path
                # if init_loc[1]['energy'] > 10:
                #   use local horizon
                #   add flag that says updates are now required for receding horizon
                # else:
                #   update weights and find Dijkstra's shortest path

                # Use the energy function to perform a local search
                energy_low = float('inf')
                for neighbor in local_set:
                    for node in pa_nom_dict[key].g.nodes(data='true'):
                        if neighbor == node[0] and node[0][0] not in weighted_nodes:
                            if node[1]['energy'] < energy_low:
                                energy_low = node[1]['energy']
                                next_node = node[0]
                                break
                if energy_low == float('inf'):
                    next_node = init_loc
                    print 'No feasible location to move, therefore stay in current position'

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
                    # use energy function to get ideal final state to move to
                    energy_fin = float('inf')
                    for node in local_set:
                        if node not in weighted_nodes:
                            for i in pa_nom_dict[key].g.nodes(data='true'):
                                if i[0] == node:
                                    temp_energy = i[1]['energy']
                                    break
                            if temp_energy < energy_fin:
                                energy_fin = temp_energy
                                temp_node = node
                    pa_prime.final.add(temp_node)
                else:
                    pa_prime = update_weight(pa_nom_dict[key], weighted_nodes)

                # Get control policy from current location
                ts_policy[key], pa_policy[key], tau_dict[key] = \
                        compute_control_policy2(pa_prime, dfa_dict[key], init_loc) # Look at tau later ***


                # Write updates to file
                write_to_iter_file(ts_policy[key], ts_dict[key], ets_dict[key], key, iter_step)
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
                # Must update policy match
                match_shortest = float('inf')
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
                policy_match = temp_match
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
            policy_match = temp_match
        else:
            running = False

    # Possibly just set the relaxation to the nominal + additional nodes added *** FIX
    for key in pa_nom_dict:
        tau_dict[key] = tau_dict_nom[key] + len(ts_control_policy_dict[key])-len(ts_policy_dict_nom[key])
    # Write the nominal and final control policies to a file
    for key in pa_nom_dict:
        write_to_control_policy_file(ts_policy_dict_nom[key], pa_policy_dict_nom[key], \
                output_dict_nom[key], tau_dict_nom[key], dfa_dict[key],ts_dict[key],ets_dict[key],\
                ts_control_policy_dict[key], pa_control_policy_dict[key], tau_dict[key], key)

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
    # Get rid of current node from local_set
    # local_set.remove(node)
    return local_set

def local_neighborhood(ts):
    ''' Creates a local neighborhood of nodes which the agent can communicate
    to for a more localized communication protocol, this considers actual
    distance as opposed to number of edge hops. '''
    radius = 0.2
    node_set = nx.get_node_attributes(ts.g,"position")
    # distance = []
    blocked_nodes = []
    for key, (u, v) in node_set.items():
        temp = math.sqrt((u-obs_loc[0])**2+(v-obs_loc[1])**2)
        if temp <= radius:
            blocked_nodes.append(key)
    return blocked_nodes

def obstacle_update():
    ''' This is a simple obstacle model, currently always stays at D. Will want
        to expand this to more realistic model in the future. Single integrator
        dynamics to start. '''
    obs_location = (1,0) # this corresponds to node D
    return obs_location

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
    phi2 = '[H^2 r13]^[0, 8] * [H^2 r23]^[0, 7]'
    # Add a third agent
    # phi3 = '[H^2 f]^[0, 8] * [H^3 K]^[0, 10]'
    phi3 = '[H^2 r31]^[0, 8] * [H^3 r10]^[0, 10]'
    # Currently set to use the same transition system
    phi = [phi1, phi2, phi3]
    ts_files = ['../data/ts_synth_6x6_test1.txt', '../data/ts_synth_6x6_test2.txt', '../data/ts_synth_6x6_test3.txt']
    # ts_files = ['../data/ts_synthesis_6x6_obs1.txt', '../data/ts_synthesis_6x6_obs2.txt', '../data/ts_synthesis_6x6_obs3.txt']
    case1_synthesis(phi, ts_files)
