'''
.. module:: examples_RP2.py
   :synopsis: Case studies for Theoretical Computer Science journal.

.. moduleauthor:: Cristian Ioan Vasile <cvasile@bu.edu>

'''

import logging, sys
import StringIO
import pdb, os, copy, math

import networkx as nx
import matplotlib.pyplot as plt

import twtl
from dfa import DFAType
from synthesis import expand_duration_ts, compute_control_policy, ts_times_fsa,\
                      verify, compute_control_policy2, compute_control_relaxation
from learning import learn_deadlines
from lomap import Ts


def case1_synthesis(formulas, ts_files):
    dfa_dict = {}
    for ind, f in enumerate(formulas):
        _, dfa_inf, bdd = twtl.translate(f, kind=DFAType.Infinity, norm=True)

        logging.debug('alphabet: {}'.format(dfa_inf.props))

        for u, v, d in dfa_inf.g.edges_iter(data=True):
            logging.debug('({}, {}): {}'.format(u, v, d))
        # dfa_inf.visualize(draw='matplotlib')
        # plt.show()
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

    # Get the nominal PA for each agent
    pa_nom_dict = {}
    for key in dfa_dict:
        logging.info('Constructing product automaton with infinity DFA!')
        pa = ts_times_fsa(ets_dict[key], dfa_dict[key])
        # Give initial weight attribute to all edges in pa
        nx.set_edge_attributes(pa.g,"weight",1)
        logging.info('Product automaton size is: (%d, %d)', *pa.size())

        for u in pa.g.nodes_iter():
            logging.debug('{} -> {}'.format(u, pa.g.neighbors(u)))
        # pa.visualize(draw='matplotlib')
        # plt.show()
        # Make a copy of the nominal PA to change
        pa_nom_dict[key] = copy.deepcopy(pa)

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

    # set empty control policies that will be iteratively updated
    ts_control_policy_dict = {}
    pa_control_policy_dict = {}

    for key in ts_policy_dict_nom:
        if key == 1:
            # Agent 1 is nominal since it is given highest priority
            ts_control_policy_dict[1] = ts_policy_dict_nom.get(1)
            pa_control_policy_dict[1] = list(pa_policy_dict_nom[1])
        else:
            ts_control_policy_dict[key] = []
            pa_control_policy_dict[key] = []
            # Concatenate nominal policies for searching
            ts_shortest = float('inf')
            for init_key in ts_policy_dict_nom:
                if init_key >= key:
                    break
                else:
                    if len(ts_policy_dict_nom[init_key]) < ts_shortest:
                        ts_shortest = len(ts_policy_dict_nom[init_key])
            temp_match = [[] for i in range(ts_shortest)]
            # match up all policies into a list of lists
            for init_key in ts_policy_dict_nom:
                if init_key >= key:
                    break
                else:
                    for ind, item in enumerate(ts_policy_dict_nom[init_key]):
                        if ind >= ts_shortest:
                            break
                        else:
                            temp_match[ind].append(item)
            # Now account for the current policy
            ts_temp = copy.deepcopy(ts_policy_dict_nom[key])
            for ind, item in enumerate(ts_temp):
                if ind >= ts_shortest:
                    break
                else:
                    temp_match[ind].append(item)
            # Create the policy match list of tuples
            policy_match = temp_match
            # Initialize vars
            traj_length = 0
            iter_step = 0
            ts_policy = copy.deepcopy(ts_policy_dict_nom[key])
            pa_policy = list(copy.deepcopy(pa_policy_dict_nom[key]))
            while policy_match:
                for policy in policy_match:
                    policy_match.pop(0)
                    append_flag = True
                    for item in policy[:-1]:
                        if item != policy[-1]:
                            append_flag = True
                        else:
                            append_flag = False
                            break
                    if append_flag == True:
                            ts_control_policy_dict[key].append(ts_policy.pop(0))
                            pa_control_policy_dict[key].append(pa_policy.pop(0))
                            traj_length += 1
                            break
                    else: # append_flag == False
                        iter_step += 1
                        # Update PA with new weights, node to change is u
                        pa_prime = update_weight(pa_nom_dict[key], policy[:-1])
                        # Now recompute the control policy with updated edge weights
                        init_loc = pa_control_policy_dict[key][-1]
                        # Get control policy from current location
                        ts_policy, pa_policy = \
                                compute_control_policy2(pa_prime, dfa_dict[key], init_loc)
                        pa_policy = list(pa_policy)
                        # Write updates to file
                        write_to_iter_file(ts_policy, ts_dict[key], ets_dict[key], key, iter_step)
                        # Get rid of the duplicate node
                        ts_policy.pop(0)
                        pa_policy.pop(0)
                        # Must update policy match
                        match_shortest = float('inf')
                        for match_key in ts_policy_dict_nom:
                            if match_key >= key:
                                break
                            else:
                                if len(ts_control_policy_dict[match_key]) > traj_length:
                                    if len(ts_control_policy_dict[match_key][traj_length:]) < ts_shortest:
                                        ts_shortest = len(ts_control_policy_dict[match_key][traj_length:])
                        temp_match = [[] for i in range(ts_shortest)]
                        # Add all previous control policies to temp_match
                        for match_key in ts_control_policy_dict:
                            if match_key >= key:
                                break
                            else:
                                if len(ts_control_policy_dict[match_key]) > traj_length:
                                    for ind, item in enumerate(ts_control_policy_dict[match_key][traj_length:]):
                                        if ind >= ts_shortest:
                                            break
                                        else:
                                            temp_match[ind].append(item)
                        # Now account for the current policy
                        for ind, item in enumerate(ts_policy):
                            if ind >= ts_shortest:
                                break
                            else:
                                temp_match[ind].append(item)
                        # Set policy_match
                        policy_match = temp_match
                        break

            # Finalize the control policies due to possibly being the longest policy
            while ts_policy:
                ts_control_policy_dict[key].append(ts_policy.pop(0))
                pa_control_policy_dict[key].append(pa_policy.pop(0))

    # Write the nominal and final control policies to a file
    for key in pa_nom_dict:
        write_to_control_policy_file(ts_policy_dict_nom[key], pa_policy_dict_nom[key], \
                output_dict_nom[key], tau_dict_nom[key], dfa_dict[key],ts_dict[key],ets_dict[key],\
                ts_control_policy_dict[key], pa_control_policy_dict[key], key)


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
    if os.path.isfile('../output/control_policy_updates_RP3.txt'):
        with open('../output/control_policy_updates_RP3.txt', 'a+') as f1:
            f1.write('Control Policy for agent %s at step ' % key)
            f1.write('%s:  ' % iter_step)
            f1.write('%s\n\n' % out.getvalue())
    else:
        with open('../output/control_policy_updates_RP3.txt', 'w+') as f1:
            f1.write('Control Policy for agent %s at step ' % key)
            f1.write('%s:  ' % iter_step)
            f1.write('%s\n\n' % out.getvalue())
    f1.close()
    out.close()

def write_to_control_policy_file(ts_nom_policy, pa_nom_policy, output, tau, dfa, ts, ets, ts_policy, pa_policy, key):
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
        if os.path.isfile('../output/control_policy_RP3.txt'):
            with open('../output/control_policy_RP3.txt', 'a+') as f2:
                f2.write('Nominal Control Policy for agent %s.\n' % key)
                f2.write('Generated PA control policy is: (')
                f2.write(') -> ('.join('%s %s' % x for x in pa_nom_policy))
                f2.write(') \nGenerated TS control policy is: %s \n\n' % ts_nom_policy)
                f2.write('Final Control policy for agent %s.\n' % key)
                f2.write('Generated PA control policy is: (')
                f2.write(') -> ('.join('%s %s' % x for x in pa_policy))
                f2.write(') \nGenerated TS control policy is:  %s \n\n' % ts_policy)
        else:
            with open('../output/control_policy_RP3.txt', 'w+') as f2:
                f2.write('Nominal Control Policy for agent %s.\n' % key)
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
    logging.basicConfig(filename='../output/examples_RP3.log', level=loglevel,
                        format=fs, datefmt=dfs)
    root = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    ch.setFormatter(logging.Formatter(fs, dfs))
    root.addHandler(ch)

if __name__ == '__main__':
    setup_logging()
    # case study 1: Synthesis
    # phi = '[H^2 A]^[0, 6] * ([H^1 B]^[0, 3] | [H^1 C]^[1, 4]) * [H^1 D]^[0, 6]'
    # phi1 = '[H^2 A]^[0, 3] * [H^2 G]^[0, 7]'
    phi1 = '[H^2 V]^[0, 7] * [H^2 M]^[0, 7]'
    # Add another agent with a separate TWTL to coordinate
    # phi2 = '[H^2 G]^[0, 3] * [H^2 A]^[0, 7]'
    phi2 = '[H^2 N]^[0, 8] * [H^2 X]^[0, 7]'
    # Add a third agent ***
    # phi3 = '[H^2 B]^[0, 4] * [H^4 E]^[0, 8]'
    phi3 = '[H^2 f]^[0, 8] * [H^3 K]^[0, 10]'
    # Currently set to use the same transition system
    phi = [phi1, phi2, phi3]
    # ts_files = ['../data/ts_synthesis_RP2_1.txt', '../data/ts_synthesis_RP2_2.txt', '../data/ts_synthesis_RP2_3.txt']
    ts_files = ['../data/ts_synthesis_6x6_alph1.txt', '../data/ts_synthesis_6x6_alph2.txt', '../data/ts_synthesis_6x6_alph3.txt']
    case1_synthesis(phi, ts_files)
