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
                      verify, compute_control_policy2
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
    pa_prime_dict = {}
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
        pa_prime_dict[key] = copy.deepcopy(pa)

    # Compute optimal path in Pa_Prime and project onto the TS, initial policy
    ts_policy_dict_nom = {}
    pa_policy_dict_nom = {}
    output_dict_nom = {}
    tau_dict_nom = {}
    for key in pa_prime_dict:
        ts_policy_dict_nom[key], pa_policy_dict_nom[key], output_dict_nom[key], tau_dict_nom[key] = \
                    compute_control_policy(pa_nom_dict[key], dfa_dict[key], dfa_dict[key].kind)
    pa_policy_dict = copy.deepcopy(pa_policy_dict_nom)
    ts_policy_dict = copy.deepcopy(ts_policy_dict_nom)

    # Perform initial check on nominal control policies
    for key in ts_policy_dict_nom:
        if ts_policy_dict_nom[key] is None:
            logging.info('No control policy found!')

    # set empty control policies that will be iteratively updated
    ts_control_policy_dict = {}
    pa_control_policy_dict = {}

    for key in ts_policy_dict:
        if key == 1:
            # Agent 1 is nominal since it is given highest priority
            ts_control_policy_dict[1] = ts_policy_dict_nom.get(1)
            pa_control_policy_dict[1] = list(pa_policy_dict_nom[1])
        else:
            ts_control_policy_dict[key] = []
            pa_control_policy_dict[key] = []
            # Concatenate policies for searching
            policy_match_nom = create_policy_match(ts_policy_dict_nom, ts_control_policy_dict, key)
            policy_match = create_policy_match(ts_policy_dict, ts_control_policy_dict, key)
            policy_nom_length = len(policy_match_nom)
            traj_length = 0
            iter_step = 0
            # Need to fix iterators!!!
            # We don't really give a fuck about policy_nom_length, change this!!! ***
            while traj_length < policy_nom_length: # while finish is not True:
                if key == 2:
                    pa_ind = 0
                    for u, v in policy_match[traj_length:]:
                        if u != v:
                            ts_control_policy_dict[key].append(v)
                            pa_control_policy_dict[key].append(pa_policy_dict[key][pa_ind])
                            pa_ind += 1
                            traj_length += 1
                            break
                        else:
                            # update recompute steps for resolution
                            iter_step += 1
                            # Update PA with new weights, node to change is u
                            pa_prime_dict[key] = update_weight(pa_nom_dict[key], [u])
                            # Now recompute the control policy with updated edge weights
                            init_loc = pa_control_policy_dict[key][traj_length-1]
                            # Get control policy from current location
                            ts_policy_dict[key], pa_policy_dict[key] = \
                                    compute_control_policy2(pa_prime_dict[key], dfa_dict[key], init_loc)
                            # Write updates to file
                            write_to_iter_file(ts_policy_dict[key], ts_dict[key], ets_dict[key], key, iter_step)
                            # Must update policy match
                            policy_match = create_policy_match(ts_policy_dict, ts_control_policy_dict, key)
                            policy_m_length = len(policy_match)
                            if policy_m_length < policy_nom_length:
                                for i in policy_match_nom[policy_m_length:]:
                                    policy_match.append(policy_match_nom[i])
                            break
                elif key == 3:
                    pa_ind = 0
                    for u, v, w in policy_match[traj_length:]:
                        if w != (u or v):
                            ts_control_policy_dict[key].append(w)
                            pa_control_policy_dict[key].append(pa_policy_dict[key][pa_ind])
                            pa_ind += 1
                            traj_length += 1
                        else:
                            # update recompute steps for resolution
                            iter_step += 1
                            # Update PA with new weights, node to change is u
                            pa_prime_dict[key] = update_weight(pa_nom_dict[key], [u, v])
                            # Now recompute the control policy with updated edge weights
                            init_loc = pa_control_policy_dict[key][traj_length-1]
                            # Get control policy from current location
                            ts_policy_dict[key], pa_policy_dict[key] = \
                                    compute_control_policy2(pa_prime_dict[key], dfa_dict[key], init_loc)
                            # Write updates to file
                            write_to_iter_file(ts_policy_dict[key], ts_dict[key], ets_dict[key], key, iter_step)
                            # Must update policy match
                            policy_match = create_policy_match(ts_policy_dict, ts_control_policy_dict, key)
                            policy_length = len(policy_match)
                            if policy_length < policy_nom_length:
                                for i in policy_match_nom[policy_length:]:
                                    policy_match.append(i)
                else:
                    logging.info('More than 3 robots, not accounted for in code (yet..)!')

            # Finalize the control policies due to varied lengths
            if key == 2:
                if ts_policy_dict[key] is None:
                    logging.info('No control policy found!')
                else:
                    if len(ts_policy_dict[key]) > traj_length:
                        for i in ts_policy_dict[key][traj_length:]:
                            ts_control_policy_dict[key].append(i)
            elif key == 3:
                if ts_policy_dict[key] is None:
                    logging.info('No control policy found!')
                else:
                    if len(ts_policy_dict[key]) <= traj_length:
                        # This is the shortest trajectory, therefore we are done
                        pass
                    else:
                        # Now we perform updates on the last bit of trajectory
                        policy_match = create_policy_match_fin(ts_policy_dict, ts_control_policy_dict, key)
                        while traj_length < policy_nom_length:
                            pa_ind = 0
                            for u, v in policy_match[traj_length:]:
                                if u != v:
                                    ts_control_policy_dict[key].append(w)
                                    pa_control_policy_dict[key].append(pa_policy_dict[key][pa_ind])
                                    pa_ind += 1
                                    traj_length += 1
                                else:
                                    # update recompute steps for resolution
                                    iter_step += 1
                                    # Update PA with new weights, node to change is u
                                    pa_prime_dict[key] = update_weight(pa_prime_dict.get(key), [u, v])
                                    # Now recompute the control policy with updated edge weights
                                    init_loc = pa_control_policy_dict[key][traj_length-1]
                                    # Get control policy from current location
                                    ts_policy_dict[key], pa_policy_dict[key] = \
                                                 compute_control_policy2(pa_prime_dict[key], dfa_dict[key], init_loc)
                                    # Write updates to file
                                    write_to_iter_file(ts_policy_dict[key], ts_dict[key], ets_dict[key], key, iter_step)
                                    # Must update policy match
                                    policy_match = create_policy_match_fin(ts_policy_dict, ts_control_policy_dict, key)
                                    policy_m_length = len(policy_match)
                                    if policy_length < policy_nom_length:
                                        for i in policy_match_nom[policy_m_length:]:
                                            policy_match.append(i)
                        # Now if agent 3 control policy is the longest then we need to append
                        if len(ts_policy_dict[key]) > traj_length:
                            for i in policy_dict[key][traj_length:]:
                                ts_control_policy_dict[key].append(i)
            else:
                 logging.info('This will have crashed by this point...')

    # Compute optimal path in PA and then project onto the TS
    # Write the nominal and final control policies to a file
    pdb.set_trace()
    for key in pa_nom_dict:
        write_to_control_policy_file(pa_nom_dict[key],dfa_dict[key],ts_dict[key],ets_dict[key],ts_control_policy_dict[key],key)



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

def create_policy_match(ts_policy_dict, ts_control_policy_dict, key):
    ''' Will handle the policy_match tuple creation while
    iterating through the control policies due to agent priorities '''
    if key == 2:
        policy_match = zip(ts_control_policy_dict.get(1), ts_policy_dict.get(2))
    elif key == 3:
        policy_match = zip(ts_control_policy_dict.get(1), ts_control_policy_dict.get(2), ts_policy_dict.get(3))
    else:
        logging.info('More than 3 robots not accounted for in code (yet..)!')
    return policy_match

def create_policy_match_fin(ts_policy_dict, ts_control_policy_dict, key):
    ''' Handles the policy_match tuple creation after the main iteration,
    in this case only applies to the 3 agent case. '''
    if len(ts_control_policy_dict.get(1)) > len(ts_control_policy_dict.get(2)):
        policy_match = zip(ts_control_policy_dict.get(1), ts_policy_dict.get(key))
    else:
        policy_match = zip(ts_control_policy_dict.get(2), ts_policy_dict.get(key))
    return policy_match

def write_to_iter_file(policy, ts, ets, key, iter_step):
    ''' Writes each iteration of the control policy to an output file
    to keep track of the changes and updates being made. '''
    policy = [x for x in policy if x not in ets.state_map]
    out = StringIO.StringIO()
    for u, v in zip(policy[:-1], policy[1:]):
        print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
    print>>out, policy[-1],
    logging.info('Generated control policy is: %s', out.getvalue())
    if os.path.isfile('../output/control_policy_updates_RP2.txt'):
        with open('../output/control_policy_updates_RP2.txt', 'a+') as f1:
            f1.write('Control Policy for agent %s at step ' % key)
            f1.write('%s:  ' % iter_step)
            f1.write('%s\n\n' % out.getvalue())
    else:
        with open('../output/control_policy_updates_RP2.txt', 'w+') as f1:
            f1.write('Control Policy for agent %s at step ' % key)
            f1.write('%s:  ' % iter_step)
            f1.write('%s\n\n' % out.getvalue())
    f1.close()
    out.close()

def write_to_control_policy_file(pa_nom, dfa, ts, ets, control_policy, key):
    ''' This writes the nominal and final control policy for each agent to
    an output file. '''
    ts_policy, pa_policy, output, tau = compute_control_policy(pa_nom, dfa, dfa.kind)
    logging.info('Max deadline: %s', tau)
    if ts_policy is not None:
        logging.info('Generated output word is: %s', [tuple(o) for o in output])
        policy = [x for x in ts_policy if x not in ets.state_map]
        out = StringIO.StringIO()
        for u, v in zip(policy[:-1], policy[1:]):
            print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
        print>>out, policy[-1],
        logging.info('Generated control policy is: %s', out.getvalue())
        if os.path.isfile('../output/control_policy_RP2.txt'):
            with open('../output/control_policy_RP2.txt', 'a+') as f2:
                f2.write('Nominal Control Policy for agent %s.\n' % key)
                f2.write('Generated control policy is: %s \n\n' % out.getvalue())
                f2.write('Final Control policy for agent %s.\n' % key)
                f2.write('Generated control policy is:  %s \n\n' % control_policy)
        else:
            with open('../output/control_policy_RP2.txt', 'w+') as f2:
                f2.write('Nominal Control Policy for agent %s.\n' % key)
                f2.write('Generated control policy is: %s \n\n' % out.getvalue())
                f2.write('Final Control policy for agent %s.\n' % key)
                f2.write('Generated control policy is:  %s \n\n' % control_policy)
        f2.close()
        out.close()
    else:
        logging.info('No control policy found!')

def setup_logging():
    fs, dfs = '%(asctime)s %(levelname)s %(message)s', '%m/%d/%Y %I:%M:%S %p'
    loglevel = logging.DEBUG
    logging.basicConfig(filename='../output/examples_RP2.log', level=loglevel,
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
    phi1 = '[H^2 A]^[0, 3] * [H^2 G]^[0, 7]'
    # Add another agent with a separate TWTL to coordinate
    phi2 = '[H^2 G]^[0, 3] * [H^2 A]^[0, 7]'
    # Add a third agent ***
    phi3 = '[H^2 C]^[0, 3] * [H^2 F]^[0, 7]'
    # Currently set to use the same transition system
    phi = [phi1, phi2]
    ts_files = ['../data/ts_synthesis_RP2_1.txt', '../data/ts_synthesis_RP2_2.txt']
    case1_synthesis(phi, ts_files)
