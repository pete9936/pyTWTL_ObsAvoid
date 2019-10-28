'''
alt_funcs.py

This file will store alternatives to current function implementations
we may want to pursue in the future.

'''

import logging, sys
import StringIO
import pdb, os, copy, math

import networkx as nx
import matplotlib.pyplot as plt

import twtl
from dfa import DFAType
from synthesis import expand_duration_ts, compute_control_policy, ts_times_fsa,\
                      verify
from learning import learn_deadlines
from lomap import Ts


def update_weight(ts, pa_prime, obs_loc):
    ''' Update edge weights of PA based on a potential field function if
        obstacles are detected. Compute this based on the position of then
        nodes w.r.t the obstacle given as 'position' attribute in the TS '''
    radius = 0.2
    # set breakpoint
    # pdb.set_trace()
    node_set = nx.get_node_attributes(ts.g,"position")
    # distance = []
    blocked_nodes = []
    for key, (u, v) in node_set.items():
        temp = math.sqrt((u-obs_loc[0])**2+(v-obs_loc[1])**2)
        if temp <= radius:
            blocked_nodes.append(key)

    # This searches the edges and if an edge is connected to the obstacle
    # node then we assign updated weight
    for s_token in blocked_nodes:
        for i in pa_prime.g.edges():
            for item in i:
                if s_token in item:
                    weight_new = 10
                    temp = list(i)
                    temp.append(weight_new)
                    pa_prime.g.add_weighted_edges_from([tuple(temp)])
                    break

    # Need to implement a method which goes beyond the first ring, essentially
    # updating the weights of edges based on the distance function *****
    # C_k = 10 # a scalar parameter related to the radius of obstacle update
    # neighborhood = 0.6 # defines influence region of obstacle upd
    # for key, (u, v) in node_set.items():
    #   distance = math.sqrt((u-obs_loc[0])**2+(v-obs_loc[1])**2)
    #   if distance <= neighborhood:
    #       weight_new = C_k*math.exp(-distance/(2*sig**2))
    #       for i in pa_prime.g.edges():
    #           for item in i:
    #               if key in item and weight > 1:
    #                   temp = list(i)
    #                   temp.append(weight_new)
    #                   pa_prime.g.add_weighted_edges_from([tuple(temp)])
    #                   break
    return pa_prime

def file_writer_stuff(policy, output, ts, ets):
''' mostly just scrap info for file writing commands '''
    for policy, output, ts, ets in [(control_policy_1, output_1, ts_1, ets_1),\
                                    (control_policy_2, output_2, ts_2, ets_2)]:
            logging.info('Generated output word is: %s', [tuple(o) for o in output])
            policy = [x for x in policy if x not in ets.state_map]
            out = StringIO.StringIO()
            for u, v in zip(policy[:-1], policy[1:]):
                print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
            print>>out, policy[-1],
            logging.info('Generated control policy is: %s', out.getvalue())
            # Print control policy to a file
            if os.path.isfile('../output/control_policy_updates_RP2.txt'):
                with open('../output/control_policy_updates_RP2.txt', 'a+') as f1:
                    f1.write('Control Policy at step %s:' % count)
                    f1.write('  %s \n\n' % out.getvalue())
            else:
                with open('../output/control_policy_updates_RP2.txt', 'w+') as f1:
                    f1.write('Control Policy at step %s:' % count)
                    f1.write('  %s \n\n' % out.getvalue())
            f1.close()
            out.close()

def relaxed_control_policy2(tree, dfa, pa, init_key=None, init_set=None, constraint=None):
    '''Computes a control policy with minimum maximum temporal relaxation. It
    also returns the value of the optimal relaxation. This accounts for starting
    at a location other than the initial. Main issue to overcome is the tree.init
    index which changes through the iteration. Maybe use a counter... *** 9/14
    '''
    assert tree.wdf

    if init_set is None:
        init_set = tree.init

    if tree.unr: # primitive/unrelaxable formula
        # Find out how to update the tree.init element
        paths = partial_control_policies(pa, dfa, init_set, tree.final, constraint)
        return ControlPathsSet([ControlPath(p) for p in paths])

    if tree.wwf and tree.operation == Op.event: # leaf within operator
        paths = partial_control_policies(pa, dfa, init_set, tree.final, constraint)
        return ControlPathsSet([ControlPath(path, len(path) - tree.high - 1)
                                    for path in paths])

    if not tree.wwf and tree.operation == Op.event:
        M_ch = relaxed_control_policy2(tree.left, dfa, pa, init_key, init_set, constraint)
        if tree.low == 0:
            for cpath in M_ch:
                cpath.tau = max(len(cpath.path) - tree.high - 1, path.tau)
            return M_ch

        M = ControlPathsSet()
        for cp in M_ch:
            paths = nx.shortest_path(pa.g, source=init_key, target=cp.path[0], weight='weight')
            sat_paths = [p[:-1]+cp.path for p_i, p in paths.iteritems()
                                         if p_i in init_set]
            tau = max(len(cp.path)+tree.low-tree.high, cp.tau) #TODO: should I subtract -1?
            M.paths.extend([ControlPath(p, tau) for p in sat_paths])
        return M

    if tree.operation == Op.cat:
        M_left = relaxed_control_policy2(tree.left, dfa, pa, init_key, init_set)
        M_right = relaxed_control_policy2(tree.right, dfa, pa, init_key, init_set, constraint)
        # concatenate paths from M_left with paths from M_rigth
        M = M_left + M_right
        return M

    if tree.operation == Op.intersection:
        M_left = relaxed_control_policy2(tree.left, dfa, pa, init_key, init_set, constraint)
        M_right = relaxed_control_policy2(tree.right, dfa, pa, init_key, init_set, constraint)
        # intersection of M_left and M_rigtht
        M = M_left & M_right
        return M

    if tree.operation == Op.union:
        if constraint is None:
            c_left = {s: ch.both | ch.left for s, ch in tree.choices.iteritems()}
            c_right = {s: ch.both | ch.right for s, ch in tree.choices.iteritems()}
        else:
            c_left = dict()
            c_right = dict()
            for s in tree.choices.viewkeys() & constraint.viewkeys():
                c_left[s] = constraint[s] & (tree.choices[s].both | tree.choices[s].left)
                c_right[s] = constraint[s] & (tree.choices[s].both | tree.choices[s].right)

        M_left = relaxed_control_policy2(tree.left, dfa, pa, init_key, init_set, c_left)
        M_right = relaxed_control_policy2(tree.right, dfa, pa, init_key, init_set, c_right)
        # union of M_left and M_rigtht
        M = M_left | M_right
        return M

    raise ValueError('Unknown operation: {}!'.format(tree.operation))


def compute_control_policy2(pa, dfa, init_loc=None):
    ''' Computes a control policy from product automaton pa. This takes into
    account the updated initial state where collision was detected. Will update
    this in the future for computational efficiency. *** '''
    if init_loc is not None:
        init_set = {init_loc[1]}
        init_key = init_loc
        pa_init_keys = [init_loc]
        optimal_ts_path = simple_control_policy2(pa)
        optimal_tau = None
    else:
        init_set = None
        init_key = pa.init.keys()[0]
        pa_init_keys = pa.init.keys()
        # Find the policies given the initial state
        pdb.set_trace()
        policies = relaxed_control_policy2(dfa.tree, dfa, pa, init_key, init_set)
        if not policies:
            return None, None, None, None
        # keep only policies which start from the initial PA state
        policies.paths = [p for p in policies if p.path[0] in pa_init_keys]
        # choose optimal policy with respect to temporal robustness
        optimal_pa_path = min(policies, key=attrgetter('tau'))
        optimal_ts_path = [x for x, _ in optimal_pa_path.path]
        optimal_tau = optimal_pa_path.tau
    if optimal_ts_path is None:
        return None, None, None, None
    output_word = policy_output_word(optimal_ts_path, set(dfa.props.keys()))
    return optimal_ts_path, optimal_pa_path, output_word, optimal_tau

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
