'''
.. module:: examples_RP1.py
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
                      verify
from learning import learn_deadlines
from lomap import Ts


def case1_synthesis(formula, ts_file):
    _, dfa_inf, bdd = twtl.translate(formula, kind=DFAType.Infinity, norm=True)

    logging.debug('alphabet: {}'.format(dfa_inf.props))

    for u, v, d in dfa_inf.g.edges_iter(data=True):
        logging.debug('({}, {}): {}'.format(u, v, d))

    # dfa_inf.visualize(draw='matplotlib')
    # plt.show()

    logging.debug('\nEnd of translate\n\n')

    logging.info('The bound of formula "%s" is (%d, %d)!', formula, *bdd)
    logging.info('Translated formula "%s" to infinity DFA of size (%d, %d)!',
                 formula, *dfa_inf.size())

    logging.debug('\n\nStart policy computation\n')

    ts = Ts(directed=True, multi=False)
    ts.read_from_file(ts_file)
    ets = expand_duration_ts(ts)


    for name, dfa in [('infinity', dfa_inf)]:
        logging.info('Constructing product automaton with %s DFA!', name)
        pa = ts_times_fsa(ets, dfa)
        # Give initial weight attribute to all edges in pa
        nx.set_edge_attributes(pa.g,"weight",1)
        logging.info('Product automaton size is: (%d, %d)', *pa.size())

        # Print nodes of the Product Automata to a file
        with open('../output/pa_print_nodes_RP1.txt', 'w+') as f1:
            f1.write('\n'.join('%s %s' % x for x in pa.g.nodes()))
        f1.close()

        # Print edges of the Product Automata to a file
        with open('../output/pa_print_edges_RP1.txt', 'w+') as f2:
            f2.write('\n'.join('%s %s %s' % x for x in pa.g.edges(data=True)))
        f2.close()

        if name == 'infinity':
            for u in pa.g.nodes_iter():
                logging.debug('{} -> {}'.format(u, pa.g.neighbors(u)))

            pa.visualize(draw='matplotlib')
            plt.show()

        # Make an updated version of the PA in order to account for an obstacle
        # count = 0
        # control_policy = [] # set empty control policy that will be iteratively updated
        # while count < 10: # need a better way of defining goal has been reached...

        # Make a copy of the nominal PA to change
        pa_prime = copy.deepcopy(pa)
        # set breakpoint
        # pdb.set_trace()
        obs_location = obstacle_update()
        pa_prime = update_weight(ts, pa_prime, obs_location)

        # Compute optimal path in Pa_Prime and project onto the TS
        # Need to account for weight introduced in the pa_prime structure for control policy_2
        policy_2, output_2, tau_2 = compute_control_policy(pa_prime, dfa, dfa.kind)

        # Update control policy
        # This will error out if we go beyond bound, temporary patch made with if statement...
        # if len(policy_2) >= count:
        #    control_policy.append(policy_2[count-1])
        # else:
        #    control_policy.append(policy_2[-1])

        logging.info('Max deadline: %s', tau_2)
        if policy_2 is not None:
            logging.info('Generated output word is: %s', [tuple(o) for o in output_2])

            policy_2 = [x for x in policy_2 if x not in ets.state_map]
            out = StringIO.StringIO()
            for u, v in zip(policy_2[:-1], policy_2[1:]):
                print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
            print>>out, policy_2[-1],
            logging.info('Generated control policy is: %s', out.getvalue())
            # Print control policy to a file
            with open('../output/control_policy_RP1.txt', 'w+') as f3:
                f3.write('Control Policy with obstacle present.\n')
                f3.write('Generated control policy is: %s \n\n' % out.getvalue())
                # f3.write('Relaxation is: %s \n\n' % twtl.temporal_relaxation(output_2, formula=formula))
            f3.close()
            out.close()

            logging.info('Relaxation is: %s',
                         twtl.temporal_relaxation(output_2, formula=formula))
        else:
            logging.info('No control policy found!')


        # Compute optimal path in PA and then project onto the TS
        # This is the nominal control policy
        policy, output, tau = compute_control_policy(pa, dfa, dfa.kind)

        logging.info('Max deadline: %s', tau)
        if policy is not None:
            logging.info('Generated output word is: %s', [tuple(o) for o in output])

            policy = [x for x in policy if x not in ets.state_map]
            out = StringIO.StringIO()
            for u, v in zip(policy[:-1], policy[1:]):
                print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
            print>>out, policy[-1],
            logging.info('Generated control policy is: %s', out.getvalue())
            # Print control policy to a file
            with open('../output/control_policy_RP1.txt', 'a+') as f3:
                f3.write('Nominal Control Policy.\n')
                f3.write('Generated control policy is: %s \n\n' % out.getvalue())
            #    f3.write('Relaxation is: %s \n\n', twtl.temporal_relaxation(output, formula=formula))
            f3.close()
            out.close()

            logging.info('Relaxation is: %s',
                         twtl.temporal_relaxation(output, formula=formula))
        else:
            logging.info('No control policy found!')

def obstacle_update():
    ''' This is a simple obstacle model, currently always stays at D. Will want
        to expand this to more realistic model in the future. Single integrator
        dynamics to start. '''
    obs_location = (1,0) # this corresponds to node D
    return obs_location

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

def setup_logging():
    fs, dfs = '%(asctime)s %(levelname)s %(message)s', '%m/%d/%Y %I:%M:%S %p'
    loglevel = logging.DEBUG
    logging.basicConfig(filename='../output/examples_RP1_scrap.log', level=loglevel,
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
    phi = '[H^1 A]^[0, 3] * [H^1 Base2]^[0, 9]'
    case1_synthesis(phi, '../data/ts_synthesis_RP1.txt')
