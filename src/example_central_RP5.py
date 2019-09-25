'''
.. module:: example_central_RP5.py
   :synopsis: Case studies for Theoretical Computer Science journal.

.. moduleauthor:: Cristian Ioan Vasile <cvasile@bu.edu>

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
                      verify, compute_control_policy2, compute_control_relaxation
from learning import learn_deadlines
from lomap import Ts
from lomap.algorithms.product import ts_times_ts


def case1_synthesis(formulas, ts_files):

    for ind, f in enumerate(formulas):
        _, dfa_inf, bdd = twtl.translate(f, kind=DFAType.Infinity, norm=True)

    logging.debug('\n\nStart policy computation\n')
    # pdb.set_trace()
    startG = timeit.default_timer()
    r1 = Ts(directed=True, multi=False)
    r2 = Ts(directed=True, multi=False)
    # r3 = Ts(directed=True, multi=False)
    r1.read_from_file(ts_files[0])
    r2.read_from_file(ts_files[1])
    # r3.read_from_file(ts_files[2])
    ets1 = expand_duration_ts(r1)
    ets2 = expand_duration_ts(r2)
    # ets3 = expand_duration_ts(r3)
    ts_tuple = (r1, r2)
    ets_tuple = (ets1, ets2)
    # Construct the team TS
    startTS = timeit.default_timer()

    team_ts = ts_times_ts(ets_tuple)
    print 'Size of team TS before pruning:', team_ts.size()
    # Prune team_ts to avoid duplicate nodes
    for node in team_ts.g.nodes():
        if node[0] == node[1]:
            team_ts.g.remove_node(node)

    stopTS = timeit.default_timer()
    print 'Size of team TS after pruning:', team_ts.size()
    print 'Run Time (s) to get team TS: ', stopTS - startTS

	# Find the optimal run and shortest prefix on team_ts
    # Get the nominal PA
    logging.info('Constructing product automaton with infinity DFA!')
    startPA = timeit.default_timer()
    pa = ts_times_fsa(team_ts, dfa_inf)
    stopPA = timeit.default_timer()
    # Give initial weight attribute to all edges in pa
    nx.set_edge_attributes(pa.g,"weight",1)
    logging.info('Product automaton size is: (%d, %d)', *pa.size())
    print 'Product automaton size is:', pa.size()
    print 'Run Time (s) to get PA for one agent: ', stopPA - startPA
    # pdb.set_trace()

    # Compute the optimal path in PA and project onto the TS
    startPath = timeit.default_timer()
    ts_policy, pa_policy, output, tau = compute_control_policy(pa, dfa_inf, dfa_inf.kind)
    stopPath = timeit.default_timer()
    print 'Run Time (s) to get optimal path for agents is: ', stopPath - startPath # No Chance...

    stopG = timeit.default_timer()
    print 'Run Time (s) for full algorithm: ', stopG - startG
    pdb.set_trace()


def setup_logging():
    fs, dfs = '%(asctime)s %(levelname)s %(message)s', '%m/%d/%Y %I:%M:%S %p'
    loglevel = logging.DEBUG
    logging.basicConfig(filename='../output/examples_RP5.log', level=loglevel,
                        format=fs, datefmt=dfs)
    root = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    ch.setFormatter(logging.Formatter(fs, dfs))
    root.addHandler(ch)

if __name__ == '__main__':
    setup_logging()
    # case study 1: Synthesis
    # phi1 = '([H^2 F]^[0, 7] * [H^2 P]^[0, 7]) & ([H^2 N]^[0, 8] * [H^2 H]^[0, 7])'
    phi1 = '[H^2 F]^[0, 7] * [H^2 P]^[0, 7]'
    # phi1 = '[H^2 V]^[0, 7] * [H^2 M]^[0, 7]'
    # Add another agent with a separate TWTL to coordinate
    phi2 = '[H^2 N]^[0, 8] * [H^2 X]^[0, 7]'
    # Add a third agent ***
    phi3 = '[H^2 f]^[0, 8] * [H^3 K]^[0, 10]'
    # Currently set to use the same transition system
    # phi = [phi1, phi2, phi3]
    phi = [phi1]
    # ts_files = ['../data/ts_synthesis_6x6_obs1.txt', '../data/ts_synthesis_6x6_obs2.txt', '../data/ts_synthesis_6x6_obs3.txt']
    ts_files = ['../data/ts_synthesis_4x4_1.txt', '../data/ts_synthesis_4x4_2.txt']
    case1_synthesis(phi, ts_files)
