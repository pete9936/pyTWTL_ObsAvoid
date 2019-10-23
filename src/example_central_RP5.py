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
                      verify, compute_control_policy2, compute_multiagent_policy
from learning import learn_deadlines
from lomap import Ts
from product import ts_times_ts, pa_times_pa


def case1_synthesis(formulas, ts_files):

    startG = timeit.default_timer()
    dfa_dict = {}
    for ind, f in enumerate(formulas):
        _, dfa_inf, bdd = twtl.translate(f, kind=DFAType.Infinity, norm=True)
        dfa_dict[ind+1] = copy.deepcopy(dfa_inf)

    ts_dict = {}
    ets_dict = {}
    for ind, ts_f in enumerate(ts_files):
        ts_dict[ind+1] = Ts(directed=True, multi=False)
        ts_dict[ind+1].read_from_file(ts_f)
        ets_dict[ind+1] = expand_duration_ts(ts_dict[ind+1])

    # Get the nominal PA
    logging.info('Constructing product automaton with infinity DFA!')
    startPA = timeit.default_timer()
    pa_dict = {}
    for key in dfa_dict:
        logging.info('Constructing product automaton with infinity DFA!')
        pa = ts_times_fsa(ets_dict[key], dfa_dict[key])
        # Give initial weight attribute to all edges in pa
        nx.set_edge_attributes(pa.g,"weight",1)
        logging.info('Product automaton size is: (%d, %d)', *pa.size())
        # Make a copy of the nominal PA to change
        pa_dict[key] = copy.deepcopy(pa)

    # Calculate PA for the entire system
    pa_tuple = (pa_dict[1], pa_dict[2])
    ets_tuple = (ets_dict[1], ets_dict[2])
    team_ts = ts_times_ts(ets_tuple)
    team_pa = pa_times_pa(pa_tuple, team_ts)
    # pdb.set_trace()

    stopPA = timeit.default_timer()
    print 'Product automaton size is:', team_pa.size()
    print 'Run Time (s) to get PA is: ', stopPA - startPA

    startPath = timeit.default_timer()
    # Compute the optimal path in PA and project onto the TS
    pa_policy_multi = compute_multiagent_policy(team_pa)

    stopPath = timeit.default_timer()
    print 'Run Time (s) to get optimal paths for agents is: ', stopPath - startPath
    stopG = timeit.default_timer()
    print 'Run Time (s) for full algorithm: ', stopG - startG
    print 'PA Policy for 2 agents: ', pa_policy_multi


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
    phi1 = '[H^2 F]^[0, 7] * [H^2 P]^[0, 7]'
    # phi1 = '[H^2 V]^[0, 7] * [H^2 M]^[0, 7]'
    # Add another agent with a separate TWTL to coordinate
    phi2 = '[H^2 N]^[0, 8] * [H^2 G]^[0, 7]'
    # phi2 = '[H^2 N]^[0, 8] * [H^2 X]^[0, 7]'
    # Add a third agent ***
    # phi3 = '[H^2 f]^[0, 8] * [H^3 K]^[0, 10]'
    # Currently set to use the same transition system
    phi = [phi1, phi2]
    # ts_files = ['../data/ts_synthesis_6x6_obs1.txt', '../data/ts_synthesis_6x6_obs2.txt']
    ts_files = ['../data/ts_synthesis_4x4_1.txt', '../data/ts_synthesis_4x4_2.txt']
    case1_synthesis(phi, ts_files)
