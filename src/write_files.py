'''
.. module:: write_files.py
   :synopsis: Functions to write to csv files.

.. moduleauthor:: Ryan Peterson <pete9936@umn.edu.edu>

'''

import logging, sys
import StringIO
import pdb, os, copy, math
import time
import operator
import csv
import networkx as nx


def write_to_land_file(land_keys):
    ''' Write the agents that have finished to a land file.'''
    with open('../output/agents_land.csv', 'w') as f:
        writer = csv.writer(f)
        for agent in land_keys:
            writer.writerow([agent])
    f.close()

def write_to_csv_iter(ts, ts_write, ids, time_wp):
    ''' Writes the control policy to an output file in CSV format to be used
    as waypoints for a trajectory run by our Crazyflies. '''
    # altitude = 1.0 # meters
    with open('../output/waypoints_dynamic.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['id', 'x[m]', 'y[m]', 'z[m]', 't[s]']
        writer.writerow(header)
        for i in range(len(ts_write)):
            node_set = nx.get_node_attributes(ts[ids[i]].g,"position")
            node = ts_write[i]
            try:
                z = node_set[node][2]
            except IndexError:
                z = 1.0
            writer.writerow([ids[i], node_set[node][0], node_set[node][1], z, time_wp])
    f.close()

def write_to_csv(ts, ts_policy, id, time_wp):
    ''' Writes the control policy to an output file in CSV format to be used
    as waypoints for a trajectory run by our Crazyflies. '''
    altitude = 1.0 # meters
    node_set = nx.get_node_attributes(ts.g,"position")
    if os.path.isfile('../output/waypoints_full.csv'):
        with open('../output/waypoints_full.csv', 'a') as f:
            writer = csv.writer(f)
            for ind, elem in enumerate(ts_policy):
                for node in ts_policy:
                    if elem == node:
                        try:
                            z = node_set[node][2]
                        except IndexError:
                            z = 1.0
                        writer.writerow([id, node_set[node][0], node_set[node][1], z, time_wp*ind])
                        break
    else:
        with open('../output/waypoints_full.csv', 'w') as f:
            writer = csv.writer(f)
            header = ['id', 'x[m]', 'y[m]', 'z[m]', 't[s]']
            writer.writerow(header)
            for ind, elem in enumerate(ts_policy):
                for node in ts_policy:
                    if elem == node:
                        try:
                            z = node_set[node][2]
                        except IndexError:
                            z = 1.0
                        writer.writerow([id, node_set[node][0], node_set[node][1], z, time_wp*ind])
                        break
    f.close()

def write_to_iter_file(policy, ts, ets, key, iter_step):
    ''' Writes each iteration of the control policy to an output file
    to keep track of the changes and updates being made. '''
    policy = [x for x in policy if x not in ets.state_map]
    out = StringIO.StringIO()
    for u, v in zip(policy[:-1], policy[1:]):
        print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
    print>>out, policy[-1],
    logging.info('Generated control policy is: %s', out.getvalue())
    if os.path.isfile('../output/control_policy_updates_S1J.txt'):
        with open('../output/control_policy_updates_S1J.txt', 'a+') as f1:
            f1.write('Control Policy for agent %s at step ' % key)
            f1.write('%s:  ' % iter_step)
            f1.write('%s\n\n' % out.getvalue())
    else:
        with open('../output/control_policy_updates_S1J.txt', 'w+') as f1:
            f1.write('Control Policy for agent %s at step ' % key)
            f1.write('%s:  ' % iter_step)
            f1.write('%s\n\n' % out.getvalue())
    f1.close()
    out.close()

def write_to_control_policy_file(ts_nom_policy, pa_nom_policy, tau, dfa, ts, ets, ts_policy, pa_policy, tau_new, key):
    ''' This writes the nominal and final control policy for each agent to
    an output file. '''
    logging.info('Max deadline: %s', tau)
    if ts_nom_policy is not None:
        # logging.info('Generated output word is: %s', [tuple(o) for o in output])
        policy = [x for x in ts_nom_policy if x not in ets.state_map]
        out = StringIO.StringIO()
        for u, v in zip(policy[:-1], policy[1:]):
            print>>out, u, '->', ts.g[u][v][0]['duration'], '->',
        print>>out, policy[-1],
        logging.info('Generated control policy is: %s', out.getvalue())
        if os.path.isfile('../output/control_policy_S1J.txt'):
            with open('../output/control_policy_S1J.txt', 'a+') as f2:
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
            with open('../output/control_policy_S1J.txt', 'w+') as f2:
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


if __name__ == '__main__':
    pass
