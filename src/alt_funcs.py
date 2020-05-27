'''
alt_funcs.py

This file will store old functions we may want to reference (set of notes).

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
    return pa_prime

def deadlock_priority(priority, D_flags, cur_priority):
    ''' Updates the priority ordering if a deadlock occurs. The highest
    priority remains, but the remaining order is altered. '''
    new_priority = []
    # Need the number of D_flags raised to avoid oscillations and proper reprioritizing
    num_deadlocks = 0
    for key in D_flags:
        if D_flags[key] == True:
            num_deadlocks += 1
    # Append highest priority and/or any previous deadlock agents already detected
    for i in range(num_deadlocks):
        new_priority.append(priority[i])
    # Now include current deadlock priority
    new_priority.append(cur_priority)
    # Add all remaining agents in standard priority ordering
    for p in priority[1::]:
        if D_flags[p] == False:
            new_priority.append(p)
    return new_priority


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

def two_hop_horizon(pa, weighted_nodes, soft_nodes, init_loc):
    ''' Compute the two hop local horizon when imminenet collision detected
    or still within the local neighborhood. This one uses my method for
    energy update of +1'''
    # if ts_policy[key][i+1] in local_set and ts_policy[key][i+1] not in soft_nodes:
        # soft_nodes.append(ts_policy[key][i+1]) # for use when defining soft_nodes
    ts_policy = []
    pa_policy = []
    # Create local one-hop set and remove current node
    local_set = pa.g.neighbors(init_loc)
    # Use the energy function to get the first hop
    energy_low = float('inf')
    for neighbor in local_set:
        for node in pa.g.nodes(data='true'):
            if neighbor == node[0] and node[0][0] not in weighted_nodes:
                if node[0][0] in soft_nodes:
                    energy_temp = node[1]['energy'] + 1
                else:
                    energy_temp = node[1]['energy']
                if energy_temp < energy_low:
                    energy_low = energy_temp
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
                if node[0][0] in soft_nodes:
                    energy_temp = node[1]['energy'] + 1
                else:
                    energy_temp = node[1]['energy']
                if energy_temp < energy_low:
                    energy_low = energy_temp
                    two_hop_node = node[0]
                    break
    if energy_low == float('inf'):
        two_hop_node = one_hop_node
        print 'No feasible location to move, therefore stay in current position'
    # Append policies returned
    ts_policy.append(two_hop_node[0])
    pa_policy.append(two_hop_node)
    return ts_policy, pa_policy

def extend_horizon(pa, weighted_nodes, soft_nodes, pa_node):
    ''' This extends the receding horizon trajectory when immediate conflict
    not seen but still in the local neighborhood of another agent. This one uses
    my method for energy update of +1'''
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
                if node[0][0] in soft_nodes:
                    energy_temp = node[1]['energy'] + 1
                else:
                    energy_temp = node[1]['energy']
                if energy_temp < energy_low:
                    energy_low = energy_temp
                    next_node = node[0]
                    break
    if energy_low == float('inf'):
        ignore_flag = True
        next_node = pa_node
        print 'No feasible location to move, therefore stay in current position'
    if energy_low < 1:
        ignore_flag = True
    ts_policy.append(next_node[0])
    pa_policy.append(next_node)
    return ts_policy, pa_policy, ignore_flag

def check_cross(cur_ind, ts, ts_prev_states, ts_next_states, priority, key_list, disc, disc_z):
    ''' Check if moving in diagonal directions causes any conflict. Look into only performing
    this if local neighborhood is true, and only with the agents in the local neighborhood.
    Also consider adding additional weight to the node directly below (downwash).'''
    weighted_nodes = []
    node_set = nx.get_node_attributes(ts.g,"position")
    cur_prev_pose = node_set[ts_prev_states[cur_ind]]
    cur_next_pose = node_set[ts_next_states[cur_ind]]
    x_cur_prev = cur_prev_pose[0]
    x_cur_next = cur_next_pose[0]
    x_cur_dist = cur_prev_pose[0] - cur_next_pose[0]
    y_cur_prev = cur_prev_pose[1]
    y_cur_next = cur_next_pose[1]
    y_cur_dist = cur_prev_pose[1] - cur_next_pose[1]
    if disc_z > 0:
        z_cur_prev = cur_prev_pose[2]
        z_cur_next = cur_next_pose[2]
        z_cur_dist = cur_prev_pose[2] - cur_next_pose[2]
    # run through higher priority set
    for p_ind, p_val in enumerate(priority):
        for k, key in enumerate(key_list):
            if p_val == key:
                comp_prev_pose = node_set[ts_prev_states[k]]
                comp_next_pose = node_set[ts_next_states[k]]
                x_comp_prev = comp_prev_pose[0]
                x_comp_next = comp_next_pose[0]
                x_comp_dist = comp_prev_pose[0] - comp_next_pose[0]
                y_comp_prev = comp_prev_pose[1]
                y_comp_next = comp_next_pose[1]
                y_comp_dist = comp_prev_pose[1] - comp_next_pose[1]
                if disc_z > 0:
                    z_comp_prev = comp_prev_pose[2]
                    z_comp_next = comp_next_pose[2]
                    z_comp_dist = comp_prev_pose[2] - comp_next_pose[2]
                # Now perform the checks
                if disc_z == 0:
                    if (x_cur_dist == disc and y_cur_dist == disc and ((x_comp_dist == -1*disc and y_comp_dist == disc \
                                   and x_cur_prev == x_comp_next and y_cur_prev == y_comp_prev) or (x_comp_dist == disc \
                                   and y_comp_dist == -1*disc and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == -1*disc and ((x_comp_dist == -1*disc and y_comp_dist == disc \
                                   and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next) or (x_comp_dist == disc \
                                   and y_comp_dist == -1*disc and y_cur_prev == y_comp_prev and x_cur_prev == x_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == disc and ((x_comp_dist == disc and y_comp_dist == disc \
                                   and x_cur_prev == x_comp_next and y_cur_prev == y_comp_prev) or (x_comp_dist == -1*disc \
                                   and y_comp_dist == -1*disc and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == disc and y_cur_dist == -1*disc and ((x_comp_dist == disc and y_comp_dist == disc \
                                   and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next) or (x_comp_dist == -1*disc \
                                   and y_comp_dist == -1*disc and x_cur_prev == x_comp_next and y_cur_prev == y_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                else:
                    # add downwash check here ***
                    if (x_cur_dist == disc and y_cur_dist == disc and ((x_comp_dist == -1*disc and y_comp_dist == disc \
                        and x_cur_prev == x_comp_next and y_cur_prev == y_comp_prev) or (x_comp_dist == disc and y_comp_dist == -1*disc \
                        and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next)) and ((z_cur_dist == z_comp_dist and z_cur_prev == z_comp_prev) \
                        or (z_cur_dist == disc_z and z_comp_dist == -1*disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next) \
                        or (z_cur_dist == -1*disc_z and z_comp_dist == disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == -1*disc and ((x_comp_dist == -1*disc and y_comp_dist == disc \
                        and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next) or (x_comp_dist == disc and y_comp_dist == -1*disc \
                        and y_cur_prev == y_comp_prev and x_cur_prev == x_comp_next)) and ((z_cur_dist == z_comp_dist and z_cur_prev == z_comp_prev) \
                        or (z_cur_dist == disc_z and z_comp_dist == -1*disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next) \
                        or (z_cur_dist == -1*disc_z and z_comp_dist == disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == disc and ((x_comp_dist == disc and y_comp_dist == disc \
                        and x_cur_prev == x_comp_next and y_cur_prev == y_comp_prev) or (x_comp_dist == -1*disc and y_comp_dist == -1*disc \
                        and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next)) and ((z_cur_dist == z_comp_dist and z_cur_prev == z_comp_prev) \
                        or (z_cur_dist == disc_z and z_comp_dist == -1*disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next) \
                        or (z_cur_dist == -1*disc_z and z_comp_dist == disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == disc and y_cur_dist == -1*disc and ((x_comp_dist == disc and y_comp_dist == disc \
                        and x_cur_prev == x_comp_prev and y_cur_prev == y_comp_next) or (x_comp_dist == -1*disc and y_comp_dist == -1*disc \
                        and x_cur_prev == x_comp_next and y_cur_prev == y_comp_next)) and ((z_cur_dist == z_comp_dist and z_cur_prev == z_comp_prev) \
                        or (z_cur_dist == disc_z and z_comp_dist == -1*disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next) \
                        or (z_cur_dist == -1*disc_z and z_comp_dist == disc_z and z_cur_prev == z_comp_next and z_comp_prev == z_cur_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    # in plane movement (xz plane)
                    elif (x_cur_dist == disc and y_cur_dist == 0 and z_cur_dist == disc_z and y_comp_dist == 0 and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == 0 and z_cur_dist == -1*disc_z and y_comp_dist == 0 and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_node = ts_next_states[k]
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == 0 and z_cur_dist == disc_z and y_comp_dist == 0 and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_node = ts_next_states[k]
                        break
                    elif (x_cur_dist == disc and y_cur_dist == 0 and z_cur_dist == -1*disc_z and y_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_prev))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    # in plane movement (yz plane)
                    elif (y_cur_dist == disc and x_cur_dist == 0 and z_cur_dist == disc_z and x_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (y_cur_dist == -1*disc and x_cur_dist == 0 and z_cur_dist == -1*disc_z and x_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (y_cur_dist == -1*disc and x_cur_dist == 0 and z_cur_dist == disc_z and x_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (y_cur_dist == disc and x_cur_dist == 0 and z_cur_dist == -1*disc_z and x_comp_dist == 0 and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    # in plane movement (diagonal plane 1)
                    elif (x_cur_dist == disc and y_cur_dist == disc and z_cur_dist == disc_z and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == -1*disc and z_cur_dist == -1*disc_z and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == -1*disc and z_cur_dist == disc_z and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == disc and y_cur_dist == disc and z_cur_dist == -1*disc_z and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    # in plane movement (diagonal plane 2)
                    elif (x_cur_dist == disc and y_cur_dist == -1*disc and z_cur_dist == disc_z and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == disc and z_cur_dist == -1*disc_z and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == -1*disc and y_cur_dist == disc and z_cur_dist == disc_z and ((x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev) or (x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next))):
                        weighted_nodes.append(ts_next_states[k])
                        break
                    elif (x_cur_dist == disc and y_cur_dist == -1*disc and z_cur_dist == -1*disc_z and ((x_cur_prev == x_comp_prev \
                                                   and y_cur_prev == y_comp_prev and z_cur_prev == z_comp_next) or (x_cur_prev == x_comp_next \
                                                   and y_cur_prev == y_comp_next and z_cur_prev == z_comp_prev))):
                        weighted_nodes.append(ts_next_states[k])
                        break
    return weighted_nodes

def update_adj_mat_diag(m, n, adj_mat, obs_mat):
    ''' Update the adjacency matrix given an obserrvation matrix '''
    for i in range(m):
        for j in range(n):
            if obs_mat[i][j] != 3:
                diag_ind = n*i + j
                adj_mat[diag_ind][diag_ind] = 1
                if j < n-1:
                    right_ind = n*i + j + 1
                    if obs_mat[i][j+1] != 3:
                        adj_mat[diag_ind][right_ind] = 1
                        adj_mat[right_ind][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][right_ind] = 0
                        adj_mat[right_ind][diag_ind] = 0
                if j > 0:
                    left_ind = n*i + j - 1
                    if obs_mat[i][j-1] != 3:
                        adj_mat[diag_ind][left_ind] = 1
                        adj_mat[left_ind][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][left_ind] = 0
                        adj_mat[left_ind][diag_ind] = 0
                if i > 0:
                    up_ind = n*(i-1) + j
                    if obs_mat[i-1][j] != 3:
                        adj_mat[diag_ind][up_ind] = 1
                        adj_mat[up_ind][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][up_ind] = 0
                        adj_mat[up_ind][diag_ind] = 0
                if i < m-1:
                    down_ind = n*(i+1) + j
                    if obs_mat[i+1][j] != 3:
                        adj_mat[diag_ind][down_ind] = 1
                        adj_mat[down_ind][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][down_ind] = 0
                        adj_mat[down_ind][diag_ind] = 0
                # Now perform the diagonal indexing
                if i == 0 and j == 0: # upper left
                    SE_index = n*(i+1) + j + 1
                    if obs_mat[i+1][j+1] != 3:
                        adj_mat[diag_ind][SE_index] = 1
                        adj_mat[SE_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][SE_index] = 0
                        adj_mat[SE_index][diag_ind] = 0
                if i > 0 and i < m-1 and j == 0: # left column (not corner)
                    NE_index = n*(i-1) + j + 1
                    SE_index = n*(i+1) + j + 1
                    if obs_mat[i-1][j+1] != 3:
                        adj_mat[diag_ind][NE_index] = 1
                        adj_mat[NE_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][NE_index] = 0
                        adj_mat[NE_index][diag_ind] = 0
                    if obs_mat[i+1][j+1] != 3:
                        adj_mat[diag_ind][SE_index] = 1
                        adj_mat[SE_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][SE_index] = 0
                        adj_mat[SE_index][diag_ind] = 0
                if i == m-1 and n == 0:  # lower left
                    NE_index = n*(i-1) + j + 1
                    if obs_mat[i-1][j+1] != 3:
                        adj_mat[diag_ind][NE_index] = 1
                        adj_mat[NE_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][NE_index] = 0
                        adj_mat[NE_index][diag_ind] = 0
                if i == 0 and j < n-1 and j > 0:  # upper row (not corner)
                    SW_index = n*(i+1) + j - 1
                    SE_index = n*(i+1) + j + 1
                    if obs_mat[i+1][j-1] != 3:
                        adj_mat[diag_ind][SW_index] = 1
                        adj_mat[SW_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][SW_index] = 0
                        adj_mat[SW_index][diag_ind] = 0
                    if obs_mat[i+1][j+1] != 3:
                        adj_mat[diag_ind][SE_index] = 1
                        adj_mat[SE_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][SE_index] = 0
                        adj_mat[SE_index][diag_ind] = 0
                if i == 0 and j == n-1: # upper right
                    SW_index = n*(i+1) + j - 1
                    if obs_mat[i+1][j-1] != 3:
                        adj_mat[diag_ind][SW_index] = 1
                        adj_mat[SW_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][SW_index] = 0
                        adj_mat[SW_index][diag_ind] = 0
                if i > 0 and j == n-1 and i < m-1:  # right column (not corner)
                    NW_index = n*(i-1) + j - 1
                    SW_index = n*(i+1) + j - 1
                    if obs_mat[i-1][j-1] != 3:
                        adj_mat[diag_ind][NW_index] = 1
                        adj_mat[NW_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][NW_index] = 0
                        adj_mat[NW_index][diag_ind] = 0
                    if obs_mat[i+1][j-1] != 3:
                        adj_mat[diag_ind][SW_index] = 1
                        adj_mat[SW_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][SW_index] = 0
                        adj_mat[SW_index][diag_ind] = 0
                if i == m-1 and j == n-1:  # bottom right
                    NW_index = n*(i-1) + j - 1
                    if obs_mat[i-1][j-1] != 3:
                        adj_mat[diag_ind][NW_index] = 1
                        adj_mat[NW_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][NW_index] = 0
                        adj_mat[NW_index][diag_ind] = 0
                if i == m-1 and j > 0 and j < n-1:  # bottom row (not corner)
                    NW_index = n*(i-1) + j - 1
                    NE_index = n*(i-1) + j + 1
                    if obs_mat[i-1][j-1] != 3:
                        adj_mat[diag_ind][NW_index] = 1
                        adj_mat[NW_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][NW_index] = 0
                        adj_mat[NW_index][diag_ind] = 0
                    if obs_mat[i-1][j+1] != 3:
                        adj_mat[diag_ind][NE_index] = 1
                        adj_mat[NE_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][NE_index] = 0
                        adj_mat[NE_index][diag_ind] = 0
                if i > 0 and i < m-1 and j > 0 and j < n-1: # all middle nodes
                    NW_index = n*(i-1) + j - 1
                    NE_index = n*(i-1) + j + 1
                    SW_index = n*(i+1) + j - 1
                    SE_index = n*(i+1) + j + 1
                    if obs_mat[i-1][j-1] != 3:
                        adj_mat[diag_ind][NW_index] = 1
                        adj_mat[NW_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][NW_index] = 0
                        adj_mat[NW_index][diag_ind] = 0
                    if obs_mat[i-1][j+1] != 3:
                        adj_mat[diag_ind][NE_index] = 1
                        adj_mat[NE_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][NE_index] = 0
                        adj_mat[NE_index][diag_ind] = 0
                    if obs_mat[i+1][j-1] != 3:
                        adj_mat[diag_ind][SW_index] = 1
                        adj_mat[SW_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][SW_index] = 0
                        adj_mat[SW_index][diag_ind] = 0
                    if obs_mat[i+1][j+1] != 3:
                        adj_mat[diag_ind][SE_index] = 1
                        adj_mat[SE_index][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][SE_index] = 0
                        adj_mat[SE_index][diag_ind] = 0
            else:
                # this indicates the region is an obstacle
                diag_ind = n*i + j
                adj_mat[diag_ind][diag_ind] = 0
                if j < n-1:
                    right_ind = n*i + j + 1
                    adj_mat[diag_ind][right_ind] = 0
                    adj_mat[right_ind][diag_ind] = 0
                if j > 0:
                    left_ind = n*i + j - 1
                    adj_mat[diag_ind][left_ind] = 0
                    adj_mat[left_ind][diag_ind] = 0
                if i > 0:
                    up_ind = n*(i-1) + j
                    adj_mat[diag_ind][up_ind] = 0
                    adj_mat[up_ind][diag_ind] = 0
                if i < m-1:
                    down_ind = n*(i+1) + j
                    adj_mat[diag_ind][down_ind] = 0
                    adj_mat[down_ind][diag_ind] = 0

                if i == 0 and j == 0: # upper left
                    SE_index = n*(i+1) + j + 1
                    adj_mat[diag_ind][SE_index] = 0
                    adj_mat[SE_index][diag_ind] = 0
                if i > 0 and i < m-1 and j == 0: # left column (not corner)
                    NE_index = n*(i-1) + j + 1
                    SE_index = n*(i+1) + j + 1
                    adj_mat[diag_ind][NE_index] = 0
                    adj_mat[NE_index][diag_ind] = 0
                    adj_mat[diag_ind][SE_index] = 0
                    adj_mat[SE_index][diag_ind] = 0
                if i == m-1 and n == 0:  # lower left
                    NE_index = n*(i-1) + j + 1
                    adj_mat[diag_ind][NE_index] = 0
                    adj_mat[NE_index][diag_ind] = 0
                if i == 0 and j < n-1 and j > 0:  # upper row (not corner)
                    SW_index = n*(i+1) + j - 1
                    SE_index = n*(i+1) + j + 1
                    adj_mat[diag_ind][SW_index] = 0
                    adj_mat[SW_index][diag_ind] = 0
                    adj_mat[diag_ind][SE_index] = 0
                    adj_mat[SE_index][diag_ind] = 0
                if i == 0 and j == n-1: # upper right
                    SW_index = n*(i+1) + j - 1
                    adj_mat[diag_ind][SW_index] = 0
                    adj_mat[SW_index][diag_ind] = 0
                if i > 0 and j == n-1 and i < m-1:  # right column (not corner)
                    NW_index = n*(i-1) + j - 1
                    SW_index = n*(i+1) + j - 1
                    adj_mat[diag_ind][NW_index] = 0
                    adj_mat[NW_index][diag_ind] = 0
                    adj_mat[diag_ind][SW_index] = 0
                    adj_mat[SW_index][diag_ind] = 0
                if i == m-1 and j == n-1:  # bottom right
                    NW_index = n*(i-1) + j - 1
                    adj_mat[diag_ind][NW_index] = 0
                    adj_mat[NW_index][diag_ind] = 0
                if i == m-1 and j > 0 and j < n-1:  # bottom row (not corner)
                    NW_index = n*(i-1) + j - 1
                    NE_index = n*(i-1) + j + 1
                    adj_mat[diag_ind][NW_index] = 0
                    adj_mat[NW_index][diag_ind] = 0
                    adj_mat[diag_ind][NE_index] = 0
                    adj_mat[NE_index][diag_ind] = 0
                if i > 0 and i < m-1 and j > 0 and j < n-1: # all middle nodes
                    NW_index = n*(i-1) + j - 1
                    NE_index = n*(i-1) + j + 1
                    SW_index = n*(i+1) + j - 1
                    SE_index = n*(i+1) + j + 1
                    adj_mat[diag_ind][NW_index] = 0
                    adj_mat[NW_index][diag_ind] = 0
                    adj_mat[diag_ind][NE_index] = 0
                    adj_mat[NE_index][diag_ind] = 0
                    adj_mat[diag_ind][SW_index] = 0
                    adj_mat[SW_index][diag_ind] = 0
                    adj_mat[diag_ind][SE_index] = 0
                    adj_mat[SE_index][diag_ind] = 0
    return adj_mat
