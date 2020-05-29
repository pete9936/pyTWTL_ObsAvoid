#
# module: DP_paths.py
#
# synopsis: Module implements the Dynamic Programming (DP) method to generate
#           minimum cost paths for receding horizon implementation and for
#           deadlock resolution
#
# author: Ryan Peterson pete9936@umn.edu
#

import math
import networkx as nx
import pdb


def local_horizonDP(pa, weighted_nodes, num_hops, init_loc):
    ''' Compute the n-hop lowest energy horizon without conflicts using
    a targeted Dynamic Programming (DP) method. '''
    # Compute the n-hop trajectory, ensures a minimum 2-hop trajectory
    # and performs check on deadlock
    ts_policy = []
    pa_policy = []
    D_flag = False
    # Initialize local set
    local_set = init_loc
    final_flag = False
    old_local_set = {}
    # Expand subgraph of nodes out to num_hops for search
    for i in range(num_hops):
        # Expand subgraph for next time step
        old_local_set[i] = local_set
        local_set_temp = []
        local_set = []
        if i == 0:
            temp_set = pa.g.neighbors(old_local_set[i])
            for node in temp_set:
                if node not in local_set_temp:
                    local_set_temp.append(node)
        else:
            for loc_node in old_local_set[i]:
                temp_set = pa.g.neighbors(loc_node)
                for node in temp_set:
                    if node not in local_set_temp:
                        local_set_temp.append(node)
        # Remove conflicting nodes in the set
        for neighbor in local_set_temp:
            if neighbor[0] not in weighted_nodes[i]:
                local_set.append(neighbor)
        # Check if the agent is in a deadlock situation, or there are no feasible transitions
        if not local_set:
            if i==0:
                D_flag = True
                return None, None, D_flag
            else:
                i = i-1
                local_set = old_local_set[i]
                break
        # Check if any of the nodes are in the final set and if so break and use node
        for node in local_set:
            if node in pa.final:
                final_flag = True
                target_node = node
                break
        else:
            continue
        break
    # Search for lowest energy value in remaining set of nodes and use for shortest path
    if final_flag == False:
        all_node_energy = nx.get_node_attributes(pa.g,'energy')
        local_set_energy = []
        for node in local_set:
            node_energy = all_node_energy[node]
            local_set_energy.append(node_energy)
        # Get index of the minimum energy path
        index_min = min(xrange(len(local_set_energy)), key=local_set_energy.__getitem__)
        target_node = local_set[index_min]

    # Perform the targeted DP method
    num_trans = i+1
    edges_all = nx.get_edge_attributes(pa.g,'new_weight')
    all_node_energy = nx.get_node_attributes(pa.g,'energy')
    node_dict = {}
    node_costs = {}
    if num_trans > 1:
        for j in range(num_trans-1):
            advance_flag = False
            node_list_temp = []
            trans_cost_temp = []
            if j == 0:
                check_nodes = pa.g.neighbors(target_node)
                for check_node in check_nodes:
                    if check_node in old_local_set[num_trans-1]:
                        advance_flag = True
                        edge = (check_node, target_node)
                        try:
                            trans_cost = all_node_energy[check_node] + edges_all[edge]
                            node_list_temp.append(check_node)
                            trans_cost_temp.append(trans_cost)
                        except KeyError:
                            # If traversing on the same state (i.e. ('r21',1)->('r21',2))
                            new_edge = pa.g.in_edges(target_node)[0]
                            trans_cost = all_node_energy[new_edge[0]] + edges_all[new_edge]
                            node_list_temp = [new_edge[0]]
                            trans_cost_temp = [trans_cost]
                            break
                if advance_flag == False:
                    try:
                        new_edge = ((target_node[0],target_node[1]-1), target_node)
                        trans_cost = all_node_energy[new_edge[0]] + edges_all[new_edge]
                        node_dict[num_trans-1] = [new_edge[0]]
                        node_costs[num_trans-1] = [trans_cost]
                    except KeyError:
                        edges = pa.g.in_edges(target_node)
                        for edge in edges:
                            if edge[0] in old_local_set[num_trans-1]:
                                trans_cost = all_node_energy[edge[0]] + edges_all[edge]
                                node_list_temp.append(edge[0])
                                trans_cost_temp.append(trans_cost)
                        node_dict[num_trans-1] = node_list_temp
                        node_costs[num_trans-1] = trans_cost_temp
                else:
                    node_dict[num_trans-1] = node_list_temp
                    node_costs[num_trans-1] = trans_cost_temp
            else:
                for old_node in node_dict[num_trans-j]:
                    old_node_index = node_dict[num_trans-j].index(old_node)
                    old_node_cost = node_costs[num_trans-j][old_node_index]
                    check_nodes = pa.g.neighbors(old_node)
                    for check_node in check_nodes:
                        if check_node in old_local_set[num_trans-j-1]:
                            advance_flag = True
                            edge = (check_node, old_node)
                            try:
                                trans_cost = old_node_cost + all_node_energy[check_node] + edges_all[edge]
                                # Need a check on trans_cost to see if it's the lowest (replace if needed)
                                if check_node in node_list_temp:
                                    check_index = node_list_temp.index(check_node)
                                    if trans_cost_temp[check_index] > trans_cost:
                                        trans_cost_temp[check_index] = trans_cost
                                else:
                                    node_list_temp.append(check_node)
                                    trans_cost_temp.append(trans_cost)
                            except KeyError:
                                # If traversing on the same state (i.e. ('r21',1)->('r21',2))
                                new_edge = pa.g.in_edges(old_node)[0]
                                trans_cost = old_node_cost + all_node_energy[new_edge[0]] + edges_all[new_edge]
                                node_list_temp = [new_edge[0]]
                                trans_cost_temp = [trans_cost]
                                break
                    # Save updated dictionary values for next step
                    if advance_flag == False:
                        try:
                            new_edge = ((old_node[0],old_node[1]-1), old_node)
                            trans_cost = old_node_cost + all_node_energy[new_edge[0]] + edges_all[new_edge]
                            node_dict[num_trans-j-1] = [new_edge[0]]
                            node_costs[num_trans-j-1] = [trans_cost]
                            break
                        except KeyError:
                            edges = pa.g.in_edges(old_node)
                            for edge in edges:
                                if edge[0] in old_local_set[num_trans-j-1]:
                                    trans_cost = old_node_cost + all_node_energy[edge[0]] + edges_all[edge]
                                    node_list_temp.append(edge[0])
                                    trans_cost_temp.append(trans_cost)
                            node_dict[num_trans-j-1] = node_list_temp
                            node_costs[num_trans-j-1] = trans_cost_temp
                            break
                if advance_flag == True:
                    node_dict[num_trans-j-1] = node_list_temp
                    node_costs[num_trans-j-1] = trans_cost_temp
        # Handle the last transition differently by just adding in the edge_weight
        for ind, old_node in enumerate(node_dict[1]):
            edge = (init_loc, old_node)
            node_costs[1][ind] = node_costs[1][ind] + edges_all[edge]

        # Construct lowest cost feasible path based on DP calculations above
        path = []
        for key in node_dict:
            if key == 1:
                # Get index of the minimum cost reachable node
                index_min = min(xrange(len(node_costs[1])), key=node_costs[1].__getitem__)
                path_node_prior = node_dict[1][index_min]
                path.append(path_node_prior)
            else:
                check_nodes = pa.g.neighbors(path_node_prior)
                path_node_cost = float('inf')
                for check_node in check_nodes:
                    if check_node in node_dict[key]:
                        temp_cost = node_costs[key][node_dict[key].index(check_node)]
                        if temp_cost < path_node_cost:
                            path_node_cost = temp_cost
                            path_node_prior = check_node
                path.append(path_node_prior)
        # Append the final target node to path
        path.append(target_node)
        # generate policy based on the generated path
        for p in path:
            ts_policy.append(p[0])
            pa_policy.append(p)
    else:
        ts_policy.append(target_node[0])
        pa_policy.append(target_node)

    return ts_policy, pa_policy, D_flag


def deadlock_path(pa, occupied_nodes, init_loc):
    ''' Find the sortest path to the nearest unoccupied node for
    deadlock resolution. '''
    ts_path = []
    pa_path = []
    conflict_nodes = [occupied_nodes[0], occupied_nodes[1]]
    # Initialize local set
    local_set = init_loc
    old_local_set = {}
    open_flag = False
    target_set = []
    i = 0
    # Expand subgraph of nodes out until unoccupied node is found
    while open_flag == False:
        # Expand subgraph for next time step
        old_local_set[i] = local_set
        local_set_temp = []
        local_set = []
        if i == 0:
            temp_set = pa.g.neighbors(old_local_set[i])
            for node in temp_set:
                if node not in local_set_temp:
                    local_set_temp.append(node)
        else:
            for loc_node in old_local_set[i]:
                temp_set = pa.g.neighbors(loc_node)
                for node in temp_set:
                    if node not in local_set_temp:
                        local_set_temp.append(node)
        # Remove conflicting nodes in the set
        for neighbor in local_set_temp:
            if neighbor[0] not in conflict_nodes:
                local_set.append(neighbor)
        # Check if unoccupied nodes
        for node in local_set:
            if node[0] not in occupied_nodes:
                target_set.append(node)
                open_flag = True
        # Perform another hop
        if not target_set:
            local_set = local_set_temp
        i = i+1
    # Search for lowest energy value in remaining set of nodes and use for shortest path
    all_node_energy = nx.get_node_attributes(pa.g,'energy')
    target_set_energy = []
    for node in target_set:
        node_energy = all_node_energy[node]
        target_set_energy.append(node_energy)
    # Get index of the minimum energy path
    index_min = min(xrange(len(target_set_energy)), key=target_set_energy.__getitem__)
    target_node = target_set[index_min]
    # Perform the targeted DP method
    num_trans = i
    edges_all = nx.get_edge_attributes(pa.g,'new_weight')
    all_node_energy = nx.get_node_attributes(pa.g,'energy')
    node_dict = {}
    node_costs = {}
    if num_trans > 1:
        for j in range(num_trans-1):
            advance_flag = False
            node_list_temp = []
            trans_cost_temp = []
            if j == 0:
                check_nodes = pa.g.neighbors(target_node)
                for check_node in check_nodes:
                    if check_node in old_local_set[num_trans-1]:
                        advance_flag = True
                        edge = (check_node, target_node)
                        try:
                            trans_cost = all_node_energy[check_node] + edges_all[edge]
                            node_list_temp.append(check_node)
                            trans_cost_temp.append(trans_cost)
                        except KeyError:
                            # If traversing on the same state (i.e. ('r21',1)->('r21',2))
                            new_edge = pa.g.in_edges(target_node)[0]
                            trans_cost = all_node_energy[new_edge[0]] + edges_all[new_edge]
                            node_list_temp = [new_edge[0]]
                            trans_cost_temp = [trans_cost]
                            break
                if advance_flag == False:
                    try:
                        new_edge = ((target_node[0],target_node[1]-1), target_node)
                        trans_cost = all_node_energy[new_edge[0]] + edges_all[new_edge]
                        node_dict[num_trans-1] = [new_edge[0]]
                        node_costs[num_trans-1] = [trans_cost]
                    except KeyError:
                        edges = pa.g.in_edges(target_node)
                        for edge in edges:
                            if edge[0] in old_local_set[num_trans-1]:
                                trans_cost = all_node_energy[edge[0]] + edges_all[edge]
                                node_list_temp.append(edge[0])
                                trans_cost_temp.append(trans_cost)
                        node_dict[num_trans-1] = node_list_temp
                        node_costs[num_trans-1] = trans_cost_temp
                else:
                    node_dict[num_trans-1] = node_list_temp
                    node_costs[num_trans-1] = trans_cost_temp
            else:
                for old_node in node_dict[num_trans-j]:
                    old_node_index = node_dict[num_trans-j].index(old_node)
                    old_node_cost = node_costs[num_trans-j][old_node_index]
                    check_nodes = pa.g.neighbors(old_node)
                    for check_node in check_nodes:
                        if check_node in old_local_set[num_trans-j-1]:
                            advance_flag = True
                            edge = (check_node, old_node)
                            try:
                                trans_cost = old_node_cost + all_node_energy[check_node] + edges_all[edge]
                                # Need a check on trans_cost to see if it's the lowest (replace if needed)
                                if check_node in node_list_temp:
                                    check_index = node_list_temp.index(check_node)
                                    if trans_cost_temp[check_index] > trans_cost:
                                        trans_cost_temp[check_index] = trans_cost
                                else:
                                    node_list_temp.append(check_node)
                                    trans_cost_temp.append(trans_cost)
                            except KeyError:
                                # If traversing on the same state (i.e. ('r21',1)->('r21',2))
                                new_edge = pa.g.in_edges(old_node)[0]
                                trans_cost = old_node_cost + all_node_energy[new_edge[0]] + edges_all[new_edge]
                                node_list_temp = [new_edge[0]]
                                trans_cost_temp = [trans_cost]
                                break
                    # Save updated dictionary values for next step
                    if advance_flag == False:
                        try:
                            new_edge = ((old_node[0],old_node[1]-1), old_node)
                            trans_cost = old_node_cost + all_node_energy[new_edge[0]] + edges_all[new_edge]
                            node_dict[num_trans-j-1] = [new_edge[0]]
                            node_costs[num_trans-j-1] = [trans_cost]
                            break
                        except KeyError:
                            edges = pa.g.in_edges(old_node)
                            for edge in edges:
                                if edge[0] in old_local_set[num_trans-j-1]:
                                    trans_cost = old_node_cost + all_node_energy[edge[0]] + edges_all[edge]
                                    node_list_temp.append(edge[0])
                                    trans_cost_temp.append(trans_cost)
                            node_dict[num_trans-j-1] = node_list_temp
                            node_costs[num_trans-j-1] = trans_cost_temp
                            break
                if advance_flag == True:
                    node_dict[num_trans-j-1] = node_list_temp
                    node_costs[num_trans-j-1] = trans_cost_temp
        # Handle the last transition differently by just adding in the edge_weight
        for ind, old_node in enumerate(node_dict[1]):
            edge = (init_loc, old_node)
            node_costs[1][ind] = node_costs[1][ind] + edges_all[edge]

        # Construct lowest cost feasible path based on DP calculations above
        path = []
        for key in node_dict:
            if key == 1:
                # Get index of the minimum cost reachable node
                index_min = min(xrange(len(node_costs[1])), key=node_costs[1].__getitem__)
                path_node_prior = node_dict[1][index_min]
                path.append(path_node_prior)
            else:
                check_nodes = pa.g.neighbors(path_node_prior)
                path_node_cost = float('inf')
                for check_node in check_nodes:
                    if check_node in node_dict[key]:
                        temp_cost = node_costs[key][node_dict[key].index(check_node)]
                        if temp_cost < path_node_cost:
                            path_node_cost = temp_cost
                            path_node_prior = check_node
                path.append(path_node_prior)
        # Append the final target node to path
        path.append(target_node)
        # generate policy based on the generated path
        for p in path:
            ts_path.append(p[0])
            pa_path.append(p)
    else:
        ts_path.append(target_node[0])
        ts_path.append(target_node[0])
        pa_path.append(target_node)
        pa_path.append(target_node)

    return ts_path, pa_path


if __name__ == '__main__':
    pass
