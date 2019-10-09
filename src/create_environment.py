#
# module: create_environment.py
#
# input: Data regarding the environment. Discretization, obstacles, initial states
# output: Creates the ts_synthesis.txt files for the environment that can be
#         executed during run time.
#
# author: Ryan Peterson pete9936@umn.edu
#

import numpy as np
import StringIO
import pdb, os, math

def create_ts(m, n):
    ''' function to create the initial grid for the transition system given
    the m x n grid'''
    # Initialize the states to their node names
    state_mat = np.arange(m*n).reshape((m,n))
    # Initialize an observation matrix with null observations
    obs_mat = np.zeros((m,n))
    adj_mat = np.zeros((m*n,m*n))
    # Populate the adjacency matrix with the initial states
    TS = update_adj_mat(m, n, adj_mat, obs_mat)
    return TS, obs_mat, state_mat

def update_obs_mat(obs_mat, state_mat, obstacles = None, init_state = None):
    ''' update the observation matrix with known data so we can update the
    adjacency matrix and therefore the environment file '''
    if obstacles != None:
        for i in range(len(obstacles)):
            index1 = obstacles[i][0]
            index2 = obstacles[i][1]
            obs_mat[index1][index2] = 3
    if init_state != None:
        state_loc = np.argwhere(state_mat == init_state)
        m = state_loc[0][0]
        n = state_loc[0][1]
        obs_mat[m][n] = 1
    return obs_mat

def update_adj_mat(m, n, adj_mat, obs_mat):
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
    return adj_mat

def create_input_file(adj_mat, state_mat, obs_mat, path, bases, disc, iter):
    ''' Given the adjacency matrix this creates the proper text file for the
        execution. '''
    nodeset1 = []
    nodeset2 = []
    weight = []
    for i in range(m*n):
        for j in range(m*n):
            if adj_mat[i][j] > 0:
                nodeset1.append(i)
                nodeset2.append(j)
                weight.append(adj_mat[i][j])
    with open(path, 'w+') as f1:
        # Print file header and initial position
        f1.write('name Simple DTS\n')
        if iter == 0:
            f1.write('init {''Base'':1}\n')
        else:
            f1.write('init {''Base%d'':1}\n' % iter)
        f1.write(';\n')

        # Publish the sets of nodes
        for i in range(m):
            for j in range(n):
                x = j*disc
                y = -i*disc
                if obs_mat[i][j] == 3:
                    continue
                elif state_mat[i][j] in bases:
                    for key in bases:
                        if key == state_mat[i][j]:
                            f1.write('%s {''prop'': set(), ''position'': (%1.2f, %1.2f)}\n' % (bases[key], x, y))
                else:
                    f1.write('r%d {''prop'':{''r%d''}, ''position'': (%1.2f, %1.2f)}\n'\
                                            % (state_mat[i][j], state_mat[i][j], x, y))
        f1.write(';\n')

        # Publish the sets of edges and edge weights
        for i in range(len(nodeset1)):
            if nodeset1[i] in bases:
                if nodeset2[i] in bases:
                    for key1 in bases:
                        if nodeset1[i] == key1:
                            for key2 in bases:
                                if nodeset2[i] == key2:
                                    f1.write('%s %s {''duration'': %d}\n'\
                                            % (bases[key1], bases[key2], weight[i]))
                else:
                    for key1 in bases:
                        if nodeset1[i] == key1:
                            f1.write('%s r%d {''duration'': %d}\n' % (bases[key1], nodeset2[i], weight[i]))
            elif nodeset2[i] in bases:
                for key2 in bases:
                    if nodeset2[i] == key2:
                        f1.write('r%d %s {''duration'': %d}\n' % (nodeset1[i], bases[key2], weight[i]))
            else:
                f1.write('r%d r%d {''duration'': %d}\n' % (nodeset1[i], nodeset2[i], weight[i]))
    # finished writing to file
    f1.close()


if __name__ == '__main__':
    m = 6
    n = 6
    TS, obs_mat, state_mat = create_ts(m,n)
    # try out the init state and obstacles functions
    init_state = 31
    obstacles = [(5,3),(4,3),(2,3),(2,4)]
    obs_mat = update_obs_mat(obs_mat, state_mat, obstacles, init_state)
    # Update the adjacency matrix
    TS = update_adj_mat(m, n, TS, obs_mat)
    paths = ['../data/ts_synth_6x6_test1.txt', '../data/ts_synth_6x6_test2.txt', '../data/ts_synth_6x6_test3.txt']
    bases = {31: 'Base', 35: 'Base2', 8: 'Base3'}
    disc = 0.5
    # Now create the proper output .txt files
    for i in range(len(paths)):
        create_input_file(TS, state_mat, obs_mat, paths[i], bases, disc, i)
