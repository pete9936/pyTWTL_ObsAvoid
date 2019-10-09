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
 ''' function to create the initial grid for the transition system given the
     grid m x n '''
    # Initialize the states to their node names, might not need the transpose ***
    state_mat = np.arange(m*n).reshape((m,n)).T

    # Initialize an observation matrix with null observations
    obs_mat = np.zeros((m,n))
    adj_mat = np.zeros((m*n,m*n))

    # Populate the adjacency matrix with the initial states
    TS = update_adj_mat(m, n, adj_mat, obs_mat)
    return TS, obs_mat, state_mat

def update_obs_mat(obs_mat, state_mat, obstacles = None, init_state = None):
    ''' update the observation matrix with known data so we can update the
    adjacency matrix and therefore the environment file '''
    pdb.set_trace()
    if obstacles != None:
        for i in range(len(obstacles)):
            index1 = obstacles[i][0]
            index2 = obstacles[i][1]
            obs_mat[index1][index2] = 3
    if init_state != None:
        state_loc = np.argwhere(state_mat == init_state)
        m = state_loc[0]
        n = state_loc[1]
        obs_mat[m][n] = 1

    return obs_mat

def update_adj_mat(m, n, adj_mat, obs_mat):
    ''' Update the adjacency matrix given an obserrvation matrix '''
    for i in range(m):
        for j in range(n):
            if obs_mat[i][j] != 3
                diag_ind = m*i + j
                adj_mat[diag_ind][diag_ind] = 1
                if j < n-1:
                    right_ind = m*i + j + 1
                    if obs_mat[diag_ind][right_ind] != 3:
                        adj_mat[diag_ind][right_ind] = 1
                        adj_mat[right_ind][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][right_ind] = 0
                        adj_mat[right_ind][diag_ind] = 0
                if j > 0:
                    left_ind = m*i + j - 1
                    if obs_mat[diag_ind][left_ind] != 3
                        adj_mat[diag_ind][left_ind] = 1
                        adj_mat[left_ind][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][left_ind] = 0
                        adj_mat[left_ind][diag_ind] = 0
                if i > 0:
                    up_ind = m*(i-1) + j
                    if obs_mat[diag_ind][up_ind] != 3:
                        adj_mat[diag_ind][up_ind] = 1
                        adj_mat[up_ind][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][up_ind] = 0
                        adj_mat[up_ind][diag_ind] = 0
                if i < m-1:
                    down_ind = m*(i+1) + j
                    if obs_mat[diag_ind][down_ind] != 3:
                        adj_mat[diag_ind][down_ind] = 1
                        adj_mat[down_ind][diag_ind] = 1
                    else:
                        adj_mat[diag_ind][down_ind] = 0
                        adj_mat[down_ind][diag_ind] = 0
            else:
                # this indicates the region is an obstacle
                diag_ind = m*i + j
                adj_mat[diag_ind][diag_ind] = 0
                if j < n-1:
                    right_ind = m*i + j + 1
                    adj_mat[diag_ind][right_ind] = 0
                    adj_mat[right_ind][diag_ind] = 0
                if j > 0:
                    left_ind = m*i + j - 1
                    adj_mat[diag_ind][left_ind] = 0
                    adj_mat[left_ind][diag_ind] = 0
                if i > 0:
                    up_ind = m*(i-1) + j
                    adj_mat[diag_ind][up_ind] = 0
                    adj_mat[up_ind][diag_ind] = 0
                if i < m-1:
                    down_ind = m*(i+1) + j
                    adj_mat[diag_ind][down_ind] = 0
                    adj_mat[down_ind][diag_ind] = 0
    return adj_mat

def create_input_file(adj_mat, path, bases, disc):
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
        f1.write('init {''Base3'':1}\n')  # Need to write for specific agent or all? ***
        f1.write(';\n')

        # Publish the sets of nodes
        for i in range(m):
            for j in range(n):
                x = (j-1)*disc
                y = -(i-1)*disc
                if T.obs(i,j) == 3
                    continue;
                elseif T.states(i,j) == Base:
                    fprintf(fid,'Base {''prop'': set(), ''position'': (%1.1f, %1.1f)}\n', x, y);
                elseif T.states(i,j) == Base2
                    fprintf(fid,'Base2 {''prop'':{''Base2''}, ''position'': (%1.1f, %1.1f)}\n', x, y);
                elseif T.states(i,j) == Base3
                    fprintf(fid,'Base3 {''prop'':{''Base3''}, ''position'': (%1.1f, %1.1f)}\n', x, y);
                else
                    fprintf(fid,'%d {''prop'':{''%d''}, ''position'': (%1.1f, %1.1f)}\n',...
                                               states[i][j], states[i][j], x, y);

        f1.write(';\n')

        f1.write('Nominal Control Policy for agent %s.\n' % key)
        f1.write('Optimal relaxation is: %s \n' % tau)
        f1.write('Generated PA control policy is: (')
        f1.write(') -> ('.join('%s %s' % x for x in pa_nom_policy))
        f1.write(') \nGenerated control policy is: %s \n\n' % ts_nom_policy)
        f1.write('Final Control policy for agent %s.\n' % key)
        f1.write('Generated PA control policy is: (')
        f1.write(') -> ('.join('%s %s' % x for x in pa_policy))
        f1.write(') \nGenerated TS control policy is:  %s \n\n' % ts_policy)

    # Publish the sets of edges and edge weights
    for i=1:length(nodeset1):
        if nodeset1(i) == Base:
        # Need to make this more functional ***

            fprintf(fid,'Base %d {''duration'': %d}\n', Nodeset2(i), Weight(i));
        elseif Nodeset2(i) == Base
            fprintf(fid,'%d Base {''duration'': %d}\n', Nodeset1(i), Weight(i));
        elseif Nodeset1(i) == Base && Nodeset2(i) == Base
            fprintf(fid,'Base Base {''duration'': %d}\n', Weight(i));
        elseif Nodeset1(i) == Base2
            fprintf(fid,'Base2 %d {''duration'': %d}\n', Nodeset2(i), Weight(i));
        elseif Nodeset2(i) == Base2
            fprintf(fid,'%d Base2 {''duration'': %d}\n', Nodeset1(i), Weight(i));
        elseif Nodeset1(i) == Base2 && Nodeset2(i) == Base2
            fprintf(fid,'Base2 Base2 {''duration'': %d}\n', Weight(i));
        elseif Nodeset1(i) == Base3
            fprintf(fid,'Base3 %d {''duration'': %d}\n', Nodeset2(i), Weight(i));
        elseif Nodeset2(i) == Base3
            fprintf(fid,'%d Base3 {''duration'': %d}\n', Nodeset1(i), Weight(i));
        elseif Nodeset1(i) == Base3 && Nodeset2(i) == Base3
            fprintf(fid,'Base3 Base3 {''duration'': %d}\n', Weight(i));
        else
            fprintf(fid,'%d %d {''duration'': %d}\n', Nodeset1(i), Nodeset2(i), Weight(i));

    f1.close()

if __name__ == '__main__':
    m = 6
    n = 6
    TS, obs_mat, state_mat = create_ts(m,n)
    pdb.set_trace()
    # try out the init state and obstacles functions
    init_state = 10
    obstacles = [(5,3),(4,3),(2,3),(2,4)]
    obs_mat = update_obs_mat(obs_mat, state_mat, obstacles, init_state)
    # Update the adjacency matrix
    TS = update_adj_mat(m, n, TS, obs_mat)
    # Now create the proper output .txt files
    path = '../data/ts_synth_6x6_test1.txt'
    bases = [(Base, 31), (Base2, 35), (Base3, 8)]
    disc = 0.5
    create_input_file(TS, path, bases, disc)
