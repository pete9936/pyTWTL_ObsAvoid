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

def create_ts(m, n, h):
    ''' function to create the initial grid for the transition system given
    the m x n grid'''
    # Initialize the states to their node names
    state_mat = np.arange(m*n*h).reshape((m*h,n))
    # Initialize an observation matrix with null observations
    obs_mat = np.zeros((m*h,n))
    adj_mat = np.zeros((m*n*h,m*n*h))
    # Populate the adjacency matrix with the initial states
    TS = update_adj_mat_3D(m, n, h, adj_mat, obs_mat)
    return TS, obs_mat, state_mat

def update_obs_mat(obs_mat, state_mat, m, obstacles = None, init_state = None):
    ''' update the observation matrix with known data so we can update the
    adjacency matrix and therefore the environment file '''
    if obstacles != None:
        for i in range(len(obstacles)):
            index1 = obstacles[i][0]
            index2 = obstacles[i][1]
            index3 = obstacles[i][2]
            obs_mat[m*index3+index1][index2] = 3
    if init_state != None:
        state_loc = np.argwhere(state_mat == init_state)
        row = state_loc[0][0]
        col = state_loc[0][1]
        obs_mat[row][col] = 1
    return obs_mat

def update_adj_mat(m, n, adj_mat, obs_mat):
    ''' Update the adjacency matrix given an observation matrix '''
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

def update_adj_mat_3D(m, n, h, adj_mat, obs_mat):
    ''' Update the adjacency matrix given an observation matrix '''
    # Breakdown of weights which are approximately Euclidean with a small
    # penalty given for change in altitude
    epsilon = 0.1
    stay_cost = 0.5
    card_cost = 1
    diag_cost = 1.414
    up_cost = 1 + epsilon
    up_card_cost = 1.414 + epsilon
    up_diag_cost = 1.55 + epsilon
    down_cost = 1 + epsilon
    down_card_cost = 1.414 + epsilon
    down_diag_cost = 1.55 + epsilon

    for k in range(h):
        for i in range(m):
            for j in range(n):
                if obs_mat[m*k+i][j] != 3:
                    diag_ind = n*m*k + n*i + j
                    adj_mat[diag_ind][diag_ind] = stay_cost
                    if h > 1:
                        if k == 0:
                            above_ind = n*m*(k+1) + n*i + j
                            if obs_mat[m*(k+1)+i][j] != 3:
                                adj_mat[diag_ind][above_ind] = up_cost
                            else:
                                adj_mat[diag_ind][above_ind] = 0
                        elif k == h-1:
                            below_ind = n*m*(k-1) + n*i + j
                            if obs_mat[m*(k-1)+i][j] != 3:
                                adj_mat[diag_ind][below_ind] = down_cost
                            else:
                                adj_mat[diag_ind][below_ind] = 0
                        else:
                            above_ind = n*m*(k+1) + n*i + j
                            below_ind = n*m*(k-1) + n*i + j
                            if obs_mat[m*(k+1)+i][j] != 3:
                                adj_mat[diag_ind][above_ind] = up_cost
                            else:
                                adj_mat[diag_ind][above_ind] = 0
                            if obs_mat[m*(k-1)+i][j] != 3:
                                adj_mat[diag_ind][below_ind] = down_cost
                            else:
                                adj_mat[diag_ind][below_ind] = 0
                    if j < n-1:
                        right_ind = n*m*k + n*i + j + 1
                        if h > 1:
                            if k == 0:
                                right_above_ind = n*m*(k+1) + n*i + j + 1
                                if obs_mat[m*(k+1)+i][j+1] != 3:
                                    adj_mat[diag_ind][right_above_ind] = up_card_cost
                                else:
                                    adj_mat[diag_ind][right_above_ind] = 0
                            elif k == h-1:
                                right_below_ind = n*m*(k-1) + n*i + j + 1
                                if obs_mat[m*(k-1)+i][j+1] != 3:
                                    adj_mat[diag_ind][right_below_ind] = down_card_cost
                                else:
                                    adj_mat[diag_ind][right_below_ind] = 0
                            else:
                                right_above_ind = n*m*(k+1) + n*i + j + 1
                                right_below_ind = n*m*(k-1) + n*i + j + 1
                                if obs_mat[m*(k+1)+i][j+1] != 3:
                                    adj_mat[diag_ind][right_above_ind] = up_card_cost
                                else:
                                    adj_mat[diag_ind][right_above_ind] = 0
                                if obs_mat[m*(k-1)+i][j+1] != 3:
                                    adj_mat[diag_ind][right_below_ind] = down_card_cost
                                else:
                                    adj_mat[diag_ind][right_below_ind] = 0
                        if obs_mat[m*k+i][j+1] != 3:
                            adj_mat[diag_ind][right_ind] = card_cost
                        else:
                            adj_mat[diag_ind][right_ind] = 0
                    if j > 0:
                        left_ind = n*m*k + n*i + j - 1
                        if h > 1:
                            if k == 0:
                                left_above_ind = n*m*(k+1) + n*i + j - 1
                                if obs_mat[m*(k+1)+i][j-1] != 3:
                                    adj_mat[diag_ind][left_above_ind] = up_card_cost
                                else:
                                    adj_mat[diag_ind][left_above_ind] = 0
                            elif k == h-1:
                                left_below_ind = n*m*(k-1) + n*i + j - 1
                                if obs_mat[m*(k-1)+i][j-1] != 3:
                                    adj_mat[diag_ind][left_below_ind] = down_card_cost
                                else:
                                    adj_mat[diag_ind][left_below_ind] = 0
                            else:
                                left_above_ind = n*m*(k+1) + n*i + j - 1
                                left_below_ind = n*m*(k-1) + n*i + j - 1
                                if obs_mat[m*(k+1)+i][j-1] != 3:
                                    adj_mat[diag_ind][left_above_ind] = up_card_cost
                                else:
                                    adj_mat[diag_ind][left_above_ind] = 0
                                if obs_mat[m*(k-1)+i][j-1] != 3:
                                    adj_mat[diag_ind][left_below_ind] = down_card_cost
                                else:
                                    adj_mat[diag_ind][left_below_ind] = 0
                        if obs_mat[m*k+i][j-1] != 3:
                            adj_mat[diag_ind][left_ind] = card_cost
                        else:
                            adj_mat[diag_ind][left_ind] = 0
                    if i > 0:
                        up_ind = n*m*k + n*(i-1) + j
                        if h > 1:
                            if k == 0:
                                up_above_ind = n*m*(k+1) + n*(i-1) + j
                                if obs_mat[m*(k+1)+i-1][j] != 3:
                                    adj_mat[diag_ind][up_above_ind] = up_card_cost
                                else:
                                    adj_mat[diag_ind][up_above_ind] = 0
                            elif k == h-1:
                                up_below_ind = n*m*(k-1) + n*(i-1) + j
                                if obs_mat[m*(k-1)+i-1][j] != 3:
                                    adj_mat[diag_ind][up_below_ind] = down_card_cost
                                else:
                                    adj_mat[diag_ind][up_below_ind] = 0
                            else:
                                up_above_ind = n*m*(k+1) + n*(i-1) + j
                                up_below_ind = n*m*(k-1) + n*(i-1) + j
                                if obs_mat[m*(k+1)+i-1][j] != 3:
                                    adj_mat[diag_ind][up_above_ind] = up_card_cost
                                else:
                                    adj_mat[diag_ind][up_above_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j] != 3:
                                    adj_mat[diag_ind][up_below_ind] = down_card_cost
                                else:
                                    adj_mat[diag_ind][up_below_ind] = 0
                        if obs_mat[m*k+i-1][j] != 3:
                            adj_mat[diag_ind][up_ind] = card_cost
                        else:
                            adj_mat[diag_ind][up_ind] = 0
                    if i < m-1:
                        down_ind = n*m*k + n*(i+1) + j
                        if h > 1:
                            if k == 0:
                                down_above_ind = n*m*(k+1) + n*(i+1) + j
                                if obs_mat[m*(k+1)+i+1][j] != 3:
                                    adj_mat[diag_ind][down_above_ind] = up_card_cost
                                else:
                                    adj_mat[diag_ind][down_above_ind] = 0
                            elif k == h-1:
                                down_below_ind = n*m*(k-1) + n*(i+1) + j
                                if obs_mat[m*(k-1)+i+1][j] != 3:
                                    adj_mat[diag_ind][down_below_ind] = down_card_cost
                                else:
                                    adj_mat[diag_ind][down_below_ind] = 0
                            else:
                                down_above_ind = n*m*(k+1) + n*(i+1) + j
                                down_below_ind = n*m*(k-1) + n*(i+1) + j
                                if obs_mat[m*(k+1)+i+1][j] != 3:
                                    adj_mat[diag_ind][down_above_ind] = up_card_cost
                                else:
                                    adj_mat[diag_ind][down_above_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j] != 3:
                                    adj_mat[diag_ind][down_below_ind] = down_card_cost
                                else:
                                    adj_mat[diag_ind][down_below_ind] = 0
                        if obs_mat[m*k+i+1][j] != 3:
                            adj_mat[diag_ind][down_ind] = card_cost
                        else:
                            adj_mat[diag_ind][down_ind] = 0
                    # Now perform the diagonal indexing
                    if i == 0 and j == 0: # upper left
                        SE_index = n*m*k + n*(i+1) + j + 1
                        if h > 1:
                            if k == 0:
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                if obs_mat[m*(k+1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_above_ind] = 0
                            elif k == h-1:
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                if obs_mat[m*(k-1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_below_ind] = 0
                            else:
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                if obs_mat[m*(k+1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_above_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_below_ind] = 0
                        if obs_mat[m*k+i+1][j+1] != 3:
                            adj_mat[diag_ind][SE_index] = diag_cost
                        else:
                            adj_mat[diag_ind][SE_index] = 0
                    if i > 0 and i < m-1 and j == 0: # left column (not corner)
                        NE_index = n*m*k + n*(i-1) + j + 1
                        SE_index = n*m*k + n*(i+1) + j + 1
                        if h > 1:
                            if k == 0:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                if obs_mat[m*(k+1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_above_ind] = 0
                            elif k == h-1:
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                if obs_mat[m*(k-1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_below_ind] = 0
                            else:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                if obs_mat[m*(k+1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_above_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_below_ind] = 0
                        if obs_mat[m*k+i-1][j+1] != 3:
                            adj_mat[diag_ind][NE_index] = diag_cost
                        else:
                            adj_mat[diag_ind][NE_index] = 0
                        if obs_mat[m*k+i+1][j+1] != 3:
                            adj_mat[diag_ind][SE_index] = diag_cost
                        else:
                            adj_mat[diag_ind][SE_index] = 0
                    if i == m-1 and j == 0:  # lower left
                        NE_index = n*m*k + n*(i-1) + j + 1
                        if h > 1:
                            if k == 0:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                if obs_mat[m*(k+1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_above_ind] = 0
                            elif k == h-1:
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                if obs_mat[m*(k-1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_below_ind] = 0
                            else:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                if obs_mat[m*(k+1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_above_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_below_ind] = 0
                        if obs_mat[m*k+i-1][j+1] != 3:
                            adj_mat[diag_ind][NE_index] = diag_cost
                        else:
                            adj_mat[diag_ind][NE_index] = 0
                    if i == 0 and j < n-1 and j > 0:  # upper row (not corner)
                        SW_index = n*m*k + n*(i+1) + j - 1
                        SE_index = n*m*k + n*(i+1) + j + 1
                        if h > 1:
                            if k == 0:
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                if obs_mat[m*(k+1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_above_ind] = 0
                            elif k == h-1:
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                if obs_mat[m*(k-1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_below_ind] = 0
                            else:
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                if obs_mat[m*(k+1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_above_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_below_ind] = 0
                        if obs_mat[m*k+i+1][j-1] != 3:
                            adj_mat[diag_ind][SW_index] = diag_cost
                        else:
                            adj_mat[diag_ind][SW_index] = 0
                        if obs_mat[m*k+i+1][j+1] != 3:
                            adj_mat[diag_ind][SE_index] = diag_cost
                        else:
                            adj_mat[diag_ind][SE_index] = 0
                    if i == 0 and j == n-1: # upper right
                        SW_index = n*m*k + n*(i+1) + j - 1
                        if h > 1:
                            if k == 0:
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                if obs_mat[m*(k+1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_above_ind] = 0
                            elif k == h-1:
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                if obs_mat[m*(k-1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_below_ind] = 0
                            else:
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                if obs_mat[m*(k+1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_above_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_below_ind] = 0
                        if obs_mat[m*k+i+1][j-1] != 3:
                            adj_mat[diag_ind][SW_index] = diag_cost
                        else:
                            adj_mat[diag_ind][SW_index] = 0
                    if i > 0 and j == n-1 and i < m-1:  # right column (not corner)
                        NW_index = n*m*k + n*(i-1) + j - 1
                        SW_index = n*m*k + n*(i+1) + j - 1
                        if h > 1:
                            if k == 0:
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                if obs_mat[m*(k+1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_above_ind] = 0
                            elif k == h-1:
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                if obs_mat[m*(k-1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_below_ind] = 0
                            else:
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                if obs_mat[m*(k+1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_above_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_below_ind] = 0
                        if obs_mat[m*k+i-1][j-1] != 3:
                            adj_mat[diag_ind][NW_index] = diag_cost
                        else:
                            adj_mat[diag_ind][NW_index] = 0
                        if obs_mat[m*k+i+1][j-1] != 3:
                            adj_mat[diag_ind][SW_index] = diag_cost
                        else:
                            adj_mat[diag_ind][SW_index] = 0
                    if i == m-1 and j == n-1:  # bottom right
                        NW_index = n*m*k + n*(i-1) + j - 1
                        if h > 1:
                            if k == 0:
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                if obs_mat[m*(k+1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_above_ind] = 0
                            elif k == h-1:
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                if obs_mat[m*(k-1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_below_ind] = 0
                            else:
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                if obs_mat[m*(k+1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_above_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_below_ind] = 0
                        if obs_mat[m*k+i-1][j-1] != 3:
                            adj_mat[diag_ind][NW_index] = diag_cost
                        else:
                            adj_mat[diag_ind][NW_index] = 0
                    if i == m-1 and j > 0 and j < n-1:  # bottom row (not corner)
                        NW_index = n*m*k + n*(i-1) + j - 1
                        NE_index = n*m*k + n*(i-1) + j + 1
                        if h > 1:
                            if k == 0:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                if obs_mat[m*(k+1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_above_ind] = 0
                                if obs_mat[m*(k+1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_above_ind] = 0
                            elif k == h-1:
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                if obs_mat[m*(k-1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_below_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_below_ind] = 0
                            else:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                if obs_mat[m*(k+1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_above_ind] = 0
                                if obs_mat[m*(k+1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_above_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_below_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_below_ind] = 0
                        if obs_mat[m*k+i-1][j-1] != 3:
                            adj_mat[diag_ind][NW_index] = diag_cost
                        else:
                            adj_mat[diag_ind][NW_index] = 0
                        if obs_mat[m*k+i-1][j+1] != 3:
                            adj_mat[diag_ind][NE_index] = diag_cost
                        else:
                            adj_mat[diag_ind][NE_index] = 0
                    if i > 0 and i < m-1 and j > 0 and j < n-1: # all middle nodes
                        NW_index = n*m*k + n*(i-1) + j - 1
                        NE_index = n*m*k + n*(i-1) + j + 1
                        SW_index = n*m*k + n*(i+1) + j - 1
                        SE_index = n*m*k + n*(i+1) + j + 1
                        if h > 1:
                            if k == 0:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                if obs_mat[m*(k+1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_above_ind] = 0
                                if obs_mat[m*(k+1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_above_ind] = 0
                            elif k == h-1:
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                if obs_mat[m*(k-1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_below_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_below_ind] = 0
                            else:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                if obs_mat[m*(k+1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_above_ind] = 0
                                if obs_mat[m*(k+1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_above_ind] = 0
                                if obs_mat[m*(k+1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_above_ind] = up_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_above_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j+1] != 3:
                                    adj_mat[diag_ind][NE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NE_below_ind] = 0
                                if obs_mat[m*(k-1)+i-1][j-1] != 3:
                                    adj_mat[diag_ind][NW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][NW_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j+1] != 3:
                                    adj_mat[diag_ind][SE_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SE_below_ind] = 0
                                if obs_mat[m*(k-1)+i+1][j-1] != 3:
                                    adj_mat[diag_ind][SW_below_ind] = down_diag_cost
                                else:
                                    adj_mat[diag_ind][SW_below_ind] = 0
                        if obs_mat[m*k+i-1][j-1] != 3:
                            adj_mat[diag_ind][NW_index] = diag_cost
                        else:
                            adj_mat[diag_ind][NW_index] = 0
                        if obs_mat[m*k+i-1][j+1] != 3:
                            adj_mat[diag_ind][NE_index] = diag_cost
                        else:
                            adj_mat[diag_ind][NE_index] = 0
                        if obs_mat[m*k+i+1][j-1] != 3:
                            adj_mat[diag_ind][SW_index] = diag_cost
                        else:
                            adj_mat[diag_ind][SW_index] = 0
                        if obs_mat[m*k+i+1][j+1] != 3:
                            adj_mat[diag_ind][SE_index] = diag_cost
                        else:
                            adj_mat[diag_ind][SE_index] = 0
                else:
                    # this indicates the region is an obstacle
                    diag_ind = n*m*k + n*i + j
                    adj_mat[diag_ind][diag_ind] = 0
                    if h > 1:
                        if k == 0:
                            above_ind = n*m*(k+1) + n*i + j
                            adj_mat[diag_ind][above_ind] = 0
                        elif k == h-1:
                            below_ind = n*m*(k-1) + n*i + j
                            adj_mat[diag_ind][below_ind] = 0
                        else:
                            above_ind = n*m*(k+1) + n*i + j
                            below_ind = n*m*(k-1) + n*i + j
                            adj_mat[diag_ind][above_ind] = 0
                            adj_mat[diag_ind][below_ind] = 0
                    if j < n-1:
                        if h > 1:
                            if k == 0:
                                right_above_ind = n*m*(k+1) + n*i + j + 1
                                adj_mat[diag_ind][right_above_ind] = 0
                            elif k == h-1:
                                right_below_ind = n*m*(k-1) + n*i + j + 1
                                adj_mat[diag_ind][right_below_ind] = 0
                            else:
                                right_above_ind = n*m*(k+1) + n*i + j + 1
                                right_below_ind = n*m*(k-1) + n*i + j + 1
                                adj_mat[diag_ind][right_above_ind] = 0
                                adj_mat[diag_ind][right_below_ind] = 0
                        right_ind = n*m*k + n*i + j + 1
                        adj_mat[diag_ind][right_ind] = 0
                    if j > 0:
                        if h > 1:
                            if k == 0:
                                left_above_ind = n*m*(k+1) + n*i + j - 1
                                adj_mat[diag_ind][left_above_ind] = 0
                            elif k == h-1:
                                left_below_ind = n*m*(k-1) + n*i + j - 1
                                adj_mat[diag_ind][left_below_ind] = 0
                            else:
                                left_above_ind = n*m*(k+1) + n*i + j - 1
                                left_below_ind = n*m*(k-1) + n*i + j - 1
                                adj_mat[diag_ind][left_above_ind] = 0
                                adj_mat[diag_ind][left_below_ind] = 0
                        left_ind = n*m*k + n*i + j - 1
                        adj_mat[diag_ind][left_ind] = 0
                    if i > 0:
                        if h > 1:
                            if k == 0:
                                up_above_ind = n*m*(k+1) + n*(i-1) + j
                                adj_mat[diag_ind][up_above_ind] = 0
                            elif k == h-1:
                                up_below_ind = n*m*(k-1) + n*(i-1) + j
                                adj_mat[diag_ind][up_below_ind] = 0
                            else:
                                up_above_ind = n*m*(k+1) + n*(i-1) + j
                                up_below_ind = n*m*(k-1) + n*(i-1) + j
                                adj_mat[diag_ind][up_above_ind] = 0
                                adj_mat[diag_ind][up_below_ind] = 0
                        up_ind = n*m*k + n*(i-1) + j
                        adj_mat[diag_ind][up_ind] = 0
                    if i < m-1:
                        if h > 1:
                            if k == 0:
                                down_above_ind = n*m*(k+1) + n*(i+1) + j
                                adj_mat[diag_ind][down_above_ind] = 0
                            elif k == h-1:
                                down_below_ind = n*m*(k-1) + n*(i+1) + j
                                adj_mat[diag_ind][down_below_ind] = 0
                            else:
                                down_above_ind = n*m*(k+1) + n*(i+1) + j
                                down_below_ind = n*m*(k-1) + n*(i+1) + j
                                adj_mat[diag_ind][down_above_ind] = 0
                                adj_mat[diag_ind][down_below_ind] = 0
                        down_ind = n*m*k + n*(i+1) + j
                        adj_mat[diag_ind][down_ind] = 0
                    if i == 0 and j == 0: # upper left
                        if h > 1:
                            if k == 0:
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                adj_mat[diag_ind][SE_above_ind] = 0
                            elif k == h-1:
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                adj_mat[diag_ind][SE_below_ind] = 0
                            else:
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                adj_mat[diag_ind][SE_above_ind] = 0
                                adj_mat[diag_ind][SE_below_ind] = 0
                        SE_index = n*m*k + n*(i+1) + j + 1
                        adj_mat[diag_ind][SE_index] = 0
                    if i > 0 and i < m-1 and j == 0: # left column (not corner)
                        if h > 1:
                            if k == 0:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                adj_mat[diag_ind][NE_above_ind] = 0
                                adj_mat[diag_ind][SE_above_ind] = 0
                            elif k == h-1:
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                adj_mat[diag_ind][NE_below_ind] = 0
                                adj_mat[diag_ind][SE_below_ind] = 0
                            else:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                adj_mat[diag_ind][NE_above_ind] = 0
                                adj_mat[diag_ind][SE_above_ind] = 0
                                adj_mat[diag_ind][NE_below_ind] = 0
                                adj_mat[diag_ind][SE_below_ind] = 0
                        NE_index = n*m*k + n*(i-1) + j + 1
                        SE_index = n*m*k + n*(i+1) + j + 1
                        adj_mat[diag_ind][NE_index] = 0
                        adj_mat[diag_ind][SE_index] = 0
                    if i == m-1 and j == 0:  # lower left
                        if h > 1:
                            if k == 0:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                adj_mat[diag_ind][NE_above_ind] = 0
                            elif k == h-1:
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                adj_mat[diag_ind][NE_below_ind] = 0
                            else:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                adj_mat[diag_ind][NE_above_ind] = 0
                                adj_mat[diag_ind][NE_below_ind] = 0
                        NE_index = n*m*k + n*(i-1) + j + 1
                        adj_mat[diag_ind][NE_index] = 0
                    if i == 0 and j < n-1 and j > 0:  # upper row (not corner)
                        if h > 1:
                            if k == 0:
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][SE_above_ind] = 0
                                adj_mat[diag_ind][SW_above_ind] = 0
                            elif k == h-1:
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][SE_below_ind] = 0
                                adj_mat[diag_ind][SW_below_ind] = 0
                            else:
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][SE_above_ind] = 0
                                adj_mat[diag_ind][SW_above_ind] = 0
                                adj_mat[diag_ind][SE_below_ind] = 0
                                adj_mat[diag_ind][SW_below_ind] = 0
                        SW_index = n*m*k + n*(i+1) + j - 1
                        SE_index = n*m*k + n*(i+1) + j + 1
                        adj_mat[diag_ind][SW_index] = 0
                        adj_mat[diag_ind][SE_index] = 0
                    if i == 0 and j == n-1: # upper right
                        if h > 1:
                            if k == 0:
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][SW_above_ind] = 0
                            elif k == h-1:
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][SW_below_ind] = 0
                            else:
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][SW_above_ind] = 0
                                adj_mat[diag_ind][SW_below_ind] = 0
                        SW_index = n*m*k + n*(i+1) + j - 1
                        adj_mat[diag_ind][SW_index] = 0
                    if i > 0 and j == n-1 and i < m-1:  # right column (not corner)
                        if h > 1:
                            if k == 0:
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][NW_above_ind] = 0
                                adj_mat[diag_ind][SW_above_ind] = 0
                            elif k == h-1:
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][NW_below_ind] = 0
                                adj_mat[diag_ind][SW_below_ind] = 0
                            else:
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][NW_above_ind] = 0
                                adj_mat[diag_ind][SW_above_ind] = 0
                                adj_mat[diag_ind][NW_below_ind] = 0
                                adj_mat[diag_ind][SW_below_ind] = 0
                        NW_index = n*m*k + n*(i-1) + j - 1
                        SW_index = n*m*k + n*(i+1) + j - 1
                        adj_mat[diag_ind][NW_index] = 0
                        adj_mat[diag_ind][SW_index] = 0
                    if i == m-1 and j == n-1:  # bottom right
                        if h > 1:
                            if k == 0:
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                adj_mat[diag_ind][NW_above_ind] = 0
                            elif k == h-1:
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                adj_mat[diag_ind][NW_below_ind] = 0
                            else:
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                adj_mat[diag_ind][NW_above_ind] = 0
                                adj_mat[diag_ind][NW_below_ind] = 0
                        NW_index = n*m*k + n*(i-1) + j - 1
                        adj_mat[diag_ind][NW_index] = 0
                    if i == m-1 and j > 0 and j < n-1:  # bottom row (not corner)
                        if h > 1:
                            if k == 0:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                adj_mat[diag_ind][NE_above_ind] = 0
                                adj_mat[diag_ind][NW_above_ind] = 0
                            elif k == h-1:
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                adj_mat[diag_ind][NE_below_ind] = 0
                                adj_mat[diag_ind][NW_below_ind] = 0
                            else:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                adj_mat[diag_ind][NE_above_ind] = 0
                                adj_mat[diag_ind][NW_above_ind] = 0
                                adj_mat[diag_ind][NE_below_ind] = 0
                                adj_mat[diag_ind][NW_below_ind] = 0
                        NW_index = n*m*k + n*(i-1) + j - 1
                        NE_index = n*m*k + n*(i-1) + j + 1
                        adj_mat[diag_ind][NW_index] = 0
                        adj_mat[diag_ind][NE_index] = 0
                    if i > 0 and i < m-1 and j > 0 and j < n-1: # all middle nodes
                        if h > 1:
                            if k == 0:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][NE_above_ind] = 0
                                adj_mat[diag_ind][NW_above_ind] = 0
                                adj_mat[diag_ind][SE_above_ind] = 0
                                adj_mat[diag_ind][SW_above_ind] = 0
                            elif k == h-1:
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][NE_below_ind] = 0
                                adj_mat[diag_ind][NW_below_ind] = 0
                                adj_mat[diag_ind][SE_below_ind] = 0
                                adj_mat[diag_ind][SW_below_ind] = 0
                            else:
                                NE_above_ind = n*m*(k+1) + n*(i-1) + j + 1
                                NW_above_ind = n*m*(k+1) + n*(i-1) + j - 1
                                NE_below_ind = n*m*(k-1) + n*(i-1) + j + 1
                                NW_below_ind = n*m*(k-1) + n*(i-1) + j - 1
                                SE_above_ind = n*m*(k+1) + n*(i+1) + j + 1
                                SW_above_ind = n*m*(k+1) + n*(i+1) + j - 1
                                SE_below_ind = n*m*(k-1) + n*(i+1) + j + 1
                                SW_below_ind = n*m*(k-1) + n*(i+1) + j - 1
                                adj_mat[diag_ind][NE_above_ind] = 0
                                adj_mat[diag_ind][NW_above_ind] = 0
                                adj_mat[diag_ind][NE_below_ind] = 0
                                adj_mat[diag_ind][NW_below_ind] = 0
                                adj_mat[diag_ind][SE_above_ind] = 0
                                adj_mat[diag_ind][SW_above_ind] = 0
                                adj_mat[diag_ind][SE_below_ind] = 0
                                adj_mat[diag_ind][SW_below_ind] = 0
                        NW_index = n*m*k + n*(i-1) + j - 1
                        NE_index = n*m*k + n*(i-1) + j + 1
                        SW_index = n*m*k + n*(i+1) + j - 1
                        SE_index = n*m*k + n*(i+1) + j + 1
                        adj_mat[diag_ind][NW_index] = 0
                        adj_mat[diag_ind][NE_index] = 0
                        adj_mat[diag_ind][SW_index] = 0
                        adj_mat[diag_ind][SE_index] = 0
    return adj_mat


def create_input_file(adj_mat, state_mat, obs_mat, path, bases, disc, m, n, h, iter):
    ''' Given the adjacency matrix this creates the proper text file for the
        execution. '''
    nodeset1 = []
    nodeset2 = []
    weight = []
    for i in range(m*n*h):
        for j in range(m*n*h):
            if adj_mat[i][j] > 0:
                nodeset1.append(i)
                nodeset2.append(j)
                weight.append(adj_mat[i][j])
    with open(path, 'w+') as f1:
        # Print file header and initial position
        f1.write('name Simple DTS\n')
        if iter == 0:
            f1.write('init {\'Base\':1}\n')
        else:
            base_ind = iter+1
            f1.write('init {\'Base%d\':1}\n' % base_ind)
        f1.write(';\n')

        # Publish the sets of nodes
        for k in range(h):
            for i in range(m):
                for j in range(n):
                    x = disc*m/2 - disc/2 - i*disc
                    y = disc*n/2 - disc/2 - j*disc
                    z = 0.4 + k*0.3  # this is due to our lab testing environment
                    if obs_mat[m*k+i][j] == 3:
                        continue
                    elif state_mat[m*k+i][j] in bases:
                        for key in bases:
                            if key == state_mat[i][j]:
                                if bases[key] == 'Base':
                                    f1.write('%s {\'prop\': set(), \'position\': (%1.2f, %1.2f, %1.2f)}\n' % (bases[key], x, y, z))
                                else:
                                    f1.write('%s {\'prop\':{\'%s\'}, \'position\': (%1.2f, %1.2f, %1.2f)}\n' % (bases[key], bases[key], x, y, z))
                    else:
                        f1.write('r%d {\'prop\':{\'r%d\'}, \'position\': (%1.2f, %1.2f, %1.2f)}\n'\
                                                % (state_mat[m*k+i][j], state_mat[m*k+i][j], x, y, z))
        f1.write(';\n')

        # Publish the sets of edges and edge weights
        for i in range(len(nodeset1)):
            if nodeset1[i] in bases:
                if nodeset2[i] in bases:
                    for key1 in bases:
                        if nodeset1[i] == key1:
                            for key2 in bases:
                                if nodeset2[i] == key2:
                                    f1.write('%s %s {\'duration\': %d, \'edge_weight\': %f}\n'\
                                            % (bases[key1], bases[key2], 1.0, weight[i]))
                else:
                    for key1 in bases:
                        if nodeset1[i] == key1:
                            f1.write('%s r%d {\'duration\': %d, \'edge_weight\': %f}\n' % (bases[key1], nodeset2[i], 1.0, weight[i]))
            elif nodeset2[i] in bases:
                for key2 in bases:
                    if nodeset2[i] == key2:
                        f1.write('r%d %s {\'duration\': %d, \'edge_weight\': %f}\n' % (nodeset1[i], bases[key2], 1.0, weight[i]))
            else:
                f1.write('r%d r%d {\'duration\': %d, \'edge_weight\': %f}\n' % (nodeset1[i], nodeset2[i], 1.0, weight[i]))
    # finished writing to file
    f1.close()


if __name__ == '__main__':
    m = 6
    n = 6
    h = 3
    TS, obs_mat, state_mat = create_ts(m,n,h)
    # try out the init state and obstacles functions
    init_state = [30, 34, 1, 4, 6]
    obstacles = [(3,2,0),(3,2,1),(3,2,2),(2,5,0),(5,3,0),(5,3,1)] # (row,column,altitude/height)
    paths = ['../data/ts_6x6x3_5Ag_1.txt', '../data/ts_6x6x3_5Ag_2.txt', '../data/ts_6x6x3_5Ag_3.txt', \
            '../data/ts_6x6x3_5Ag_4.txt', '../data/ts_6x6x3_5Ag_5.txt']
    bases = {30: 'Base', 34: 'Base2', 1: 'Base3', 4: 'Base4', 6: 'Base5'}
    disc = 0.43
    for i in range(len(init_state)):
        obs_mat = update_obs_mat(obs_mat, state_mat, m, obstacles, init_state[i])
        # Update the adjacency matrix
        TS = update_adj_mat_3D(m, n, h, TS, obs_mat)
        # Now create the proper output .txt files
        create_input_file(TS, state_mat, obs_mat, paths[i], bases, disc, m, n, h, i)
