'''
.. module:: plotting.py
   :synopsis: Functions to plot simulation trajectories on both 2D and 3D
              transition systems.

.. moduleauthor:: Ryan Peterson <pete9936@umn.edu.edu>

'''

import pdb, math
import matplotlib.pyplot as plt


def plot_2D(TS):
''' Create a 2D plot of the transition system '''
    plt.figure()
    # plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
    # plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
    # plt.xlim(0.5, 4.5)
    # Create a grided scatter plot with path highlighted
    cell_size = 0.5
    gridx = []
    gridy = []
    gridz = []
    for k in enumerate(h):
        for i in enumerate(m):
            for j in enumerate(n):
                gridx.append(j-cell_size)
                gridy.append(-i+cell_size)
                gridz.append(k-cell_size)
    # Give specific colors to the obstacles, init, and final locations
    obs_locx = []
    obs_locy = []
    obs_locz = []
    init_locx = []
    init_locy = []
    init_locz = []
    final_locx = []
    final_locy = []
    final_locz = []
    ROI_locx = []
    ROI_locy = []
    ROI_locz = []
    # Workaround for obstacle indexing
    obs_tmp = T.obs
    for k=1:h
        for i=1:m
            for j=1:n
                if ischar(obs_tmp{m*(k-1)+i,j})
                    if contains(obs_tmp{m*(k-1)+i,j},'o')
                        obs_tmp{m*(k-1)+i,j} = 3;
                    elif contains(obs_tmp{m*(k-1)+i,j},'r')
                        obs_tmp{m*(k-1)+i,j} = 2;
    for k=1:h
        for i=1:m
            for j=1:n
                if obs_tmp{m*(k-1)+i,j} == 3
                    obs_locx = [obs_locx, gridx(m*n*(k-1)+j)];
                    obs_locy = [obs_locy, gridy(T.states(m*(k-1)+i,j))];
                    obs_locz = [obs_locz, gridz(m*n*(k-1)+j)];
                elif obs_tmp{m*(k-1)+i,j} == 2
                    ROI_locx = [ROI_locx, gridx(m*n*(k-1)+j)];
                    ROI_locy = [ROI_locy, gridy(T.states(m*(k-1)+i,j))];
                    ROI_locz = [ROI_locz, gridz(m*n*(k-1)+j)];

    % Generate text for each node value
    a = Alphabet(1:m*n*h)'; c = cellstr(a); dx = 0.15; dy = 0.18;
    % Put bases in as well
    for i=1:m*n
        if c{i} == Base
            c{i} = 'Base';
        elseif c{i} == Base2
            c{i} = 'Base2';
        elseif c{i} == Base3
            c{i} = 'Base3';
        end
    end
    % Create set of transition arrows to plot
    % Use Nodeset1 and Nodeset2
    Nodeset1arr = [];
    Nodeset2arr = [];
    for i=1:n*m*h
        for j=1:n*m*h
            if T.adj(i,j) > 0
                Nodeset1arr = [Nodeset1arr; i];
                Nodeset2arr = [Nodeset2arr; j];

    p1 = [];
    p2 = [];
    dp = [];
    for i=1:length(Nodeset1)
        p1_x = gridx(Nodeset1arr(i));
        p1_y = gridy(Nodeset1arr(i));
        p1_z = gridz(Nodeset1arr(i));
        p2_x = gridx(Nodeset2arr(i));
        p2_y = gridy(Nodeset2arr(i));
        p2_z = gridz(Nodeset2arr(i));
        p1 = [p1; p1_x p1_y p1_z];
        p2 = [p2; p2_x p2_y p2_z];
        dp = [dp; (p2_x-p1_x) (p2_y-p1_y) (p2_z-p1_z)];

    % Generate figure for nominal path
    figure(3)
    scatter3(gridx,gridy,gridz,150,'filled','bs')
    % text(gridx+dx, gridy+dy, gridz, c);
    hold on
    title('Transition System (TS)')
    % set(get(gca,'title'),'Position',[3.0 -0.15])
    set(gcf,'Color','[1 1 1]')
    scatter3(obs_locx,obs_locy,obs_locz,200,'filled','ro')
    scatter3(ROI_locx,ROI_locy,ROI_locz,200,'filled','go')
    quiver3(p1(:,1),p1(:,2),p1(:,3),dp(:,1),dp(:,2),dp(:,3),1.5,'k')
    axis off
    hold off

    plt.show()






if __name__ == '__main__':
    pass
