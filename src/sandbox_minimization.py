license_text='''
    Sandbox module for automata minimization.
    Copyright (C) 2015-2016  Cristian Ioan Vasile <cvasile@bu.edu>
    Hybrid and Networked Systems (HyNeSs) Group, BU Robotics Lab,
    Boston University

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
'''
.. module:: sandbox_minimization.py
   :synopsis: Sandbox module for automata minimization.

.. moduleauthor:: Cristian Ioan Vasile <cvasile@bu.edu>

'''

import networkx as nx
from PartitionRefinement import PartitionRefinement

import twtl
from dfa import DFAType


def minimize(dfa):
    ''' Under construction :)
    Based on ...
    '''
    # refine partition of states by reversed neighborhoods
    partition = PartitionRefinement(dfa.g.nodes_iter())
    partition.refine(dfa.final)
    unrefined = dict([(id(p), p) for p in partition])
    while unrefined:
        part = unrefined.pop(unrefined.iterkeys().next())
        for symbol in dfa.alphabet:
            neighbors = set([v for v, _, d in dfa.g.in_edges_iter(part, data=True)
                                if symbol in d['input']])
            for new,old in partition.refine(neighbors):
                if id(old) in unrefined or len(new) < len(old):
                    unrefined[id(new)] = new
                else:
                    unrefined[id(old)] = old
    print dfa.g.number_of_nodes()
    print len(list(partition))
    # convert partition to DFA
    nx.condensation


if __name__ == '__main__':
    phi1 = '[H^2 R2 & [H^2 R1]^[0, 8]]^[0, 20] & [H^2 R2 & [H^2 R3]^[0, 14]]^[0, 70]'
#     phi1 = '[H^2 R2]^[0, 20]'
    _, dfa, bdd = twtl.translate(phi1, kind=DFAType.Normal, norm=True)
    
    dfa = minimize(dfa)
