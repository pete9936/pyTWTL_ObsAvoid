# This file includes the function which updates the edge weights of the DFA in
# number of ways. These different methods include:
#
#  1. Scaling the edge weights due to obstacles according to a vector potential field
#     which acts to "repel" an agent from an obstacle.
#  2. Decreasing the edge weights approaching the goal region as to attract the
#     agent to the goal postiion.
#  3. Perturb the edge weights if deadlock is detected according to the type of
#     deadlock detected. This follows schemes similar to those of traffic rules.


import logging
import copy

import networkx as nx
import re
import itertools
from model import Model

def read_from_file(self, path):
	"""
	Reads a LOMAP Ts object from a given file
	"""

	##
	# Open and read the file
	##
	try:
		with open(path, 'r') as f:
			lines = f.read().splitlines()
	except:
		raise FileError('Problem opening file %s for reading.' % path)
	line_cnt = 0;

	##
	# Part-1: Model attributes
	##

	# Name of the model
	try:
		m = re.match(r'name (.*$)', lines[line_cnt])
		self.name = m.group(1)
		line_cnt += 1
	except:
		raise FileError("Line 1 must be of the form: 'name name_of_the_transition_system', read: '%s'." % lines[line_cnt])

	# Initial distribution of the model
	# A dictionary of the form {'state_label': probability}
	try:
		m = re.match(r'init (.*$)', lines[line_cnt])
		self.init = eval(m.group(1))
		line_cnt += 1
	except:
		raise FileError("Line 2 must give the initial distribution of the form {'state_label': 1}, read: '%s'." % lines[line_cnt])

	# Single state for det-ts, multiple states w/ prob. 1 for nondet-ts
	for init in self.init:
		if self.init[init] != 1:
			raise FileError('Initial probability of state %s cannot be %f in a transition system.' % (init, self.init[init]))
	# RP 9/5: What is meant by initial probability here?

	##
	# End of part-1
	##

	if(lines[line_cnt] != ';'):
		raise FileError("Expected ';' after model attributes, read: '%s'." % (line_cnt, lines[line_cnt]))
	line_cnt += 1

	##
	# Part-2: State attributes
	##

	# We store state attributes in a dict keyed by states as
	# we haven't defined them yet
	# RP 9/5: This can act as our update as well
	state_attr = dict();
	try:
		while(line_cnt < len(lines) and lines[line_cnt] != ';'):
			m = re.search('(\S*) (.*)$', lines[line_cnt]);
			exec("state_attr['%s'] = %s" % (m.group(1),m.group(2)));
			line_cnt += 1
		line_cnt+=1
	except:
		raise FileError('Problem parsing state attributes.')

	##
	# Part-3: Edge list with attributes
	##
	try:
		self.g = nx.parse_edgelist(lines[line_cnt:], comments='#', create_using=nx.MultiDiGraph())
	except:
		raise FileError('Problem parsing definitions of the transitions.')

	# Add state attributes to nodes of the graph
	try:
		for node in state_attr.keys():
			# Reset label of the node
			self.g.node[node]['label'] = node
			for key in state_attr[node].keys():
				# Copy defined attributes to the node in the graph
				# This is a shallow copy, we don't touch state_attr[node][key] afterwards
				self.g.node[node][key] = state_attr[node][key]
				# Define custom node label
				self.g.node[node]['label'] = r'%s\n%s: %s' % (self.g.node[node]['label'], key, state_attr[node][key])
	except:
		raise FileError('Problem setting state attributes.')



'''
def update_edge_weights(edge_list):
    '''Updates edge weights in order to find a new control policy.'''

    # copy attributes
    edge_list_new = copy.deepcopy(edge_list)

    # edge_list_new.weight = edge_list.weight
    # edge_list_new.label = edge_list.label

    for edge, label in edge_list:
        if label == 'update':
            new_weight = potential_field(edge)
            edge = new_weight
    return edge_list

def potential_field(edge):
    '''Update edge weight using a potential field function.'''
    pass
    return weight
'''
