#!/usr/bin/python

import numpy as np
from subprocess import call
import os

mu = np.arange(0,1,0.05)

avg_degree = 20
max_degree = 50
n_runs_per_mu = 50
n_nodes_list = [1000]

for n_nodes in n_nodes_list:
	for i in range(len(mu)):
		print i
		cmd_mkdir = "mu_"+str(mu[i])+"_N_"+str(n_nodes)
		os.system("mkdir "+cmd_mkdir)
		for j in range(n_runs_per_mu):
			fn_call = "./benchmark"
			fn_params = " -N "+str(n_nodes)+" -k "+str(avg_degree)+" -maxk "+str(max_degree)+" -mu "+str(mu[i])
			os.system(fn_call+fn_params)
			cmd_mv_network = "mv network.dat "+cmd_mkdir+"/network_run_"+str(j+1)+".dat"
			cmd_mv_community = "mv community.dat "+cmd_mkdir+"/community_run_"+str(j+1)+".dat"
			os.system(cmd_mv_network)
			os.system(cmd_mv_community)

