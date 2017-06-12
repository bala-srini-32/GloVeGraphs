import os
import numpy as np 

for i in range(1,6):
	for j in np.arange(0.0,1,0.1):
		path = '../data/mu_'+str(j)+'_N_1000/'
		ip = path + 'network_run_'+str(i)+'.dat'
		op = path + 'embedding_run_'+str(i)+'.emb'
		cmd = 'python runner.py --input '+ip+' --output '+op+' --iterations 10 --learning-rate 0.1'
		print cmd
		os.system(cmd)