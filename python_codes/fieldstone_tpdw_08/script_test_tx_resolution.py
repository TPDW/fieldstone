import os
import numpy as np
import matplotlib.pyplot as plt

nnx_list = [8,16,32,48,64]

traction_list=[]

for nnx in nnx_list:
	string_to_run = "python3 fieldstone.py " + str(nnx) + " " + str(nnx) + " 1"
	os.system(string_to_run)
	string_to_move = "mv tractions_x.ascii tractions_x_" + str(nnx) + ".ascii"
	os.system(string_to_move)
	traction = np.loadtxt("tractions_x_" + str(nnx) + ".ascii")

	traction_list.append(traction[int(nnx/2)])

plt.plot(nnx_list,traction_list)
plt.savefig("tractions_list.pdf")