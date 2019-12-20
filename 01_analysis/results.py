import numpy as np
import os
import imp

import economy
import policies

# dataFiles = os.listdir("tpsp_data/")

basePath = os.path.expanduser('~')
projectPath = basePath + "/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
dataPath = projectPath + "tpsp_data_mini/"
resultsPath = projectPath + "results/"

rcvPath = resultsPath + "rcv.csv"

# Economic Parameters
beta = np.genfromtxt(dataPath + 'beta.csv', delimiter=',')
theta = np.genfromtxt(dataPath + 'theta.csv', delimiter=',')
mu = np.genfromtxt(dataPath + 'mu.csv', delimiter=',')
nu = np.genfromtxt(dataPath + 'nu.csv', delimiter=',')

# Military Parameters
alpha_0 = 0  # force gained (lost) in offensive operations, regardless of distance
alpha_1 = -.1   # extra force gained (lost) for every log km traveled
gamma = 1
c_hat = .2  # relative cost of war

params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu, "alpha_0":alpha_0, "alpha_1":alpha_1, "c_hat":c_hat, "gamma":gamma}

# welfare weights
b = np.repeat(0, len(nu))

vars = {"b":b}

# Data
tau = np.genfromtxt(dataPath + 'tau.csv', delimiter=',')
Xcif = np.genfromtxt(dataPath + 'Xcif.csv', delimiter=',')
Y = np.genfromtxt(dataPath + 'Y.csv', delimiter=',')
Eq = np.genfromtxt(dataPath + 'Eq.csv', delimiter=',')
Ex = np.genfromtxt(dataPath + 'Ex.csv', delimiter=',')
r = np.genfromtxt(dataPath + 'r.csv', delimiter=',')
D = np.genfromtxt(dataPath + 'D.csv', delimiter=',')
ccodes = np.genfromtxt(dataPath + 'ccodes.csv', delimiter=',', dtype="str")
dists = np.genfromtxt(dataPath + 'mDists.csv', delimiter=',')
M = np.genfromtxt(dataPath + "milex.csv", delimiter=",")

M = M / np.min(M)  # normalize milex
W = np.log(dists+1)

N = len(Y)

E = Eq + Ex

data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M}  # Note: log distance

imp.reload(policies)
# imp.reload(economy)
pecmy = policies.policies(data, params, b, rcv_path=rcvPath)  # generate pecmy and rcv vals
# This will finish in 6.5 hours if each opt takes 75 seconds (13 governments)
# lots of nonsense negative numbers for Turkey's (11) best response for U.S. (12), replace with open/closed value instead

# br_11_12 = pecmy.br_war_ji(np.ones(pecmy.x_len), np.zeros(pecmy.N), 11, 12, full_opt=False)
# pecmy.rcv[0][11, 12] = pecmy.G_hat(br_11_12, np.zeros(pecmy.N), ids=np.array([11]))
# pecmy.rcv

# export regime change vals
# np.savetxt("results/rcv0.csv", pecmy.rcv[0], delimiter=",")
# np.savetxt("results/rcv1.csv", pecmy.rcv[1], delimiter=",")

# calculate free trade vals
tau_hat_ft = 1 / pecmy.ecmy.tau
ge_dict_ft = pecmy.ecmy.geq_solve(tau_hat_ft, np.ones(pecmy.N))
ge_x_ft = pecmy.ecmy.unwrap_ge_dict(ge_dict_ft)
G_hat_ft = pecmy.G_hat(ge_x_ft, np.zeros(pecmy.N))

# export free trade vals (b=0)
# np.savetxt("results/Ghatft.csv", G_hat_ft, delimiter=",")