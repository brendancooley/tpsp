### SETUP ###

import os
import autograd.numpy as np
from scipy import optimize
import imp
import timeit
import time
import autograd as ag
import statsmodels.api as sm

import economy
# import economyOld
import policies
import helpers


### IMPORT DATA ###

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
# alpha_0 = 1  # force gained (lost) in offensive operations, regardless of distance
alpha_1 = .1   # extra force gained (lost) for every log km traveled
gamma = 1
c_hat = .2  # relative cost of war

# params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu, "alpha_0":alpha_0, "alpha_1":alpha_1, "c_hat":c_hat, "gamma":gamma}
params = {"beta":beta,"theta":theta,"mu":mu,"nu":nu, "alpha_1":alpha_1, "c_hat":c_hat, "gamma":gamma}

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

M = M / np.max(M)  # normalize milex
W = np.log(dists+1)

N = len(Y)

E = Eq + Ex

data = {"tau":tau,"Xcif":Xcif,"Y":Y,"E":E,"r":r,"D":D,"W":W,"M":M}  # Note: log distance (plus 1)

theta_dict = dict()
# theta_dict["b"] = b
theta_dict["alpha"] = .3
theta_dict["c_hat"] = .3
theta_dict["sigma_epsilon"] = .1
theta_dict["gamma"] = 1

theta_vec = [i for i in theta_dict.values()]

np.append(theta_vec, b)
print(str(theta_dict))

### TEST B ESTIMATOR ###

m = M / np.ones_like(tau)
m = m.T

m_diag = np.diagonal(m)
m_frac = m / m_diag
# m = np.diag(M)


sigma_epsilon = .1
epsilon = np.reshape(np.random.normal(0, sigma_epsilon, N ** 2), (N, N))
np.fill_diagonal(epsilon, 0)

imp.reload(policies)
pecmy = policies.policies(data, params, b, rcv_path=rcvPath)
b_init = np.repeat(.5, N)
out = pecmy.est_loop(b_init, theta_dict)

# pecmy.est_theta(b_init, m, theta_dict)
# pecmy.rcv[1]
# pecmy.est_b_grid(b_init, m, theta_dict, epsilon)
# np.any(np.array([-1, 1]) < 0)

Theta = []
b = []
b.append(b_init)
b.append(b_init)

tau
pecmy.rcv[.5]

id = 3
b_test = np.repeat(.5, pecmy.N)
b_test[id] = .7
wv_m = pecmy.war_vals(b_test, m, theta_dict, epsilon) # calculate war values
ids_j = np.delete(np.arange(pecmy.N), id)
wv_m_i = wv_m[:,id][ids_j]
# wv_m
# wv_m_i

ge_x_sv = np.ones(pecmy.x_len)
test = pecmy.br(ge_x_sv, b_test, m, wv_m_i, id)
test

np.random.choice([1, -1])
ccodes


pecmy.G_hat(test, b_test)







out = pecmy.est_theta(b_init, m, theta_dict)

out.summary()
out.params
out.bse
out.resid
out.fittedvalues

estars = pecmy.epsilon_star(b_init, m, theta_dict, W)
weights = pecmy.weights(estars, sigma_epsilon)

pecmy.rcv[0]
pecmy.rhoM(theta_dict, epsilon) # testing new rho function
pecmy.rhoM(theta_dict, 0)
# positive shocks give better war performance
# pecmy.est_b_i_grid(0, b_init, m, theta_dict, epsilon)

# Japan's BR is getting screwed up sometimes with positive m
# Looks like this is because ROW is extremely threatening to Japan at trial values, hard to satisfy constraint
    # bumping up alpha_0 seems to fix the problem
id = 2
b_test = np.repeat(.4, pecmy.N)
# starting values
tau_hat_nft = 1.1 / pecmy.ecmy.tau
np.fill_diagonal(tau_hat_nft, 1)
# tau_hat_nft * pecmy.ecmy.tau
ge_x_sv = np.ones(pecmy.x_len)
ge_dict = pecmy.ecmy.rewrap_ge_dict(ge_x_sv)
tau_hat_sv = ge_dict["tau_hat"]
tau_hat_sv[id] = tau_hat_nft[id] # start slightly above free trade
ge_dict_sv = pecmy.ecmy.geq_solve(tau_hat_sv, np.ones(pecmy.N))
ge_x_sv = pecmy.ecmy.unwrap_ge_dict(ge_dict_sv)



test = pecmy.br(ge_x_sv, b_test, m, wv_m_i, id)


# Q: is tau_ij invariant to coerion from k?
id = 0

m = np.diag(M)
m_prime = np.copy(m)
m_prime[5, 0] = m[5,5]
m_prime[5, 5] = 0  # U.S. spends all effort coercing China


wv_m = pecmy.war_vals(b_init, m, theta_dict, epsilon) # calculate war values
ids_j = np.delete(np.arange(pecmy.N), id)
wv_m_i = wv_m[:,id][ids_j]


wv_m_prime = pecmy.war_vals(b_init, m_prime, theta_dict, epsilon) # calculate war values
wv_m_prime_i = wv_m_prime[:,id][ids_j]

pecmy.br(np.ones(pecmy.x_len), b_init, m, wv_m_i, id)
pecmy.br(np.ones(pecmy.x_len), b_init, m_prime, wv_m_prime_i, id)

# A: no, substantial spillovers, China liberalizes toward everybody in response to US coercion










### NO CHANGES ###

tau_hat_base = np.ones((N, N))
D_hat_base = np.ones(N)
m = np.diag(M)

dict_base = dict()
dict_base["tau_hat"] = "tau_hat_base"


### FREE TRADE ###

tau_hat_ft = 1 / tau
ecmy = economy.economy(data, params)
ecmy.geq_solve(tau_hat_ft, np.ones(ecmy.N))

### DOUBLE BARRIERS ###

tau_hat_db = tau * 2
for i in range(len(Y)):
    tau_hat_db[i, i] = 1

imp.reload(economy)
ecmy = economy.economy(data, params)

# ecmy.tau
# tau_hat_pos = np.ones_like(ecmy.tau)
# tau_hat_pos[ecmy.tau < 1] = 1 / ecmy.tau[ecmy.tau < 1]
# tau_hat_pos

ecmy.purgeD()
test = ecmy.geq_solve(tau_hat_ft, D_hat_base)
testBase = ecmy.geq_solve(tau_hat_base, D_hat_base)
testBase_x = ecmy.unwrap_ge_dict(testBase)
# ecmy.U_hat(test)
test_x = ecmy.unwrap_ge_dict(test)

b = np.repeat(1, len(nu))
theta_dict = dict()
theta_dict["b"] = b
theta_dict["alpha"] = np.array([alpha_0, alpha_1])
theta_dict["gamma"] = gamma
theta_dict["c_hat"] = .2



imp.reload(economy)
imp.reload(policies)
# pecmy = policies.policies(data, params, b)
pecmy = policies.policies(data, params, b, rcv_path="results/rcv.csv")

m = np.diag(pecmy.M)
alpha = np.array([alpha_0, alpha_1])
c_hat = .1
# m = np.zeros_like(pecmy.ecmy.tau)
# for i in range(pecmy.N):
#     m[i, ] = pecmy.M[i] / pecmy.N
# U.S. threatens China
# m[5,0] = pecmy.M[5] / 2
# m[5,5] = pecmy.M[5] / 2
b = np.repeat(.5, pecmy.N)
theta_dict["b"] = b
theta_dict["alpha"] = alpha
theta_dict["c_hat"] = c_hat

lsolve_test = pecmy.Lsolve(tau_hat_base, m, theta_dict, 0)

wv = pecmy.war_vals(m, theta_dict)
i = 0
ni = np.delete(np.arange(pecmy.N), i)
wi = wv[:,0][ni]
br_test = pecmy.br(np.ones_like(test_x), b, m, wi, i)

test_b_i = pecmy.est_b_i(0, m, alpha, c_hat, scipy=True)
# shrinking the bounds seems to cause more instability...
# why is it searching such crazy numbers in the first place?
# probably similar to the reason we had to reduce step size in solver


# pecmy.Lsolve(tau_hat_base, m, theta_dict, 0)
# this is just very unstable...which is probably the underlying problem in the estimation.









np.array([1, 2, 3])[-1]


test = pecmy.est_xy_tau(m, alpha, c_hat)

# trying with general geq constraints and log objective
    # maybe go back to eq constraints after this?
    # attempting...


pecmy.ecmy.tau




tau_hat = np.ones_like(pecmy.ecmy.tau)
ge_dict2 = pecmy.ecmy.geq_solve(tau_hat, np.ones(pecmy.N))
pecmy.ecmy.unwrap_ge_dict(ge_dict2)


wv = pecmy.war_vals(m, theta_dict)

j2 = np.delete(np.arange(pecmy.N), 2)
wv2 = wv[:,0][j2]
tLsolve = pecmy.Lsolve(tau_hat_base, m, theta_dict, 5, ft=True)
tLsolve







ecmy.geq_solve(pecmy.tau_hat_ft_i(tau_hat_base, 5), np.ones(pecmy.N))
tLsolve
pecmy.br(np.ones(pecmy.x_len), b, m, wv2, 2)




# x = tLsolve[pecmy.N:]
#
# ge_x = []
# ge_x.extend(np.ones(pecmy.N**2+pecmy.N))
# ge_x.extend(x)
# ecmy.rewrap_ge_dict(np.array(ge_x))


tLsolve[pecmy.N:pecmy.N**2]


len(tLsolve) - pecmy.N - pecmy.lambda_i_len
pecmy.x_len - pecmy.N ** 2


# for i in range(pecmy.N):
#     ji = np.delete(np.arange(pecmy.N), i)
#     wvi = wv[:,i][ji]
#
#     tLsolve = pecmy.Lsolve(tau_hat_base, m, theta_dict, i)
#
#
# # solving for 2's best response requires recursion, what does this mean for estimation?
#     # starting at free trade works
#     # similarly with 5
#
# pecmy.br(np.ones(pecmy.x_len), b, m, wvi, i)


ecmy.rewrap_ge_dict(pecmy.rewrap_xy_tau(test['x'])["ge_x"])

# D_hats look weird too, is this a problem in Lsolve?
# pecmy.ecmy.D
# none of this matters because deficts are purged in pecmy.


# code 3: more than 3*n iterations in LSQ subproblem seems to occur randomly

# attempting with sqrt output on loss function, still gives code 3 but parameter estimates look better along the way...
    # attempted with general ge constraints, attempting with id-specific constraints
    # this seems to stick at corners

# attempting to log G_hat output...what is wrong with this?
    # it's bad for mil constraints but we can code these separately
    # works fine for Lsolve
# could also try duplicating general and id-specific ge constraints
    # this is running now
    # shows flashes of brilliance, and flashes of hiding in corner
    # fails with code 3 again
# trying with just id-specific constraints...
    # around minute 15 bs get stuck at 2 and only multipliers are moving around, objective gets stuck at same value
    # gets through 28 jacobian evaluations before giving code 3 again
# diagonal is good..
    # code 3 seems totally random, try grabbing trial output and retrying...
    # seeing some of the same behavior here that we sometimes see in Lsolve...pushing solutions toward corner...this is probably because solver is getting into convex regions of objective



np.any(np.isnan([np.nan, np.nan]))


pecmy.x_len + pecmy.lambda_i_len * pecmy.N + pecmy.N  # input
pecmy.x_len - pecmy.N**2 - self.N  # ge_constraints


78 * pecmy.N




wv = pecmy.war_vals(m, theta_dict)

j0 = np.delete(np.arange(pecmy.N), 0)
wv0 = wv[:,0][j0]

pecmy.br(np.ones(pecmy.x_len), b, m, wv0, 0)


x = []
x.extend([1, 2, 3], [4, 5, 6])




pecmy.x_len
pecmy.lambda_i_len










pecmy.lambda_i_len
pecmy.x_len

pecmy.Lsolve(tau_hat_base, m, theta_dict, 0)


pecmy.x_len
pecmy.lambda_i_len * pecmy.N + pecmy.N + pecmy.x_len



out = pecmy.rewrap_xy_tau(test['x'])
len(pecmy.rewrap_lambda(out["lambda_x"])[0])
pecmy.lambda_i_len
pecmy.lambda_i_len_td

for i in range(pecmy.N):
    pecmy.Lsolve(tau_hat_base, m, theta_dict, i)

# adding bounds on b screws things up for some reason







theta_dict = dict()


test = pecmy.est_b(m, alpha, c_hat)
# TODO: seems to want to shrink bs toward zero for some reason
    # probably has something to do with getting revenue out of Lagrangian...this is why minimizing FOC not ideal



start_time = time.time()
pecmy.br(np.ones(pecmy.x_len), b, m, wv0, 0)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test = pecmy.Lsolve(tau_hat_base, m, theta_dict, 0)
print("--- %s seconds ---" % (time.time() - start_time))
test_li = test[pecmy.x_len:]
pecmy.rewrap_lambda_i(test_li)
# TODO: these aren't right (for complete m case), chi multipliers should be positive
    # doesn't converge fully and doesn't satisfy mil constraints for U.S. and China
    # solves ok with just U.S. threat, and returns multiplier



pecmy.Lzeros(np.ones(pecmy.x_len), np.zeros(pecmy.lambda_i_len), theta_dict["b"], tau_hat_base, wv0, 0)









pecmy.Lsolve(tau_hat_base, m, 0)


test_lbda = pecmy.L_solve_lbda(np.ones(pecmy.x_len), b, np.ones_like(pecmy.ecmy.tau), wv0, 0)
pecmy.rewrap_lambda_i(test_lbda['x'])

test_b = pecmy.est_b(m, alpha, c_hat)





4 % 3
start_time = time.time()
test_br_m1 = pecmy.br(testBase_x, m, 0, mpec=True)
print("--- %s seconds ---" % (time.time() - start_time))





test = pecmy.estimate()


testA = np.copy(test)
testx = np.copy(test['x'])
test_dict = pecmy.rewrap_y(testx)
test_dict["theta_m"]

# 07/23/2019
    # originally feeding wrong war values to Lagrangian, checking everything again w/ equality constraints
        # compare war vals to rcvs and make sure everything looks right
        # still fails with equality constraints, trying with lower bounds on multipliers but epsilon ball around 0 in dGm_ji
        # still returns nonsense after 230 minutes

war_diffs = np.array([-1, 1, 0])
np.where(war_diffs < 0, 0, war_diffs)


pecmy.rewrap_y(test['x'])









np.dot(np.array([1,2,3]), np.array([1,2,3]))


test1 = pecmy.rewrap_y(test['x'])
pecmy.rewrap_m(test1['m'])









# CLAIMS 'INEQUALITY CONSTRAINTS INCOMPATIBLE' after 8923 function evaluations, ~40 minutes of running
# only 14 jacobian evaluations
# attempting with equality constraints
    # here we get 'Singular matrix C in LSQ subproblem' in first Jacobian evaluation, refers to constraint qualification problem, study up on this
    # Might be due to starting at zero multipliers, or because derivative wrt own policy is linear combination of other derivatives when gamma = 1
    # removing derivative wrt own allocation doesn't seem to help though
    # problem seems to be that we weren't selecting self.M[j], rather selecting whole vector
    # fixing this lets everything run, but gamma shrinks to zero and everything goes to shit
    # got objective value down to 56 w/o gamma but hit same error at minute 111, not clear why here, estimates were pretty stable generally. SSE seemed to be monotonically decreasing in c_hat for most of estimation routine.
    # try starting at zero lambda vector
    # problem is that equality constraint jacobian inversion fails for weird trial paramters, try with upper and lower constraints?

lambda_x_init = np.zeros(pecmy.lambda_i_len * pecmy.N)
lambda_dict = pecmy.rewrap_lambda(lambda_x_init)
pecmy.rewrap_lambda_i(lambda_dict[1])["chi_i"][0] = 1

m = np.diag(M)
rhoM = pecmy.rhoM(theta_dict)
m[0, 1] = 2
m[1, 1] = 1
m_x = pecmy.unwrap_m(m)

pecmy.dGdm_ji(m_x, 0, 1, theta_dict, rhoM, lambda_dict)
pecmy.dGdm_ii(m_x, 1, theta_dict, rhoM, lambda_dict)

for i in range(5):
    print(i)

rhoM

pecmy.rewrap_m(m_x)
testG = ag.grad(pecmy.chi)
testG(m_x, 0, 1, theta_dict, rhoM)



pecmy.rcv[0]
pecmy.war_vals(np.diag(pecmy.M), theta_dict)
# for i in range(pecmy.N):
m[5, 0] = m[5,5] * .5
m[5, 5] = m[5,5] * .5
pecmy.war_vals(m, theta_dict)


ids = np.arange(0, pecmy.N)
test = np.delete(ids, 3)
id = np.where(test == i)[0]

test[id]

test = pecmy.br_war_ji(np.ones(pecmy.ecmy.ge_x_len), b, 0, 5, full_opt=True)
test_dict = pecmy.ecmy.rewrap_ge_dict(test)
test_dict




np.arange(0, 1.1, .1)



lambda_x_init = np.zeros(pecmy.lambda_i_len * pecmy.N)
m_x_init = np.diag(M).flatten()
x_init = np.append(np.append(testBase_x, lambda_x_init), m_x_init)


test = pecmy.br_m(x_init, 0)




ge_ft_x = pecmy.ecmy.unwrap_ge_dict(pecmy.ecmy.geq_solve(tau_hat_ft, D_hat_base))

pecmy.G_hat(testBase_x)[1]
pecmy.G_hat(pecmy.br_war_ji(testBase_x, 1, 0))[1]
pecmy.G_hat(ge_ft_x)[1]
pecmy.G_hat(pecmy.br_war_ji(ge_ft_x, 1, 0))[1]









m = np.diag(M)
m_x = pecmy.unwrap_m(m)
m = pecmy.rewrap_m(m_x)


# lambda_x = np.zeros(pecmy.N*pecmy.lambda_i_len)
# lambda_dict = pecmy.rewrap_lambda(lambda_x)
# pecmy.unwrap_lambda(lambda_dict) == lambda_x


np.append(m[0, ], m[1, ])
m = np.diag(M)
# for i in range(pecmy.N):
m[5, 0] = m[5,5] * .5
m[5, 5] = m[5,5] * .5

m.flatten()

test0 = pecmy.Lsolve(tau_hat_base, m, 0)  # works with mil constraints for zero when we don't square the output and use diagonal military matrix

test_br_m1 = pecmy.br(testBase_x, m, 0, mpec=True)













m = np.diag(M)
m[5, 0] = m[5,5] * .5
m[5, 5] = m[5,5] * .5
test_br_m2 = pecmy.br(testBase_x, m, 0, mpec=True)
np.exp(pecmy.G_hat(test_br_m2, ids=np.array([5])))



# TODO: check that this still works with unlogged objective
# recursive formulation finds zeros for all economies

# lambda_base = np.zeros(pecmy.lambda_i_len)
baseline_x = pecmy.br(testBase_x, m, 1)
baseline_dict = pecmy.ecmy.rewrap_ge_dict(baseline_x)
# np.exp(pecmy.G_hat(baseline_x, np.array([1])))

# TODO: try starting at free trade vec for id if we don't find solution
# ecmy.geq_solve(tau_hat_ft_i(1, pecmy.ecmy.tau), np.ones(pecmy.N))
# lm works for id 0 and logged objective with fct=.1, starting at no change, also works for unlogged objective. Also works starting from free trade, also works with squared output
# also works for 1 when we start at free trade and use unlogged objective, also works with squared output
# 2 works if I square the output and start from free trade
    # same with 3, 4
# 5 works with squared output starting from base


test2 = pecmy.Lsolve(tau_hat_ft_i(5, pecmy.ecmy.tau), 5)


test_dict = pecmy.ecmy.rewrap_ge_dict(test2['x'][0:pecmy.ecmy.ge_x_len])

test_dict = pecmy.ecmy.rewrap_ge_dict(test['x'][0:pecmy.ecmy.ge_x_len])
np.exp(pecmy.G_hat(test[0][0:pecmy.ecmy.ge_x_len], np.array([1])))
pecmy.ecmy.tau * test_dict["tau_hat"]



x = []
x.extend(testBase_x)  # TODO this doesn't work to flatten, same in Lzeros
x.extend(lambda_base)

pecmy.Lzeros(np.array(x), tau_hat_base, 0)


testGrad = ag.grad(pecmy.Lagrange)
testGrad(testBase_x, tau_hat_base, lambda_base, 0)





start_time = time.time()
time.time() - start_time

testGrad = ag.grad(pecmy.G_hat)
testGrad(testBase_x, np.array([0]))


start_time = time.time()
test_br_m1 = pecmy.br(testBase_x, m, 0, mpec=True)
print("--- %s seconds ---" % (time.time() - start_time))
# pecmy.G_hat(test_br_m1, np.array([0]))

# start_time = time.time()
# test_ne = pecmy.nash_eq(m)
# pecmy.ecmy.rewrap_ge_dict(test_ne)
# print("--- %s seconds ---" % (time.time() - start_time))

# test to see if we can get constraints to bind (U.S. spends half of effort coercing China)
m = np.diag(M)
m[5, 0] = m[5,5] * .5
m[5, 5] = m[5,5] * .5

start_time = time.time()
test_br_m2 = pecmy.br(testBase_x, m, 0, mpec=True)
print("--- %s seconds ---" % (time.time() - start_time))
pecmy.G_hat(test_br_m2, np.array([0]))

start_time = time.time()
test_ne2 = pecmy.nash_eq(m)
pecmy.ecmy.rewrap_ge_dict(test_ne)
print("--- %s seconds ---" % (time.time() - start_time))

# np.inf
# pecmy.rho()
# pecmy.W
# 1 / (1 + np.exp(1))
# pecmy.G_hat(test_x, np.repeat(.5, len(Y)))

ccodes
ecmy.rewrap_ge_dict(pecmy.br_war_ij(testBase_x, 5, 1))



start_time = time.time()
test_ne = pecmy.nash_eq()
pecmy.ecmy.rewrap_ge_dict(test_ne)
print("--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
warij = pecmy.br_war_ij(testBase_x, 0, 1, mpec=True)
print("--- %s seconds ---" % (time.time() - start_time))

pecmy.ecmy.rewrap_ge_dict(warij)["tau_hat"] * pecmy.ecmy.tau
pecmy.ecmy.rewrap_ge_dict(warij)["X_hat"] * pecmy.ecmy.Xcif


# 38 seconds on laptop
# pecmy.ecmy.tau

start_time = time.time()
test_br = pecmy.br(testBase_x, 0, mpec=False)
print("--- %s seconds ---" % (time.time() - start_time))
# test_brc = pecmy.br_cor(testBase_x)
# much longer, and requires some recursion to get out of ge solution traps

start_time = time.time()
test_ne = pecmy.nash_eq()
pecmy.ecmy.rewrap_ge_dict(test_ne)
print("--- %s seconds ---" % (time.time() - start_time))
# 480 with del2 method on laptop (with some printing)
# 524 seconds for iterationstart_time = time.time()


# trade values in the thousands, tens of thousands for unfavored countries






### DEBUG ZONE ###

imp.reload(economy)
ecmy = economy.economy(data, params)
ecmy.purgeD()


tau_hat = np.ones_like(tau)
for k in range(ecmy.N):
    if k != 1:
        tau_hat[1, k] = 1 / ecmy.tau[1, k]
D_hat_base = np.ones(N)
ge_dict = dict()
ge_dict["tau_hat"] = tau_hat
ge_dict["D_hat"] = D_hat_base

tau_hat * ecmy.tau

test = ecmy.geq_solve(tau_hat, D_hat_base, fct=1)


# ecmy.update_ge_dict(test['x'], ge_dict)

ecmy.update_ge_dict(test[1][0], ge_dict)

# ecmy = economy.economy(data, params)
# ecmy.purgeD()

tau_hat_test = np.array([[1.        , 1.20908617, 1.22400761, 2.13669735, 0.29,
        1.11933451],
       [1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        ],
       [1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        ],
       [1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        ],
       [1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        ],
       [1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        ]])

imp.reload(economy)
imp.reload(policies)
pecmy = policies.policies(data, params, b)
pecmy.ecmy.geq_solve(tau_hat_test, np.ones(pecmy.N))

ecmy.geq_solve(tau_hat_test, np.ones(ecmy.N))
