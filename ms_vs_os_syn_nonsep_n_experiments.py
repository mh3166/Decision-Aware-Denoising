import numpy as np
import pandas as pd
import sys

def generate_data_fixed(n, mu_val, nu_val, B0):
    B = mu_val.shape[0]
    mod_arr_trunc = np.arange(n + 2*B0)%(B)
    mu = np.zeros(n + 2*B0)
    nu = np.zeros(n + 2*B0)
    for i in range(B):
        ind_arr = 1*(mod_arr_trunc == i)
        mu += ind_arr*mu_val[i]
        nu += ind_arr*nu_val[i]
    return mu, nu
    
def generate_data(mu_val, nu_val, p_arr, n):
    indices = np.random.choice(range(len(mu_val)),n, p=p_arr)
    mu = mu_val[indices]
    nu = nu_val[indices]
    # mu_hat = np.random.normal(mu, 1/nu**0.5)
    return mu, nu

def generate_mu_hat(mu, nu):
    mu_hat = np.random.normal(mu, 1/nu**0.5)
    return mu_hat

def generate_mu_hat_bin(mu, nu, S_0):
    mu_0 = np.ones(mu.shape)
    p = 1/(S_0/(nu * 1) + 1)
    C = mu_0/p
    mu_hat = C*np.random.binomial(S_0, p)/S_0 - 1 + mu
    return mu_hat

def solve_problem(mu_plug_in):
    sol_ind = np.argmax(mu_plug_in, axis = 1)
    sol = np.zeros(mu_plug_in.shape)
    sol[np.arange(mu_plug_in.shape[0]), sol_ind] = 1
    V = np.sum(sol * mu_plug_in)
    return sol, V

def delta_cov(L, cov_vec, h):
    L1 = np.matmul(L, np.diag(cov_vec))
    return h*(L1 + L1.T) + np.diag(cov_vec) * h**2

# One-Shot VGC
def OneShot_VGC(T, L, cov, h, n_b, B, B0, K, S, obj):
    delta_h_cov = delta_cov(L, cov, h)
    delta_h_cov_1 = delta_h_cov[B0:n_b-B0,B0:n_b-B0].copy()
    total_obj_h = 0
    for i in range(K):
        nth_cov_h = delta_h_cov_1[B*i:B*(i+1), B*i:B*(i+1)]
        for s in range(S):
            delta_h = np.random.multivariate_normal(np.zeros(B), nth_cov_h)
            sol_h, obj_h = solve_problem(T[B*i:B*(i+1)] + delta_h)
            total_obj_h += obj_h
    obj_h = total_obj_h/S
    os_vgc = (obj_h - obj)/h

    return os_vgc

def OneShot_VGC_big(T, L, cov, h, n_b, B, B0, K, S, obj):
    delta_h_cov = delta_cov(L, cov, h)
    delta_h_cov_1 = delta_h_cov[B0:n_b-B0,B0:n_b-B0].copy()
    total_obj_h = 0
    for s in range(S):
        delta_h = np.random.multivariate_normal(np.zeros(n), delta_h_cov_1)
        delta_h = delta_h.reshape(-1, B)
        sol_h, obj_h = solve_problem(T + delta_h)
        total_obj_h += obj_h
    obj_h = total_obj_h/S
    os_vgc = (obj_h - obj)/h

    return os_vgc

def MultiShot_VGC(T, L, nu, h, n_b, B, B0, S, sol):
    std_h = (h ** 2 + 2 * h / nu ** 0.5) ** 0.5
    total_obj_h_1 = 0
    total_obj_h_2 = 0
    L_diag = np.diag(L)[B0 : n_b - B0]
    for s in range(S):
        delta_h = np.random.normal(0, std_h)
        mod_arr_trunc = np.arange(n)%(B+2)
        for i in range(B+2):
            ind_arr = np.zeros(n_b)
            ind_arr_trunc = (mod_arr_trunc == i)*1
            ind_arr[B0 : n_b - B0] = ind_arr_trunc
            T_h = np.matmul(L,ind_arr*delta_h)[B0 : n_b - B0].reshape(-1,B) + T
            sol_h, obj_h = solve_problem(T_h)
            V_h = np.sum((T_h * sol_h).reshape(-1, B), axis = 1)
            V = np.sum((T * sol).reshape(-1, B), axis = 1)
            nu_denom_arr = np.max(np.matmul((L > 0)*1, ind_arr * nu)[B0 : n_b - B0].reshape(-1,B), axis = 1)
            nu_denom_arr = nu_denom_arr + (nu_denom_arr == 0)*1
            ms_vgc_i = np.sum((V_h - V)/(0.5 * h * nu_denom_arr ** 0.5))
            total_obj_h_1 += ms_vgc_i

        # delta_h = np.random.normal(0, std_h)
        for i in range(n_b - 2*B0):
            ind_arr = np.zeros(n_b)
            ind_arr[i + B0] = 1
            T_h = np.matmul(L,ind_arr*delta_h)[B0 : n_b - B0].reshape(-1,B) + T
            sol_h, obj_h = solve_problem(T_h)
            ms_vgc_i = (obj_h - obj) / (0.5 * h * nu[B0 + i] ** 0.5)
            total_obj_h_2 += ms_vgc_i
    ms_vgc = total_obj_h_1/S
    ms_vgc_s = total_obj_h_2/S
    
    return ms_vgc, ms_vgc_s

def MultiShot_VGC_fast(T, L, nu, h, n_b, B, B0, S, sol):
    std_h = (h ** 2 + 2 * h / nu ** 0.5) ** 0.5
    total_obj_h_1 = 0
    L_diag = np.diag(L)[B0 : n_b - B0]
    for s in range(S):
        delta_h = np.random.normal(0, std_h)
        mod_arr_trunc = np.arange(n)%(B+2)
        for i in range(B+2):
            ind_arr = np.zeros(n_b)
            ind_arr_trunc = (mod_arr_trunc == i)*1
            ind_arr[B0 : n_b - B0] = ind_arr_trunc
            T_h = np.matmul(L,ind_arr*delta_h)[B0 : n_b - B0].reshape(-1,B) + T
            sol_h, obj_h = solve_problem(T_h)
            V_h = np.sum((T_h * sol_h).reshape(-1, B), axis = 1)
            V = np.sum((T * sol).reshape(-1, B), axis = 1)
            nu_denom_arr = np.max(np.matmul((L > 0)*1, ind_arr * nu)[B0 : n_b - B0].reshape(-1,B), axis = 1)
            nu_denom_arr = nu_denom_arr + (nu_denom_arr == 0)*1
            ms_vgc_i = np.sum((V_h - V)/(0.5 * h * nu_denom_arr ** 0.5))
            total_obj_h_1 += ms_vgc_i
    ms_vgc = total_obj_h_1/S
    
    return ms_vgc

def MultiShot_VGC_fast_2(T, L, nu, h, n_b, B, B0, S, sol):
    std_h = ((h ** 2 + 2 * h) / nu) ** 0.5
    total_obj_h_1 = 0
    L_diag = np.diag(L)[B0 : n_b - B0]
    for s in range(S):
        delta_h = np.random.normal(0, std_h)
        mod_arr_trunc = np.arange(n)%(B+2)
        for i in range(B+2):
            ind_arr = np.zeros(n_b)
            ind_arr_trunc = (mod_arr_trunc == i)*1
            ind_arr[B0 : n_b - B0] = ind_arr_trunc
            T_h = np.matmul(L,ind_arr*delta_h)[B0 : n_b - B0].reshape(-1,B) + T
            sol_h, obj_h = solve_problem(T_h)
            V_h = np.sum((T_h * sol_h).reshape(-1, B), axis = 1)
            V = np.sum((T * sol).reshape(-1, B), axis = 1)
            nu_denom_arr = np.max(np.matmul((L > 0)*1, ind_arr * nu)[B0 : n_b - B0].reshape(-1,B), axis = 1)
            nu_denom_arr = nu_denom_arr + (nu_denom_arr == 0)*1
            ms_vgc_i = np.sum((V_h - V)/(0.5 * h))
            total_obj_h_1 += ms_vgc_i
    ms_vgc = total_obj_h_1/S
    
    return ms_vgc

def Stein_Correction(T, L, nu, h, n_b, B, B0, S, sol):
    step_h = h / nu ** 0.5
    stein_bias_est = 0
    L_diag = np.diag(L)[B0 : n_b - B0]
    mod_arr_trunc = np.arange(n)%(B+2)
    for i in range(B+2):
        ind_arr = np.zeros(n_b)
        ind_arr_trunc = (mod_arr_trunc == i)*1
        ind_arr[B0 : n_b - B0] = ind_arr_trunc
        T_h = np.matmul(L,ind_arr*step_h)[B0 : n_b - B0].reshape(-1,B)
        T_h_pos = T + T_h
        T_h_neg = T - T_h
        sol_h_pos, obj_h_pos = solve_problem(T_h_pos)
        sol_h_neg, obj_h_neg = solve_problem(T_h_neg)
        nu_denom_arr = (1/(2 * h * nu ** 0.5))[B0 : n_b - B0].reshape(-1,B)
        # print((sol_h_pos - sol_h_neg),ind_arr_trunc.reshape(-1,B))
        stein_bias_arr = (sol_h_pos - sol_h_neg) * nu_denom_arr * ind_arr_trunc.reshape(-1,B)
        stein_bias = np.sum(stein_bias_arr)
        stein_bias_est += stein_bias
    return stein_bias_est

def MultiShot_VGC_fast_1(T, L, nu, h, n_b, B, B0, S, sol):
    std_h = (h ** 2 + 2 * h / nu ** 0.5) ** 0.5
    std_2h = (4 * h ** 2 + 4 * h / nu ** 0.5) ** 0.5
    total_obj_h_1 = 0
    L_diag = np.diag(L)[B0 : n_b - B0]
    for s in range(S):
        delta_h = np.random.normal(0, std_h)
        delta_2h = np.random.normal(0, std_2h)
        mod_arr_trunc = np.arange(n)%(B+2)
        for i in range(B+2):
            ind_arr = np.zeros(n_b)
            ind_arr_trunc = (mod_arr_trunc == i)*1
            ind_arr[B0 : n_b - B0] = ind_arr_trunc
            T_h = np.matmul(L,ind_arr*delta_h)[B0 : n_b - B0].reshape(-1,B) + T
            T_2h = np.matmul(L,ind_arr*delta_2h)[B0 : n_b - B0].reshape(-1,B) + T
            sol_h, obj_h = solve_problem(T_h)
            sol_2h, obj_2h = solve_problem(T_2h)
            V_h = np.sum((T_h * sol_h).reshape(-1, B), axis = 1)
            V_2h = np.sum((T_2h * sol_2h).reshape(-1, B), axis = 1)
            V = np.sum((T * sol).reshape(-1, B), axis = 1)
            nu_denom_arr = np.max(np.matmul((L > 0)*1, ind_arr * nu)[B0 : n_b - B0].reshape(-1,B), axis = 1)
            nu_denom_arr = nu_denom_arr + (nu_denom_arr == 0)*1
            ms_vgc_i = np.sum((4*V_h - V_2h - 3*V)/(2 * h * nu_denom_arr ** 0.5))
            total_obj_h_1 += ms_vgc_i
    ms_vgc = total_obj_h_1/S
    
    return ms_vgc

def generate_data_fixed(n, mu_val, nu_val, B0):
    B = mu_val.shape[0]
    mod_arr_trunc = np.arange(n + 2*B0)%(B)
    mu = np.zeros(n + 2*B0)
    nu = np.zeros(n + 2*B0)
    for i in range(B):
        ind_arr = 1*(mod_arr_trunc == i)
        mu += ind_arr*mu_val[i]
        nu += ind_arr*nu_val[i]
    return mu, nu

trial = int(sys.argv[1])
total_trials = 200
n_arr = [20, 40, 80, 160, 320, 640, 1280]
h_arr = [2**(-5)]
B_arr = [20]

np.random.seed(999)
seed_arr = np.random.choice(range(10000), total_trials, replace=False)
print(seed_arr)

exp_array = []
for n in n_arr:
    for h in h_arr:
        for B in B_arr:
            exp_array.append([n, h, B])

summary_table = [['n', 'h', 'B', 't', 'in_samp_perf', 'oos_perf', 'est', 'method']]

for exp in exp_array:
    n = exp[0]
    h = exp[1]
    B = exp[2]
    t = trial
    
    # Fixed/Calculated Parameters
    S = 50
    B0 = 4
    K = int(n / B)
    # h = n ** (-1/4)
    # h = 2**(-5)
    
    n_b = n + 2 * B0
    mu_val = np.array([0, 0.5, 1, 0])
    nu_val = np.array([2, 10, 6, 2])
    p_arr = [0.33,0.33,0.34]
    
    # L Matrix
    L = np.zeros((n_b,n_b))
    for i in range(L.shape[0]):
        L[i,i] = 0.5
    for i in range(L.shape[0]-1):
        L[i,i+1] = 0.25
        L[i+1,i] = 0.25
    
    # Problem Parameters
    np.random.seed(101)
    # mu, nu = generate_data(mu_val, nu_val, p_arr, n_b)
    mu, nu = generate_data_fixed(n, mu_val, nu_val, B0)
    cov = 1 / nu
    
    # Data
    np.random.seed(seed_arr[t])
    mu_hat = generate_mu_hat(mu, nu)
    
    # Remove parameters not in problem
    mu_trunc = mu[B0:B0 + n].reshape(-1,B)
    nu_trunc = nu[B0:B0 + n].reshape(-1,B)
    mu_hat_trunc = mu_hat[B0:B0 + n].reshape(-1,B)
    cov_trunc = 1/nu_trunc
    
    # Solve optimization problem with fixed Plug-in
    plug_in = np.matmul(L,mu_hat)[B0:B0 + n].reshape(-1,B)
    sol, obj = solve_problem(plug_in)
    oos_perf = np.sum(sol * mu_trunc)
    insamp_perf = np.sum(sol * mu_hat_trunc)
    
    #One-Shot VGC
    os_vgc = OneShot_VGC_big(plug_in, L, cov, h, n_b, B, B0, K, S, obj)
    os_vgc_est = insamp_perf - os_vgc
    
    # Multi-Shot VGC
    ms_vgc = MultiShot_VGC_fast_2(plug_in, L, nu, h, n_b, B, B0, S, sol)
    ms_vgc_est = insamp_perf - ms_vgc

    # Stein Correction
    stein_corr = Stein_Correction(plug_in, L, nu, h, n_b, B, B0, S, sol)
    stein_corr_est = insamp_perf - stein_corr

    row_ms = [n, h, B, t, insamp_perf, oos_perf, ms_vgc_est, 'MS']
    row_os = [n, h, B, t, insamp_perf, oos_perf, os_vgc_est, 'OS']
    row_sc = [n, h, B, t, insamp_perf, oos_perf, stein_corr_est, 'Stein']
    summary_table.append(row_ms)
    summary_table.append(row_os)
    summary_table.append(row_sc)
summary_df_2 = pd.DataFrame(data = summary_table[1:], columns = summary_table[0])
summary_df_2.to_csv('data_syn/os_vs_ms_nonsep_n_' + str(trial) + '.csv', index = False)