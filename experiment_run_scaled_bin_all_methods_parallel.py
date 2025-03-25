import time
import sys
import torch
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.binomial import Binomial
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
device_cpu = torch.device("cpu")
device_cuda = torch.device("cuda")


# Methods for creating key matrices
def construct_L0_matrix(dist_matrix, theta):
    k_matrix = 1*(dist_matrix <= theta)
    total_units = torch.sum(k_matrix, axis = 1)
    return k_matrix/total_units[:,None]


def construct_PSD_delta_cov(L, cov, h, device):
    d_cov = delta_cov(L, cov, h).to(device)
    if torch.linalg.cholesky_ex(d_cov).info > 0:
        d_cov = nearPSD_1(d_cov, device, e = 0.00001).to(device_cpu)
    return d_cov

def method_experiment(T, L, m_d, h, solve_knapsack, VGC, p_hat, p, var, C_R, B, a, rho, n, K, device):
    h_ms = n ** (-1/6)

    obj, sol, dual_var, gap = solve_knapsack(-T * C_R, B, a, rho, n, device_cpu)
    true_obj = torch.dot(sol, -p * C_R).item()
    true_mse = torch.sum(((T - p) * C_R)**2).item()
    hat_mse = torch.sum(((T - p_hat) * C_R)**2).item() + 2*torch.sum(C_R**2 * torch.diag(L)*var)
    in_sample_obj = torch.dot(sol, -p_hat * C_R).item()
    stein_bias = Stein_h(T, L, C_R, dual_var, n, h_ms, 1/var, device)
    stein_slow_bias = Stein_Slow_h(T, L, C_R, B, a, rho, n, h_ms, 1/var, device)
    sol_decoup = 1*(-T * C_R + dual_var <= 0)
    obj_decoup = (sol_decoup * (-T * C_R + dual_var))
    vgc_bias, vgc_bias_decoup = VGC(m_d, T, C_R, dual_var, B, a, rho, n, obj, obj_decoup, h, K, device)
    ms_vgc_bias, ms_vgc_bias_sec = MS_VGC_decoup_h(T, L, C_R, B, a, rho, n, obj_decoup, dual_var, h_ms, 1/var, K, device)
    msos_vgc_bias, msos_vgc_bias_sec = VGC_MSOS_h(T, L, C_R, B, a, rho, n, obj, dual_var, h_ms, 1/var, n, device)
    # Put output in a dictionary
    out_dict = {}
    out_dict['true_obj'] = true_obj
    out_dict['in_sample_obj'] = in_sample_obj
    out_dict['vgc_est'] = in_sample_obj - vgc_bias
    out_dict['vgc_est_decoup'] = in_sample_obj - vgc_bias_decoup
    out_dict['mse_hat'] = hat_mse.item()
    out_dict['mse'] = true_mse
    out_dict['stein_est'] = in_sample_obj + stein_bias
    out_dict['stein_slow_est'] = in_sample_obj + stein_slow_bias
    out_dict['ms_vgc_est'] = in_sample_obj - ms_vgc_bias
    out_dict['ms_vgc_sec_est'] = in_sample_obj - ms_vgc_bias_sec
    out_dict['ms_vgc_os_est'] = in_sample_obj - msos_vgc_bias
    out_dict['ms_vgc_sec_os_est'] = in_sample_obj - msos_vgc_bias_sec
    out_dict['gap'] = gap
    return out_dict

# def nearPSD_0(A, device, e = 1e-8):
#     vec = torch.diagonal(A).to(device) - (torch.sum(A, axis = 1).to(device) - torch.diagonal(A).to(device))
#     n = vec.shape[0]
#     return A - torch.diag(vec * (vec < 0)) + torch.diag(torch.ones(n) * e).to(device)

def nearPSD_0(A, device, alpha = 1):
    vec = 2*torch.diagonal(A).to(device) - torch.sum(A, axis = 1).to(device)
    return A - alpha*torch.diag(vec * (vec < 0))

def nearPSD_1(A, device, e = 1e-8):
    L, Q = torch.linalg.eigh(A)
    L_plus = L*(L > 0)
    L_minus = L*(L < 0)
    p = torch.sum(L)/torch.sum(L_plus)
    C = torch.linalg.multi_dot([Q,torch.diag(L_plus + e),Q.transpose(0,1)])
    return C*(p**.5)


def delta_cov(L, cov, h):
    L1 = torch.matmul(L, cov)
    return h*(L1 + L1.T) + cov * h**2


# Solve non-regularized knapsack
def solve_knapsack(Z, b, a, rho, n, device):
    x = torch.zeros(n).to(device)
    topkind = torch.topk(Z, b, largest = False).indices
    x[topkind] = 1
    lbda = -torch.kthvalue(Z, b).values
    obj = torch.dot(Z,x)
    return obj, x, lbda, 0

def update_x_l2(T, lbd, rho, a):
    val = -(T + lbd*a)/(2 * rho)
    x = 1*(val > 1) + (val <= 1)*(val >= 0)*val
    return x


def solve_reg_knapsack_l2(Z, b, a, rho, n, device, precision=1e-8):
    # Initialize Parameters
    min_Z = -max(Z).item()
    max_Z = -(min(Z)).item() + 2*rho
    lbd = (min_Z + max_Z) * 0.5 
    x = update_x_l2(Z, lbd, rho, a)
    diff = (torch.dot(a,x) - b).item()
    diff2 = abs(max_Z - min_Z)
    # Bisection Loop
    while(abs(diff2) > precision):
        if diff > 0:
            min_Z = lbd
            lbd = (max_Z + lbd) * 0.5
        else:
            max_Z = lbd
            lbd = (min_Z + lbd) * 0.5
        x = update_x_l2(Z, lbd, rho, a)
        diff = (torch.dot(a,x) - b).item()
        diff2 = abs(max_Z - min_Z)
    # Compute obj val and check dual gap
    obj = torch.dot(x,Z) + rho*torch.dot(x,x)
    grad = Z + lbd * a + 2 * rho * x
    mu1 = grad*(x == 0)
    mu2 = -grad*(x == 1)
    L = torch.dot(x,Z) + rho*torch.dot(x,x) + lbd*(torch.dot(x,a) - b) + torch.sum(mu1*(- x)) + torch.sum(mu2*(1 - x))
    dual_gap = abs(L - obj)
    
    grad_full = torch.sum(grad - mu1 + mu2)
    feas = (torch.dot(x,a) - b)
    primal_feas = (feas > 0) * feas + torch.sum((x < 0) * -x) + torch.sum((x >= 1) * (x - 1))
    dual_feas = torch.sum((-mu1 > 0) * (-mu1)) + torch.sum((-mu2 > 0) * (-mu2)) + -lbd * (lbd < 0)
    
    return obj, x, dual_gap.item()

# VGC methods
def VGC_h(m_d, T, C_R, lbda, B, a, rho, n, obj, obj_decoup, h, K, device):
    total = 0
    total_decoup = 0
    for i in range(K):
        delta_h = m_d.sample().to(torch.device("cpu"))
        # print(delta_h)
        cost_vec = (-T* C_R - delta_h * C_R).to(torch.device("cpu"))
        obj_d, sol_d, lbda_d, gap_d = solve_knapsack(cost_vec, B, a, rho, n, torch.device("cpu"))
        sol_h_decoup = 1*(cost_vec + lbda <= 0)
        obj_h_decoup = sol_h_decoup * (cost_vec + lbda)
        total += (obj_d - obj)/h
        total_decoup += torch.sum(obj_h_decoup - obj_decoup)/h
    vgc_bias = total/K
    vgc_bias_decoup = total_decoup/K
    return vgc_bias.item(), vgc_bias_decoup.item()

# VGC methods
def VGC_MSOS_h(T, L, C_R, B, a, rho, n, obj, lbda, h, nu, K, device):
    total = 0
    total_second = 0
    # std_h = ((h ** 2 + 2 * h) / nu) ** 0.5
    # std_2h = ((4 * h ** 2 + 4 * h) / nu) ** 0.5
    std_h = ((h ** 2 + 2 * h) / nu) ** 0.5
    std_2h = ((4 * h ** 2 + 4 * h) / nu) ** 0.5
    L_diag = torch.diag(L)
    for i in range(K):
        rand_ind = torch.randint(0,n,(1,))[0]
        rand_vec = torch.zeros(n)
        L_col = L[:,rand_ind]
        L_vec = torch.zeros(n)
        L_denom = L[rand_ind,rand_ind]
        L_vec[rand_ind] = L_denom
        delta_h = torch.normal(0, std_h[rand_ind]) 
        delta_2h = torch.normal(0, std_2h[rand_ind]) 

        cost_vec_h = (-T* C_R - L_vec * delta_h * C_R).to(torch.device("cpu")) 
        cost_vec_2h = (-T* C_R - L_vec * delta_2h * C_R).to(torch.device("cpu"))

        obj_h, sol_h, lbda_h, gap_h = solve_knapsack(cost_vec_h, B, a, rho, n, torch.device("cpu"))
        obj_2h, sol_2h, lbda_2h, gap_2h = solve_knapsack(cost_vec_2h, B, a, rho, n, torch.device("cpu"))
        
        total_second += (4 * obj_h - obj_2h - 3 * obj)/(2 * h / n * L_denom)
        total += (obj_h - obj)/(h / n * L_denom)
    vgc_bias = total/K
    vgc_bias_sec = total_second/K
    return vgc_bias.item(), vgc_bias_sec.item()


def VGC_h_l2(m_d, T, C_R, B, a, rho, n, obj, h, K, device):
    total = 0
    for i in range(K):
        delta_h = m_d.sample().to(torch.device("cpu"))
        cost_vec = (-T* C_R - delta_h * C_R).to(torch.device("cpu"))
        obj_d, sol_d, gap_d = solve_reg_knapsack_l2(-T* C_R - delta_h * C_R, B, a, rho, n, torch.device("cpu"))
        total += (obj_d - obj)/h
    vgc_bias = total/K
    return vgc_bias.item()

def Stein_h(T, L, C_R, lbda, n, h, nu, device):
    total = 0
    ones_vec = torch.ones(n)
    L_diag = torch.diag(L)
    cost_vec_pos = (-T* C_R - L_diag * ones_vec * h * C_R / (nu ** 0.5)).to(torch.device("cpu"))
    cost_vec_neg = (-T* C_R + L_diag * ones_vec * h * C_R / (nu ** 0.5)).to(torch.device("cpu"))
    sc_terms = (((cost_vec_pos + lbda) <= 0)*1 - ((cost_vec_neg + lbda) <= 0)*1) / (2*h*nu**0.5)
    return torch.sum(sc_terms).item()

def Stein_Slow_h(T, L, C_R, B, a, rho, n, h, nu, device):
    total = 0
    ones_vec = torch.ones(n)
    pert_vec = ones_vec * h / (nu ** 0.5)
    for j in range(n):
        unit_vec = torch.zeros(n)
        unit_vec[j] = pert_vec[j]
        pert = torch.matmul(L, unit_vec)
        cost_vec_pos = (C_R * (-T - pert)).to(torch.device("cpu"))
        cost_vec_neg = (C_R * (-T + pert)).to(torch.device("cpu"))
        obj_pos, sol_pos, lbda_pos, gap_pos = solve_knapsack(cost_vec_pos, B, a, rho, n, torch.device("cpu"))
        obj_neg, sol_neg, lbda_neg, gap_neg = solve_knapsack(cost_vec_neg, B, a, rho, n, torch.device("cpu"))
        sc_term = (sol_pos[j] - sol_neg[j]) / (2*h*nu[j]**0.5)
        total += sc_term
    return total.item()

def MS_VGC_h(T, L, C_R, B, a, rho, n, obj, h, nu, K, device):
    total = 0
    for j in range(n):
        std_h = (h ** 2 + 2 * h / nu[j] ** 0.5) ** 0.5
        std_2h = (4 * h ** 2 + 4 * h / nu[j] ** 0.5) ** 0.5
        unit_vec = torch.zeros(n)
        unit_vec[j] = 1
        for i in range(K):
            delta_h = torch.normal(0, std_h)
            delta_2h = torch.normal(0, std_2h)
            # print(delta_h)
            # cost_vec_h = (-T* C_R - L[j,j] * delta_h * unit_vec * C_R).to(torch.device("cpu"))
            # cost_vec_2h = (-T* C_R - L[j,j] * delta_2h * unit_vec * C_R).to(torch.device("cpu"))
            cost_vec_h = (-T* C_R - torch.matmul(L, delta_h * unit_vec * C_R)).to(torch.device("cpu")) 
            cost_vec_2h = (-T* C_R - torch.matmul(L, delta_2h * unit_vec * C_R)).to(torch.device("cpu"))
            
            obj_h, sol_h, lbda_h, gap_h = solve_knapsack(cost_vec_h, B, a, rho, n, torch.device("cpu"))
            obj_2h, sol_2h, lbda_2h, gap_2h = solve_knapsack(cost_vec_2h, B, a, rho, n, torch.device("cpu"))
            total += (4 * obj_h - obj_2h - 3 * obj)/(2 * h * L[j,j])
    vgc_bias = total/K
    return vgc_bias.item()

def MS_VGC_decoup_h(T, L, C_R, B, a, rho, n, obj, lbda, h, nu, K, device):
    total = 0
    total_second = 0
    # std_h = ((h ** 2 + 2 * h) / nu) ** 0.5
    # std_2h = ((4 * h ** 2 + 4 * h) / nu) ** 0.5
    std_h = ((h ** 2 + 2 * h) / nu) ** 0.5
    std_2h = ((4 * h ** 2 + 4 * h) / nu) ** 0.5
    L_diag = torch.diag(L)
    for i in range(K):
        delta_h = torch.normal(0, std_h)
        delta_2h = torch.normal(0, std_2h)
        # cost_vec_h = (-T* C_R - L[j,j] * delta_h * unit_vec * C_R).to(torch.device("cpu"))
        # cost_vec_2h = (-T* C_R - L[j,j] * delta_2h * unit_vec * C_R).to(torch.device("cpu"))
        cost_vec_h = (-T* C_R - L_diag * delta_h * C_R).to(torch.device("cpu")) 
        cost_vec_2h = (-T* C_R - L_diag * delta_2h * C_R).to(torch.device("cpu"))
        sol_h = 1*(cost_vec_h + lbda <= 0)
        sol_2h = 1*(cost_vec_2h + lbda <= 0)
        obj_h = (sol_h * (cost_vec_h + lbda))
        obj_2h = (sol_2h * (cost_vec_2h + lbda))
        # obj_h, sol_h, lbda_h, gap_h = solve_knapsack(cost_vec_h, B, a, rho, n, torch.device("cpu"))
        # obj_2h, sol_2h, lbda_2h, gap_2h = solve_knapsack(cost_vec_2h, B, a, rho, n, torch.device("cpu"))
        total_second += torch.sum((4 * obj_h - obj_2h - 3 * obj)/(2 * h * L_diag))
        total += torch.sum((obj_h - obj)/(h * L_diag))
    vgc_bias = total/K
    vgc_bias_sec = total_second/K
    return vgc_bias.item(), vgc_bias_sec.item()



probs_year = pd.read_csv("probabilities_census_tracts_year.csv")
probs_total = pd.read_csv("probabilities_census_tracts_total.csv")
centroid_df = pd.read_csv("centroid_table.csv")

y = 2018
probs_selected = probs_year[(probs_year['YEAR'] == y)].copy()
probs_combined = probs_total.merge(probs_selected, how = 'left')
probs_combined = probs_combined.fillna(0)
probs_combined['crashes'] += 1*(probs_combined['crashes'] == 0)
probs_combined = probs_combined[probs_combined['prob_ped_injured_total'] != 0]
probs_combined = probs_combined[probs_combined['crashes_total'] <= 2000]
p = probs_combined['prob_ped_injured_total'].to_numpy()
S = probs_combined['crashes'].to_numpy()
# C_R = torch.FloatTensor(np.log(probs_combined['crashes_total'].to_numpy())).to(device_cpu)
C_R = torch.FloatTensor(probs_combined['crashes_total'].to_numpy()/1000).to(device_cpu)

n = p.shape[0]
# C_R = torch.ones(n).to(device)
h = n ** (-1/6)
var_0 = p*(1-p)*S/(S)**2
var_1 = np.ones(n) * np.mean(p*(1-p)*S/(S)**2)
q = 1
# Convert to Torch
S = torch.FloatTensor(S).to(device_cpu)
p = torch.FloatTensor(p)
p_cuda = torch.FloatTensor(p).to(device)

# Compute distance matrix
rcdf = probs_combined.merge(centroid_df)
reduced_rcdf = rcdf[['boro_code', 'ct2010', 'c_lat', 'c_lon']].drop_duplicates()
reduced_rcdf["id"] =  reduced_rcdf["boro_code"].astype(str) + reduced_rcdf["ct2010"].astype(str)
reduced_rcdf = reduced_rcdf[['id', 'c_lat', 'c_lon']].copy()
pairwise = pd.DataFrame(
    squareform(pdist(reduced_rcdf.iloc[:, 1:])),
    columns = reduced_rcdf.id,
    index = reduced_rcdf.id
)
dist_matrix = pairwise.values*1e6
dist_matrix_torch = torch.FloatTensor(dist_matrix).to(device)

unique_dist_matrix = sorted(np.unique(np.ceil(dist_matrix)))
sparse_unique_dist_matrix = [0]
for i in unique_dist_matrix[1:]:
    if i - sparse_unique_dist_matrix[-1] > 1:
        sparse_unique_dist_matrix.append(i)
        
theta_ind_arr = [x for x in range(100)] + [x * 50 for x in range(2,20)] + [x * 500 for x in range(2,20)] + [x * 10000 for x in range(1,5)] + [len(sparse_unique_dist_matrix)-1]


quantiles = [x*0.00002 for x in range(101)] + [x*0.002 for x in range(1,10)] + [x*0.02 for x in range(1,10)] + [x*0.2 for x in range(1,6)]
theta_arr = np.quantile(np.unique(dist_matrix), quantiles)
tau_arr = [np.round(x*0.00025,5) for x in range(101)]
tau_arr = [100000] + tau_arr
B_arr = [80]
q_arr = [1, 2, 3, 4]

exp_library = []
torch.manual_seed(105)
indices_arr = torch.randperm(100000)
print(indices_arr)

sim = int(sys.argv[1])
exp_num= int(sys.argv[2])

for ind in range(sim, sim + 1):
    for q in q_arr:
        for theta_ind in theta_ind_arr:
            for tau in tau_arr[0:1]:
                exp_library.append([theta_ind, tau, ind, q])
            
B = 107
rho = n ** (-1/3)
h = n ** (-1/4)
# h = 1

summary_results_vgc = [['tau', 'theta', 'theta_ind', 'trial', 'q', 'B', 'alpha', 'policy',
                        'true_obj','in_sample_obj', 'vgc_est', 'vgc_est_decoup', 'mse', 'mse_hat', 
                        'stein_est', 'stein_slow_est', 'ms_vgc_est','ms_vgc_sec_est','ms_vgc_os_est','ms_vgc_sec_os_est',
                        'gap', 'full_info_perf']]
summary_results = [['tau','theta','theta_ind','trial','q','B','alpha','gap','policy','true_perf','full_info_perf','est','method']]

fil_name = "Data_Out/results_all_methods_" + str(exp_num) + "_2018_"
if os.path.isfile(fil_name + str(sim) + ".csv"):
    existing_df = pd.read_csv(fil_name + str(sim) + ".csv")
    if not existing_df.empty:
        last_entry = existing_df.tail(1)[["theta_ind", "tau", "trial", "q"]].to_numpy().tolist()[0]
        print(last_entry)
        if last_entry in exp_library:
            starting_index = exp_library.index(last_entry) + 1
        else:
            starting_index = 0
    else:
        starting_index = 0
else:
    existing_df = pd.DataFrame(data = summary_results_vgc[1:], columns = summary_results_vgc[0])
    starting_index = 0 

fil_name_melt = "Data_Out/results_all_methods_" + str(exp_num) + "_melt_2018_"
if os.path.isfile(fil_name_melt + str(sim) + ".csv"):
    existing_df_melt = pd.read_csv(fil_name_melt + str(sim) + ".csv")
else:
    existing_df_melt = pd.DataFrame(data = summary_results[1:], columns = summary_results[0])
    
for ind_exp in range(starting_index, len(exp_library)):
    exp_parameters = exp_library[ind_exp]
    theta_ind = exp_parameters[0]
    theta = sparse_unique_dist_matrix[theta_ind]
    tau = exp_parameters[1]
    trial = exp_parameters[2]
    q = exp_parameters[3]
    torch.manual_seed(indices_arr[trial])
    S_0 = torch.ceil(S * q)
    m1 = Binomial(S_0, p)
    p_hat = m1.sample().to(device_cpu) / S_0
    p_hat_cuda = p_hat.to(device)
    a = torch.ones(n).to(device_cpu)
    full_out = solve_knapsack(-C_R*p, B, a, 0, n, torch.device("cpu"))
    full_info_obj = full_out[0].item()
    
    tries = 0
    while True:
        try:
            # GENERATE L MATRIX
            st = time.perf_counter()
            var_hat = p_hat * (1-p_hat) / S_0 
            if tries > 5:
                var_hat = var_hat + (var_hat == 0) * (1+tries)*1e-6 * torch.rand(var_hat.shape[0])
            else:
                var_hat = var_hat + (var_hat == 0) * 1e-6 * torch.rand(var_hat.shape[0])
            cov_hat = torch.FloatTensor(torch.diag(var_hat))
            cov_hat_cuda = cov_hat.to(device)
            L_0_cuda = construct_L0_matrix(dist_matrix_torch, theta)
            L_0 = L_0_cuda.to(device_cpu)
        
            # GENERATE DELTA COV MATRIX AND SAMPLER
            d_cov = construct_PSD_delta_cov(L_0_cuda, cov_hat_cuda, h, device).to(device_cpu)
            m_d = MultivariateNormal(torch.FloatTensor(np.zeros(n)).to(device), d_cov.to(device))
            break
        except:
            print("Error, Trying Again")
            tries += 1

    # COMPUTE PLUGIN
    pert_vec = torch.normal(0, torch.ones(n)*0.000001)
    T = torch.matmul(L_0, p_hat).to(device_cpu) + pert_vec
    
    # SOLVE KNAPSACK
    K = 25
    results = method_experiment(T, L_0, m_d, h, solve_knapsack, VGC_h, p_hat, p, var_hat, C_R, B, a, rho, n, K, device)    
    
    # ['true_obj','in_sample_obj', 'vgc_est', 'mse', 'mse_hat', 'stein_est', 'ms_vgc_est','gap']
    row = [tau, theta, theta_ind, trial, q, B, 0, 'unreg',
           results['true_obj'], results['in_sample_obj'], 
           results['vgc_est'], results['vgc_est_decoup'],
           results['mse'], results['mse_hat'], 
           results['stein_est'], 
           results['stein_slow_est'], 
           results['ms_vgc_est'], 
           results['ms_vgc_sec_est'],
           results['ms_vgc_os_est'], 
           results['ms_vgc_sec_os_est'],
           results['gap'], 
           full_info_obj]
    summary_results_vgc.append(row)
    
    #['tau','theta','theta_ind','trial','q','B','alpha','gap','policy','est','method']
    row_pre = [tau, theta, theta_ind, trial, q, B, 0, results['gap'], 'unreg', results['true_obj'], full_info_obj]
    row_oracle = row_pre + [results['true_obj'], 'Oracle']
    row_osvgc = row_pre + [results['vgc_est'], 'One-Shot VGC']
    row_osvgc_decoup = row_pre + [results['vgc_est_decoup'], 'One-Shot VGC Decoup']
    row_msvgc = row_pre + [results['ms_vgc_est'], 'Multi-Shot VGC']
    row_msvgc_sec = row_pre + [results['ms_vgc_sec_est'], 'Multi-Shot VGC Second']
    row_osmsvgc = row_pre + [results['ms_vgc_os_est'], 'One-Shot Multi-Shot VGC']
    row_osmsvgc_sec = row_pre + [results['ms_vgc_sec_os_est'], 'One-Shot Multi-Shot VGC Second']
    row_msehat = row_pre + [results['mse_hat'], 'MSE']
    row_mse = row_pre + [results['mse'], 'True MSE']
    row_stein = row_pre + [results['stein_est'], 'Stein']
    row_stein_slow = row_pre + [results['stein_slow_est'], 'Stein Slow']
    summary_results.append(row_oracle)
    summary_results.append(row_osvgc)
    summary_results.append(row_osvgc_decoup)
    summary_results.append(row_msvgc)
    summary_results.append(row_msvgc_sec)
    summary_results.append(row_osmsvgc)
    summary_results.append(row_osmsvgc_sec)
    summary_results.append(row_msehat)
    summary_results.append(row_mse)
    summary_results.append(row_stein)
    summary_results.append(row_stein_slow)
    

    
    print("time: ", time.perf_counter() - st)
    print(row)
    print(exp_parameters)
    if ind_exp % 100 == 0:
        summary_results_vgc_df = pd.DataFrame(data = summary_results_vgc[1:], columns = summary_results_vgc[0])
        new_df = pd.concat([existing_df,summary_results_vgc_df])                   
        new_df.to_csv(fil_name + str(sim) + ".csv", index = False)

        summary_results_df = pd.DataFrame(data = summary_results[1:], columns = summary_results[0])
        new_df_melt = pd.concat([existing_df_melt,summary_results_df])                   
        new_df_melt.to_csv(fil_name_melt + str(sim) + ".csv", index = False)
