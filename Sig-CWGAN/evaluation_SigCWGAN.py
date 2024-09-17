# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:40:29 2024

@author: hugom
"""

# Requires libraries
import os
import torch
import test_metrics_spyder as tm
# from lib.utils import load_pickle, to_numpy
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from geomloss import SamplesLoss
from scipy.stats import wasserstein_distance

import powerlaw


### Loading of fake time series (SigCWGAN)
path_fake01_ts = r"x_fake_M.pt"
device = "cuda"
# x_fake_M = load_pickle(os.path.join(os.path.dirname(experiment_dir), 'x_fake_M.torch')).to(device)
x_fake01_M = torch.load(path_fake01_ts).to(device)

# Setting appropiate dimensions of dataset to apply functions in a vectorized way
x_fake01_Mm = torch.unsqueeze(x_fake01_M,1)


### Loading of real time series
# Input correct path to retrieve stock prices
data_path = r".\data500SP_v1.csv"
data1 = pd.read_csv(data_path, index_col=0, header=[0,1])
data1.index = pd.to_datetime(data1.index)

# Stocks to be excluded from analysis
incomplete_stocks = ["FOXA","FOX","DOW","UBER","CTVA","OTIS","CARR","ABNB","CEG","GEHC","KVUE","VLTO"]
# incomplete_stocks = ["FOXA","FOX","DOW","UBER","CTVA","OTIS","CARR","ABNB","CEG","GEHC","KVUE","VLTO","APA","TRGP","OXY","PCG"]
# Computation of log of returns
log_returns = data1['Adj Close'].div(data1['Adj Close'].shift(periods=1)).apply(np.log)[1:]
log_returns = log_returns[["GOOG"]]
# log_returns.drop(columns=incomplete_stocks, inplace=True)
# plt.plot(log_returns)
stock_price = data1['Adj Close'][["GOOG"]]
stock_price.to_csv("GOOG.csv")
# A brief summary of log return of prices
summary_logreturns = log_returns.describe().T

# Definition of powerlaw function to compute exponent of powerlaw distribution
def power_law(y):
    y_pos = y[y>=0]
    y_neg = np.abs(y[y<0])
    power_law_pos = powerlaw.Fit(y_pos, parameter_range={"alpha": [2,6]}).power_law.alpha
    power_law_neg = powerlaw.Fit(y_neg, parameter_range={"alpha": [2,6]}).power_law.alpha
    # return np.average(np.array([power_law_pos, power_law_neg]))
    return np.array([power_law_pos, power_law_neg])

# Real data as tensor data type
x_real_N = log_returns.to_numpy().T
x_real_N = torch.from_numpy(x_real_N).float().unsqueeze(2)
# Setting appropiate dimensions of dataset to apply functions in a vectorized way
x_real_Nn = torch.unsqueeze(x_real_N,1)


# Real data as numpy array data type
x_real_np = x_real_N.detach().clone()
x_real_np = x_real_np.cpu().numpy()[:, :, 0]
# x_real_df = pd.DataFrame(x_real_np)

# Fake data as numpy array data type
x_fake01_np = x_fake01_M.detach().clone()
x_fake01_np = x_fake01_np.cpu().numpy()[:, :, 0]


# Number of lags
max_lag = 100

# Function to compute the stylized features on sampled paths (using Pytorch implementation)
def compute_stylfeat(x_M, max_lag=max_lag):
    # Time dependence
    acf_vmap = torch.vmap(tm.acf_torch)
    acf_M = acf_vmap(x_M, max_lag=max_lag, dim=(0, 1))
    # print(acf_M.shape, acf_M)
    
    cfv_vmap = torch.vmap(tm.coarfine_vol_torch)
    cfv_M = cfv_vmap(x_M, max_lag=max_lag, dim=(0, 1))
    # print(cfv_M.shape, cfv_M)
    
    vc_vmap = torch.vmap(tm.vol_clust_torch)
    vc_M = vc_vmap(x_M, max_lag=max_lag, dim=(0, 1))
    # print(vc_M.shape, vc_M)
    
    le_vmap = torch.vmap(tm.lev_eff_torch)
    le_M = le_vmap(x_M, max_lag=max_lag, dim=(0, 1))
    # print(le_M.shape, le_M)
    
    # Single value
    mean_vmap = torch.vmap(torch.mean)
    mean_M = mean_vmap(x_M, dim=(0, 1))
    # print(mean_M.shape, mean_M)
    
    std_vmap = torch.vmap(torch.std)
    std_M = std_vmap(x_M, dim=(0, 1))
    # print(std_M.shape, std_M)
    
    skw_vmap = torch.vmap(tm.skew_torch)
    skw_M = skw_vmap(x_M, dim=(0, 1))
    # print(skw_M.shape, skw_M)
    
    kurt_vmap = torch.vmap(tm.kurtosis_torch)
    kurt_M = kurt_vmap(x_M, dim=(0, 1))
    # print(kurt_M.shape, kurt_M)
    
    
    # acf_Mmean = acf_M.mean((1))
    # cfv_Mmean = cfv_M.mean((1))
    # vc_Mmean = vc_M.mean((1))
    # le_Mmean = le_M.mean((1))
    
    print(mean_M.shape, std_M.shape, skw_M.shape, kurt_M.shape, acf_M.shape, vc_M.shape, le_M.shape)
    stylfeat_dict = {"Mean" : mean_M,
                     "Standard deviation" : std_M,
                     "Skewness" : skw_M,
                     "Kurtosis" : kurt_M,
                     "ACF" : acf_M[:,:,0],
                     "Volatility clustering" : vc_M[:,:,0],
                     "Leverage effect" : le_M[:,:,0],
                     "Coarse-fine volatility" : cfv_M[:,:,0]
                     }
    
    stylfeat_dict = {stylfeat : stylfeat_dict[stylfeat].cpu().numpy() for stylfeat in stylfeat_dict}
    
    stylfeat_dfs_list = []
    for stylfeat in stylfeat_dict:
        stylfeat_np = stylfeat_dict[stylfeat]
        # print(stylfeat, stylfeat_np.shape)
        if stylfeat_dict[stylfeat].shape[1] == 1:
            print("Single value")
            # print(stylfeat_np)
            stylfeat_df = pd.DataFrame(stylfeat_np, columns=[stylfeat], dtype=float)
        elif (stylfeat_dict[stylfeat].shape[1]) != 1 and stylfeat == "Coarse-fine volatility":
            print("Multiple value")
            # print(stylfeat_np)
            stylfeat_df = pd.DataFrame(stylfeat_np, columns=[stylfeat + f"_k={k}" for k in range(- max_lag, max_lag + 1)], dtype=float)
        else:
            print("Multiple value")
            # print(stylfeat_np)
            stylfeat_df = pd.DataFrame(stylfeat_np, columns=[stylfeat + f"_k={k}" for k in range(1, max_lag + 1)], dtype=float)
        stylfeat_dfs_list.append(stylfeat_df)
        # print(stylfeat_df)
    
    stylfeat_df_all = pd.concat(stylfeat_dfs_list, axis=1)
    
    return stylfeat_df_all


# List of stylized features
# stylfeat_names_list = ["Mean", "Standard deviation", "Skewness", "Kurtosis", "Mean-ACF", "Mean-volatility clustering", "Mean-leverage effect", "Mean-coarse-fine volatility", "Powerlaw 0<x", "Powerlaw 0>x"]

# Computation of stylized features
stylfeat_real_pre = compute_stylfeat(x_real_Nn[:])
stylfeat_fake01_pre = compute_stylfeat(x_fake01_Mm[:])

# Computation of additional stylized feature (power law)
powerlaw_real_df = pd.DataFrame(np.apply_along_axis(power_law, 1, x_real_np[:]), columns=["Powerlaw 0<x", "Powerlaw 0>x"])
powerlaw_fake01_df = pd.DataFrame(np.apply_along_axis(power_law, 1, x_fake01_np[:]), columns=["Powerlaw 0<x", "Powerlaw 0>x"])

# powerlaw_real_df.to_csv(".\computations\powerlaw_real_df.csv")
# powerlaw_fake01_df.to_csv(".\computations\powerlaw_fake01_df.csv")

# Concatenation of additional stylized features
stylfeat_real_df = pd.concat([stylfeat_real_pre, powerlaw_real_df], axis=1)
# stylfeat_real_df.to_csv(".\computations\stylfeat_real_df.csv")
# stylfeat_real_df = pd.read_csv(".\computations\stylfeat_real_df.csv", index_col=0)

stylfeat_fake01_df = pd.concat([stylfeat_fake01_pre, powerlaw_fake01_df], axis=1)
# stylfeat_fake01_df.to_csv(".\computations\stylfeat_fake01_df.csv")
# stylfeat_fake01_df = pd.read_csv(".\computations\stylfeat_fake01_df.csv", index_col=0)
stylfeat_fake01_df = stylfeat_fake01_df.loc[0:0,:]
x_fake01_np = x_fake01_np[0:1][:]


### Adding some labels to thstylized features (according to the kind of output value)
stylfeat_names_df = pd.Series(stylfeat_real_df.columns.values).str.split("_k=", expand=True)
stylfeat_names_df.columns = ["Name", "Lag"]
dupl_idx = stylfeat_names_df[stylfeat_names_df.duplicated(subset=["Name"], keep=False)].index
stylfeat_names_df.loc[dupl_idx, "Type"] = "Multiple"
stylfeat_names_df = stylfeat_names_df.replace([None, "nan"], [np.nan, np.nan])
stylfeat_names_df["Type"] = stylfeat_names_df["Type"].fillna(value="Single")

stylfeat_names_df = stylfeat_names_df[stylfeat_names_df["Type"]=="Multiple"]

stylfeat_names_df = stylfeat_names_df[~stylfeat_names_df.duplicated(subset="Name")].reset_index(drop=True)






### Algorithm to compute predictive score
from sklearn.linear_model import LinearRegression
import pickle

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()

def compute_predictive_score(x_past, x_future, x_fake):
    size = x_fake.shape[0]
    X = to_numpy(x_past.reshape(size, -1))
    Y = to_numpy(x_fake.reshape(size, -1))
    size = x_past.shape[0]
    X_test = X.copy()
    Y_test = to_numpy(x_future[:, :1].reshape(size, -1))
    model = LinearRegression()
    model.fit(X, Y)  # TSTR
    r2_tstr = model.score(X_test, Y_test)
    model = LinearRegression()
    model.fit(X_test, Y_test)  # TRTR
    r2_trtr = model.score(X_test, Y_test)
    return dict(r2_tstr=r2_tstr, r2_trtr=r2_trtr, predictive_score=np.abs(r2_trtr - r2_tstr))

# Input correct path to retrieve real path
experiment_dir = r".\x_real.torch"
p, q = 3, 3
x_real0 = load_pickle(os.path.join(os.path.dirname(experiment_dir), 'x_real_test.torch')).to("cuda")
# x_real0 = x_real_test
x_past0, x_future0 = x_real0[:, :p], x_real0[:, p:p + q]
x_future0 = x_real0[:, p:p + q]
dim0 = x_real0.shape[-1]

x_fake01 = torch.t(x_fake01_M[0:1, :251, 0])
predict_score_dict = compute_predictive_score(x_past0, x_future0, x_fake01)
predictive_score = predict_score_dict["predictive_score"]



### Algorithm to plot computed first four moments
mean_real = stylfeat_real_df["Mean"].values[0]
std_real = stylfeat_real_df["Standard deviation"].values[0]
skew_real = stylfeat_real_df["Skewness"].values[0]
kurt_real = stylfeat_real_df["Kurtosis"].values[0]


mean_fake01 = stylfeat_fake01_df["Mean"].values[0]
std_fake01 = stylfeat_fake01_df["Standard deviation"].values[0]
skew_fake01 = stylfeat_fake01_df["Skewness"].values[0]
kurt_fake01 = stylfeat_fake01_df["Kurtosis"].values[0]

pos_powerlaw_real = stylfeat_real_df["Powerlaw 0<x"].values[0]
pos_powerlaw_fake01 = stylfeat_fake01_df["Powerlaw 0<x"].values[0]
dif_pos_powerlaw = pos_powerlaw_real - pos_powerlaw_fake01

print(f"Pos powerlaw real: {'{:,.3e}'.format(pos_powerlaw_real)}", f"Pos powerlaw fake: {'{:,.3e}'.format(pos_powerlaw_fake01)}")
print(f"Difference Pos powerlaw: {'{:,.3e}'.format(dif_pos_powerlaw)}")

neg_powerlaw_real = stylfeat_real_df["Powerlaw 0>x"].values[0]
neg_powerlaw_fake01 = stylfeat_fake01_df["Powerlaw 0>x"].values[0]
dif_neg_powerlaw = neg_powerlaw_real - neg_powerlaw_fake01

print(f"Neg powerlaw real: {'{:,.3e}'.format(neg_powerlaw_real)}", f"Neg powerlaw fake: {'{:,.3e}'.format(neg_powerlaw_fake01)}")
print(f"Difference Neg powerlaw: {'{:,.3e}'.format(dif_neg_powerlaw)}")


wass_dist_log_returns = wasserstein_distance(np.array(log_returns).T[0], x_fake01_np[:][0])

plt.hist(log_returns, 50, density=True, histtype='bar', color=["grey"]*log_returns.shape[1], alpha=0.6, label="real")
plt.hist(x_fake01_np[:].T, 50, density=True, histtype='bar', color=["red"]*x_fake01_np[:].T.shape[1], alpha=0.4, label="fake")
y_min, y_max = plt.ylim()[0], plt.ylim()[1]
x_min, x_max = plt.xlim()[0], plt.xlim()[1]
plt.text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"W-distance: {'{:,.3e}'.format(wass_dist_log_returns)}", fontsize=10)

plt.text(0.05*(x_max-x_min)+x_min, 0.8*(y_max-y_min)+y_min, f"Mean real: {'{:,.3e}'.format(mean_real)}", fontsize=10)
plt.text(0.2*(x_max-x_min)+x_min, 0.8*(y_max-y_min)+y_min, f"Mean fake: {'{:,.3e}'.format(mean_fake01)}", fontsize=10)
plt.text(0.05*(x_max-x_min)+x_min, 0.7*(y_max-y_min)+y_min, f"Std real: {'{:,.3e}'.format(std_real)}", fontsize=10)
plt.text(0.2*(x_max-x_min)+x_min, 0.7*(y_max-y_min)+y_min, f"Std fake: {'{:,.3e}'.format(std_fake01)}", fontsize=10)
plt.text(0.05*(x_max-x_min)+x_min, 0.6*(y_max-y_min)+y_min, f"Skew real: {'{:,.3e}'.format(skew_real)}", fontsize=10)
plt.text(0.2*(x_max-x_min)+x_min, 0.6*(y_max-y_min)+y_min, f"Skew fake: {'{:,.3e}'.format(skew_fake01)}", fontsize=10)
plt.text(0.05*(x_max-x_min)+x_min, 0.5*(y_max-y_min)+y_min, f"Kurt real: {'{:,.3e}'.format(kurt_real)}", fontsize=10)
plt.text(0.2*(x_max-x_min)+x_min, 0.5*(y_max-y_min)+y_min, f"Kurt fake: {'{:,.3e}'.format(kurt_fake01)}", fontsize=10)


plt.xlabel("log returns")
plt.ylabel("Probability density")
plt.yscale('log')
plt.legend()
plt.show()


### Plot of sampled path along with the computed predictive score. Note that this computation is only over one single path
plt.plot(np.array(log_returns), color="dodgerblue", label="real")
plt.plot(x_fake01_np[:].T, color="orange", label="fake")
y_min, y_max = plt.ylim()[0], plt.ylim()[1]
x_min, x_max = plt.xlim()[0], plt.xlim()[1]
plt.text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"Predictive score: {'{:,.3e}'.format(predictive_score)}", fontsize=10)
plt.text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"Predictive score: {'{:,.3e}'.format(predictive_score)}", fontsize=10)
plt.text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"Predictive score: {'{:,.3e}'.format(predictive_score)}", fontsize=10)
plt.xlabel("days")
plt.ylabel("log returns")
# plt.yscale('log')
plt.legend()
plt.show()



### Algorithm to plot computed stylized features
n_rows = 2
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
fig.tight_layout(pad=2.0)

# Loop over plots of stylized features
for n, test_metric in enumerate(stylfeat_names_df.index[:],-1):
    feature_name = stylfeat_names_df.loc[n+1, "Name"]
    feature_type = stylfeat_names_df.loc[n+1, "Type"]
    print(feature_name, ":", feature_type)
    
    i = (n+1) % n_rows
    if (n+1) <= (n_rows-1): j=0
    else: j=1
    # j = (n+1) % 2
    print(i,j)
    
    if feature_type == "Single":
        feature_real_data = stylfeat_real_df[feature_name]
        feature_fake_data = stylfeat_fake01_df[feature_name]
        
        feature_real_data_torch = torch.unsqueeze(torch.Tensor(stylfeat_real_df[feature_name]), 1)
        feature_fake_data_torch = torch.unsqueeze(torch.Tensor(stylfeat_fake01_df[feature_name]), 1)
        
        wass_dist = wasserstein_distance(feature_real_data.values, feature_fake_data.values)

        # print("Wasserrstein distance", wass_dist)
    
        axes[i,j].hist(feature_real_data, bins=50, density=True, histtype='bar', alpha=0.5, color="dodgerblue", label="real")
        axes[i,j].hist(feature_fake_data, bins=50, density=True, histtype='bar', alpha=0.5, color="orange", label="fake")
        
        y_min, y_max = axes[i,j].get_ylim()[0], axes[i,j].get_ylim()[1]
        x_min, x_max = axes[i,j].get_xlim()[0], axes[i,j].get_xlim()[1]
        # axes[i,j].text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"W-distance GL: {'{:,.6f}'.format(L)}", fontsize=8)
        axes[i,j].text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"W-distance: {'{:,.3e}'.format(wass_dist)}", fontsize=10)
    
    elif feature_type == "Multiple" and feature_name == "Volatility clustering":
        feature_real_data = stylfeat_real_df.filter(regex=f"^{feature_name}_").mean()
        feature_fake_data = stylfeat_fake01_df.filter(regex=f"^{feature_name}_").mean()
        
        mse = (np.square(feature_real_data.values - feature_fake_data.values)).mean()
        print("MSE", mse)
        
        axes[i,j].set_xscale('log')
        axes[i,j].set_yscale('log')
        
        axes[i,j].plot(range(1, max_lag + 1), feature_real_data.values, color="dodgerblue", label="real")
        axes[i,j].plot(range(1, max_lag + 1), feature_fake_data.values, color="orange", label="fake")
        # axes[i,j].set_ylim([1.0e-5, 0.01])
        
        y_min, y_max = axes[i,j].get_ylim()[0], axes[i,j].get_ylim()[1]
        x_min, x_max = axes[i,j].get_xlim()[0], axes[i,j].get_xlim()[1]
        axes[i,j].text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"MSE: {'{:,.3e}'.format(mse)}", fontsize=10)
        
    elif feature_type == "Multiple" and feature_name != "Coarse-fine volatility":
        feature_real_data = stylfeat_real_df.filter(regex=f"^{feature_name}_").mean()
        feature_fake_data = stylfeat_fake01_df.filter(regex=f"^{feature_name}_").mean()
        
        mse = (np.square(feature_real_data.values - feature_fake_data.values)).mean()
        print("MSE", mse)
        
        axes[i,j].plot(range(1, max_lag + 1), feature_real_data.values, color="dodgerblue", label="real")
        axes[i,j].plot(range(1, max_lag + 1), feature_fake_data.values, color="orange", label="fake")
        if feature_name in ["ACF"]: axes[i,j].set_ylim([-1.0, 1.0])
        
        y_min, y_max = axes[i,j].get_ylim()[0], axes[i,j].get_ylim()[1]
        x_min, x_max = axes[i,j].get_xlim()[0], axes[i,j].get_xlim()[1]
        axes[i,j].text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"MSE: {'{:,.3e}'.format(mse)}", fontsize=10)
    
    elif feature_type == "Multiple" and feature_name == "Coarse-fine volatility":
        feature_real_data = stylfeat_real_df.filter(regex=f"^{feature_name}_").mean()
        feature_fake_data = stylfeat_fake01_df.filter(regex=f"^{feature_name}_").mean()
        
        mse = (np.square(feature_real_data.values - feature_fake_data.values)).mean()
        print("MSE", mse)
        
        axes[i,j].plot(range(-max_lag, max_lag + 1), feature_real_data.values, color="dodgerblue", label="real")
        axes[i,j].plot(range(-max_lag, max_lag + 1), feature_fake_data.values, color="orange", label="fake")
        
        y_min, y_max = axes[i,j].get_ylim()[0], axes[i,j].get_ylim()[1]
        x_min, x_max = axes[i,j].get_xlim()[0], axes[i,j].get_xlim()[1]
        axes[i,j].text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"MSE: {'{:,.3e}'.format(mse)}", fontsize=10)
    
    # axes[i,j].grid()
    axes[i,j].legend(loc='upper right')
    axes[i,j].set_title(feature_name)


