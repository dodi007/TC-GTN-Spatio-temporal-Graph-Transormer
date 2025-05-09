# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, FunctionTransformer
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
import os
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size': 14})

def create_adjacency_matrix(undirected_edges,directed_edges):
    adj_matrix = np.zeros((54, 54), dtype=int)
    for edge in undirected_edges:
        node1, node2 = edge
        adj_matrix[node1-1, node2-1] = 1  # Adjust for 0-indexing
        adj_matrix[node2-1, node1-1] = 1  # Undirected means symmetry

# Populate the adjacency matrix for directed edges
    for edge in directed_edges:
        node1, node2 = edge
        adj_matrix[node1-1, node2-1] = 1  # Directed, so only one direction
    
    return torch.tensor(adj_matrix, dtype=torch.float32)

def generate_directed_edges(n):
    return [(u, n) for u in range(1, n)]

undirected_edges = [
    (1, 2), (1, 4), (1, 7), (2, 3), (2, 4), (2, 5), (3, 5), (3, 21),
    (4, 5), (4, 7), (4, 9), (5, 6), (5, 9), (5, 21), (5, 22), (6, 8),
    (6, 9), (6, 10), (6, 11), (6, 17), (6, 18), (6, 22), (7, 8), (7, 9),
    (8, 9), (8, 11), (10, 11), (10, 15), (10, 18), (11, 12), (11, 15),
    (12, 13), (12, 15), (13, 14), (13, 15), (14, 15), (14, 16), (14, 17), (14, 18), (15, 18),
    (16, 17), (16, 19), (17, 18), (17, 19), (17, 20), (17, 22), (19, 20),
    (20, 21), (20, 22), (21, 22),
    (23, 24), (23, 26), (23, 29), (24, 25), (24, 26), (24, 27), (25, 27), (25, 43),
    (26, 27), (26, 29), (26, 31), (27, 28), (27, 31), (27, 43), (27, 44), (28, 30),
    (28, 31), (28, 32), (28, 33), (28, 39), (28, 40), (28, 44), (29, 30), (29, 31),
    (30, 31), (30, 33), (32, 33), (32, 37), (32, 40), (33, 34), (33, 37),
    (34, 35), (34, 37), (35, 36), (35, 37), (36, 37), (36, 38), (36, 39), (36, 40), (37, 40),
    (38, 39), (38, 41), (39, 40), (39, 41), (39, 42), (39, 44), (41, 42),
    (42, 43), (42, 44), (43, 44)
]

filtered_edges = [
    (u, v) for (u, v) in undirected_edges
    if u <= 9 and v <= 9
]

directed_edges = [
    (1, 45), (2, 45), (3, 45), (4, 45), (5, 45), (6, 45), (7, 45), (8, 45),
    (7, 47), (8, 47), (9, 47), (10, 46), (11, 46), (11, 48), (19, 50), (20, 50),
    (21, 50), (22, 50), (16, 51), (17, 51), (18, 51), (14, 51), (15, 51),
    (13, 52), (12, 53),
    (23, 45), (24, 45), (25, 45), (26, 45), (27, 45), (28, 45), (29, 45), (30, 45),
    (29, 47), (30, 47), (31, 47), (32, 46), (33, 46), (33, 48), (41, 50), (42, 50),
    (43, 50), (44, 50), (38, 51), (39, 51), (40, 51), (36, 51), (37, 51),
    (35, 52), (34, 53), (45,46),(47, 48), (48, 49), (50, 51), (46, 51),
    (49, 51), (51, 52), (52, 53),
    (54, 51), (2, 54), (3, 54), (5, 54), (6, 54), (21, 54), (22, 54),
    (24, 54), (25, 54), (27, 54), (28, 54), (43, 54), (44, 54)
]


a = create_adjacency_matrix(undirected_edges, directed_edges)
edge_index, edge_attr = dense_to_sparse(a)

def read_data_separately(file_path, q_stations, m_stations):
    
    xls = pd.ExcelFile(file_path)
    df_p = pd.read_excel(xls, 'p_eqm_meteostations', parse_dates=[0], index_col = 0)
    df_t = pd.read_excel(xls, 'Tsr_ERA5_ds_meteostations', parse_dates=[0], index_col = 0)
    df_q = pd.read_excel(xls, 'q_sve', parse_dates=[0], index_col = 0)
    
    df_p = df_p[m_stations]
    df_t = df_t[m_stations]
    df_q = df_q[q_stations]
    
    return df_p, df_t, df_q
    

def reorder_columns(data, num_nodes):
    
    num_features = data.shape[1] // num_nodes
    node_order = []

    for i in range(num_features):
        for j in range(num_nodes):
            node_order.append(j * num_features + i)

    reordered_data = data[:, node_order]
    return reordered_data

def create_node_features_tensor(data, num_nodes):
    
    num_features = data.shape[1] // num_nodes
    node_features_list = []

    for i in range(num_nodes):
        node_features_list.append(data[:, i * num_features:(i + 1) * num_features])

    node_features_tensor = torch.stack([torch.tensor(node_features, dtype=torch.float32) for node_features in node_features_list])
    return node_features_tensor  

def create_node_target_tensor(data, num_nodes):
    
    node_target_list = []
    
    for i in range(num_nodes):
        node_target_list.append(data[:, i])
        
    node_target_tensor = torch.stack([torch.tensor(node_targets, dtype=torch.float32) for node_targets in node_target_list])
    return node_target_tensor

def scale_data(data, scaler = None):
    
    if scaler:
        scaled_data = scaler.transform(data)
        return scaled_data
        
    else:        
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler


def load_data_val(file_path, q_stations, m_stations, lag, forecast):
    
    df_p, df_t, df_q = read_data_separately(file_path, q_stations, m_stations)
    
    ind = df_q.index.intersection(df_p.index)
    ind = ind.intersection(df_t.index)
    df_q = df_q.loc[ind]
    df_p = df_p.loc[ind]
    df_t = df_t.loc[ind]
    
    num_m_nodes = len(m_stations)
    num_q_nodes = len(q_stations)
    
    with open("scaler_prec.pkl", "rb") as file:
        scaler_prec = pickle.load(file)
    
    with open("scaler_temp.pkl", "rb") as file:
        scaler_temp = pickle.load(file)

    with open("scaler_target.pkl", "rb") as file:
        scaler_flow = pickle.load(file)
    
    scaled_target = scale_data(df_q.values.flatten().reshape(-1,1),scaler_flow)
    scaled_target = pd.DataFrame(scaled_target.reshape(df_q.shape),columns=df_q.columns)  
    scaled_prec = scale_data(df_p.values.flatten().reshape(-1,1),scaler_prec)
    scaled_prec = pd.DataFrame(scaled_prec.reshape(df_p.shape),columns=df_p.columns)
    scaled_temp = scale_data(df_t.values.flatten().reshape(-1,1),scaler_temp)
    scaled_temp = pd.DataFrame(scaled_temp.reshape(df_t.shape),columns=df_t.columns)
    num_nodes = num_m_nodes*2 + num_q_nodes
    scaled_exog = np.concatenate([scaled_prec, scaled_temp], axis=1)
    
    test = prepare_data_val(scaled_exog, scaled_target, df_q, lag, num_nodes, num_m_nodes, num_q_nodes, forecast)
    
    return test

def prepare_data_val(data, target_input, target, n_steps, num_nodes, num_m_nodes, num_q_nodes, forecast):
    
    input_data = np.concatenate((data[forecast:],target_input[:-forecast]),axis=1)
    output_data = np.concatenate((data,target),axis=1)
   
    reordered_data = reorder_columns(input_data, num_nodes)
    node_features_tensor = create_node_features_tensor(reordered_data, num_nodes)
    node_target_tensor = create_node_target_tensor(output_data, num_nodes)
    
    data_list = []
    num=0
    
    for i in range(0, len(data) - n_steps - forecast):
        
        graph_data = Data(x=node_features_tensor[:, i:i + n_steps, :].reshape(num_nodes, (input_data.shape[1] // num_nodes) * n_steps),  
                          edge_index=edge_index, edge_attr=edge_attr, y=node_target_tensor[:, i + n_steps:i + n_steps + forecast])
        data_list.append(graph_data)

    batch_test = Batch.from_data_list(data_list)
      
    return batch_test
        
def load_data_separately(file_path, q_stations, m_stations, lag, forecast, val_day,test_day,end_day, batch_size = None, use_temp = False):
    
    df_p, df_t, df_q = read_data_separately(file_path, q_stations, m_stations)
    
    ind = df_q.index.intersection(df_p.index)
    ind = ind.intersection(df_t.index)
    df_q = df_q.loc[ind]
    df_p = df_p.loc[ind]
    df_t = df_t.loc[ind]
    
    num_m_nodes = len(m_stations)
    num_q_nodes = len(q_stations)
    
    exogenous = df_p

    val_ind = np.where(exogenous.index == val_day)[0][0]
    test_ind = np.where(exogenous.index == test_day)[0][0]
    end_ind = np.where(exogenous.index == end_day)[0][0]
    

    #preparing target data
    
    train_target = df_q[:val_ind]
    val_target = df_q[val_ind:test_ind]
    test_target = df_q[test_ind:end_ind+1]
    
    scaled_train_target, scaler_target = scale_data(train_target.values.flatten().reshape(-1,1))
    with open("scaler_target.pkl", "wb") as file:
        pickle.dump(scaler_target, file)
    scaled_train_target = pd.DataFrame(scaled_train_target.reshape(train_target.shape),columns=train_target.columns)
    scaled_val_target = scale_data(val_target.values.flatten().reshape(-1,1),scaler_target)
    scaled_val_target = pd.DataFrame(scaled_val_target.reshape(val_target.shape),columns=val_target.columns)
    scaled_test_target = scale_data(test_target.values.flatten().reshape(-1,1),scaler_target)
    scaled_test_target = pd.DataFrame(scaled_test_target.reshape(test_target.shape),columns=test_target.columns)
         
    scaled_target = np.concatenate([scaled_train_target, scaled_val_target, scaled_test_target], axis=0)
    
    #preparing meteo data
    
    train_exog = exogenous[:val_ind]
    val_exog = exogenous[val_ind:test_ind]
    test_exog = exogenous[test_ind:end_ind+1]
    
    scaled_train_exog, scaler = scale_data(train_exog.values.flatten().reshape(-1,1))
    with open("scaler_prec.pkl", "wb") as file:
        pickle.dump(scaler, file)
    scaled_train_exog = pd.DataFrame(scaled_train_exog.reshape(train_exog.shape),columns=train_exog.columns)
    scaled_val_exog = scale_data(val_exog.values.flatten().reshape(-1,1),scaler)
    scaled_val_exog = pd.DataFrame(scaled_val_exog.reshape(val_exog.shape),columns=val_exog.columns)
    scaled_test_exog = scale_data(test_exog.values.flatten().reshape(-1,1),scaler)
    scaled_test_exog = pd.DataFrame(scaled_test_exog.reshape(test_exog.shape),columns=test_exog.columns)
    
    scaled_exog_p = np.concatenate([scaled_train_exog, scaled_val_exog, scaled_test_exog], axis=0)
    

    if use_temp:
           
        exogenous = df_t
        
        train_exog = exogenous[:val_ind]
        val_exog = exogenous[val_ind:test_ind]
        test_exog = exogenous[test_ind:end_ind+1]
                
        scaled_train_exog, scaler = scale_data(train_exog.values.flatten().reshape(-1,1))
        with open("scaler_temp.pkl", "wb") as file:
            pickle.dump(scaler, file)
        scaled_train_exog = pd.DataFrame(scaled_train_exog.reshape(train_exog.shape),columns=train_exog.columns)
        scaled_val_exog = scale_data(val_exog.values.flatten().reshape(-1,1),scaler)
        scaled_val_exog = pd.DataFrame(scaled_val_exog.reshape(val_exog.shape),columns=val_exog.columns)
        scaled_test_exog = scale_data(test_exog.values.flatten().reshape(-1,1),scaler)
        scaled_test_exog = pd.DataFrame(scaled_test_exog.reshape(test_exog.shape),columns=test_exog.columns)
        
        scaled_exog_t = np.concatenate([scaled_train_exog, scaled_val_exog, scaled_test_exog], axis=0)
        
        scaled_exog = np.concatenate([scaled_exog_p, scaled_exog_t], axis=1)
        
        num_nodes = num_m_nodes*2 + num_q_nodes
        
    else:

        scaled_exog = scaled_exog_p
        num_nodes = num_m_nodes + num_q_nodes
        
    
    df_q = df_q[:end_ind+1]
    train, val, test, all_train = prepare_data_separately(scaled_exog, scaled_target, df_q, val_ind, test_ind, lag, num_nodes, num_m_nodes, num_q_nodes, forecast, batch_size)
    return train, val, test, all_train


def prepare_data_separately(data, target_input, target,  val_index, test_index, n_steps, num_nodes, num_m_nodes, num_q_nodes, forecast, batch_size = None):
    
    input_data = np.concatenate((data[forecast:],target_input[:-forecast]),axis=1)
    output_data = np.concatenate((data,target),axis=1)
   
    reordered_data = reorder_columns(input_data, num_nodes)
    node_features_tensor = create_node_features_tensor(reordered_data, num_nodes)
    node_target_tensor = create_node_target_tensor(output_data, num_nodes)
    
    data_list = []
    num=0
    
    if batch_size:
        
        batch_train=[]
        data_list_batch=[]
        
        
        for i in range(val_index - n_steps):
            
            graph_data = Data(x=node_features_tensor[:, i:i + n_steps, :].reshape(num_nodes, (input_data.shape[1] // num_nodes) * n_steps),  
                              edge_index=edge_index, edge_attr=edge_attr, y=node_target_tensor[:, i + n_steps:i + n_steps + forecast])
            data_list.append(graph_data)
            data_list_batch.append(graph_data)
            num = num + 1
            if num%batch_size==0:
                batch = Batch.from_data_list(data_list_batch)
                data_list_batch=[]
                batch_train.append(batch)
                  
        batch = Batch.from_data_list(data_list_batch)
        batch_train.append(batch)
        
    data_list = []
    for i in range(val_index - n_steps):

        graph_data = Data(x=node_features_tensor[:, i:i + n_steps, :].reshape(num_nodes, (input_data.shape[1] // num_nodes) * n_steps),  
                          edge_index=edge_index, edge_attr=edge_attr, y=node_target_tensor[:, i + n_steps:i + n_steps + forecast])
        data_list.append(graph_data)

    all_train = Batch.from_data_list(data_list)
    
    data_list = []
    
    for i in range(val_index - n_steps, test_index - n_steps):
        
        graph_data = Data(x=node_features_tensor[:, i:i + n_steps, :].reshape(num_nodes, (input_data.shape[1] // num_nodes) * n_steps),  
                          edge_index=edge_index, edge_attr=edge_attr, y=node_target_tensor[:, i + n_steps:i + n_steps + forecast])
        data_list.append(graph_data)

    batch_val = Batch.from_data_list(data_list)
    
    data_list = []
    
    for i in range(test_index - n_steps, len(data) - n_steps - forecast):
        
        graph_data = Data(x=node_features_tensor[:, i:i + n_steps, :].reshape(num_nodes, (input_data.shape[1] // num_nodes) * n_steps),  
                          edge_index=edge_index, edge_attr=edge_attr, y=node_target_tensor[:, i + n_steps:i + n_steps + forecast])
        data_list.append(graph_data)

    batch_test = Batch.from_data_list(data_list)
    
      
    return batch_train, batch_val, batch_test, all_train


def plot_scatter(y_true, y_pred, stations, forecast,dataset):
    num_nodes = len(stations)
    
    for i in range(num_nodes):
        plt.figure(figsize=(8, 8))
        for j in range(forecast):
            plt.scatter(y_true[i][:,j].flatten(), y_pred[i][:,j].flatten(), label=f'{j + 1} day(s) ahead', alpha=0.3, s=25)
        plt.plot([0,np.max(y_true[i].flatten())],[0,np.max(y_true[i].flatten())], 'k-', label='45 degree line')
        plt.xlim(0, np.max(y_true[i].flatten()))
        plt.ylim(0, np.max(y_true[i].flatten()))
        
        for j in [0, 4]:  # indices for the first and fifth days
            model = LinearRegression()
            model.fit(y_true[i][:,j].flatten().reshape(-1, 1), y_pred[i][:,j].flatten())
            line_start = min(y_true[i][:,j].flatten())
            line_end = max(y_true[i][:,j].flatten())
            x_line = np.array([line_start, line_end])
            y_line = model.predict(x_line.reshape(-1, 1)) 
            plt.plot(x_line, y_line, '--', label=f'Regression Line {j + 1} day(s)')        
        
        plt.title(f'{stations[i]} - GNN Scatter Plot')
        plt.xlabel('Observed discharge (m³/s)')
        plt.ylabel('Predicted discharge (m³/s)')
        plt.grid(False)
        plt.legend(fontsize=10)

        plot_filename = os.path.join(os.getcwd(), f'slike/{dataset}_scatter_plot_{stations[i]}.png')
        plt.savefig(plot_filename, format='png', bbox_inches='tight')
        plt.show()
            # plt.close()  # Close the figure to avoid displaying it
        
def plot_forecast_vs_actual(y_true, y_pred, stations, forecast, start_day, end_day,dataset):
    num_nodes = len(stations)
    # Create a range of dates from start to end
    date_range = pd.date_range(start=start_day, end=end_day)
    for i in range(num_nodes):
        plt.figure(figsize=(12, 6))
        for j in range(forecast):        
            # Plot actual test values for the current station
            shifted_date_range = date_range + pd.Timedelta(days=j)
            if j==0:     
                plt.plot(shifted_date_range, y_true[i][:,j].flatten(), label='Observed', color='black', lw=1)
            else:
                plt.plot(shifted_date_range, y_true[i][:,j].flatten(), color='black', lw=1)
            # Plot forecasted values for the current station
            plt.plot(shifted_date_range, y_pred[i][:,j].flatten(), linestyle='--', alpha=0.9, lw=1, label=f'{j + 1} day(s) ahead')

            # Adding title and labels
        plt.title(f'{stations[i]} - GNN Time Series Plot', fontsize=16)
        plt.xlabel('Time',fontsize=14)
        plt.ylabel('Discharge (m³/s)',fontsize=14)
        plt.xticks(rotation=45)

        # Display legend
        plt.legend(fontsize=12, loc='upper right')

        # Optional: Adding grid for better readability
        plt.grid(False)

        # Show the plot

        plot_filename = os.path.join(os.getcwd(), f'slike/{dataset}_time_plot_{stations[i]}.png')
        plt.savefig(plot_filename, format='png', bbox_inches='tight')
        plt.show()
 
        
