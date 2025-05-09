import torch
from data_utils import load_data_separately
import torch.nn as nn
import copy 
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, max_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from models import SpatioTemporalForecaster

m_stations = ['List of names of meteorologica stations']
q_stations = ['List of names of hydrological stations']

all_stations = m_stations + q_stations
file_path = 'Path to the dataset'
forecast = 5
lag = 7
train_ratio = 0.94288
scale = False
use_temp = True
val_day = 'Date to start validation dataset'
test_day = 'Date to start training dataset'
end_day = 'End date of the dataset'
batch_size = 256

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

class QuantileLoss(nn.Module):
    def __init__(self, quantile: float):
        super(QuantileLoss, self).__init__()
        assert 0 < quantile < 1, "Quantile should be between 0 and 1"
        self.quantile = quantile

    def forward(self, predictions, targets):
        errors = targets - predictions
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)

def train(model, train, val):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.0004)

    criterion = QuantileLoss(quantile = 0.7)
    best_val_loss = float('inf')
    best_model_state = None
    """
    Param:
    - model: The neural network model to be trained.
    - data: The graph data, which includes node features (`data.x`) and edge
        indices (`data.edge_index`).
    - train_idx: Indices of nodes in the training set.
    - optimizer: The optimizer used for updating model parameters.
    - loss_fn: The loss function used to compute the training loss.
    """
    
    num=0
    train_loss = []
    val_losses = []
    
    if use_temp:
        num_m_nodes = len(m_stations)*2
    else:
        num_m_nodes = len(m_stations)
    num_q_nodes = len(q_stations)
    num_nodes = num_m_nodes + num_q_nodes
    
    device = 'cuda:1'
    
    weights = torch.tensor([1.0] * num_m_nodes + [10.0] * 2 + [15.0] *3 + [15.0] *1 + [5.0] *1 + [5.0]*2 + [5.0]*1, device=device)
    
    val = val.to(device)
    for epoch in range(200):
        model.train()
        total_loss = 0
        i = 0
        
        for data in train: 
            
            data = data.to(device)
            optimizer.zero_grad()
            num_examples = data.x.shape[0]//num_nodes
            output = model(data.x.view(num_examples,num_nodes,lag),data.edge_index)
            output = output.view(-1, num_nodes, forecast)
            loss_per_node = []
            for node_idx in range(num_m_nodes,num_nodes):
                node_output = output[:, node_idx, :]  # Get output for each node across the forecast dimension
                node_target = data.y.view(-1, num_nodes, forecast)[:, node_idx, :]  # Corresponding target for each node
                node_loss = criterion(node_output, node_target) * weights[node_idx]  # Calculate loss for each node
                
                loss_per_node.append(node_loss)
               
            loss = torch.mean(torch.stack(loss_per_node))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            i = i + 1
        # if epoch == 0: print(loss)   
        train_loss.append(total_loss/i)  
        
        model.eval()  # Set model to evaluation mode for validation
        with torch.no_grad():  # Disable gradient computation during validation
            num_examples = val.x.shape[0]//num_nodes
            output = model(val.x.view(num_examples,num_nodes,lag), val.edge_index)
            output = output.view(-1, num_nodes, forecast)
            val_loss_per_node = []
            for node_idx in range(num_m_nodes,num_nodes):
                node_output = output[:, node_idx, :]  # Get output for each node across the forecast dimension
                node_target = val.y.view(-1, num_nodes, forecast)[:, node_idx, :]  # Corresponding target for each node
                node_loss = criterion(node_output, node_target) * weights[node_idx]  # Calculate loss for each node
            
                val_loss_per_node.append(node_loss)
            
            val_loss = torch.mean(torch.stack(val_loss_per_node)).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num=0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model.pth') #save best model
            print(f"Epoch {epoch + 1}, Loss: {total_loss/i:.6f}")
            print(f"Epoch {epoch + 1}, Val_Loss: {val_loss:.6}")
        else:
            num=num+1
        val_losses.append(val_loss)
        if num==50:
            break

    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_losses)
    model.load_state_dict(best_model_state)
    
    return model

train_batch, val_batch, test_batch, all_train = load_data_separately(file_path, q_stations, m_stations, lag, forecast, val_day, test_day, end_day, batch_size = 256, use_temp = use_temp)

model_tcn = SpatioTemporalForecaster(temporal_type = 'tcn').to(device)

model = train(model_tcn, train_batch, val_batch)

@torch.no_grad()
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def test(model,test,dataset):
    
    model.eval()
    
    if use_temp:
        num_m_nodes = len(m_stations)*2
    else:
        num_m_nodes = len(m_stations)
        
    num_q_nodes = len(q_stations)   
    num_nodes = num_m_nodes + num_q_nodes
    
    y_pred = [[] for _ in range(num_q_nodes)]
    y_true = [[] for _ in range(num_q_nodes)]
    
    with torch.no_grad():
        test = test.to(device)
        # output = model(test.x, test.edge_index, test.edge_attr)
        num_examples = test.x.shape[0]//num_nodes
        output = model(test.x.view(num_examples,num_nodes,lag), test.edge_index)
        output = output.view(-1, num_nodes, forecast)
        test.y = test.y.view(-1, num_nodes, forecast)
        
        for i in range(0, len(output)):
            for node in range(num_m_nodes,num_m_nodes+num_q_nodes):
                y_pred[node-num_m_nodes].append(output[i, node, :].cpu().detach().numpy())
                y_true[node-num_m_nodes].append(test.y[i, node, :].cpu().detach().numpy())

    y_pred = [np.array(pred) for pred in y_pred]
    y_true = [np.array(true) for true in y_true]
    
    rmse_scores = []
    r2_scores = []
    max_errors = []
    mape_errors = []
    mae_errors = []
    
    rmse_scores = [[] for _ in range(num_q_nodes)]
    r2_scores = [[] for _ in range(num_q_nodes)]
    max_errors = [[] for _ in range(num_q_nodes)]
    max_errors_pred = [[] for _ in range(num_q_nodes)]
    max_errors_true = [[] for _ in range(num_q_nodes)]
    max_errors_dates = [[] for _ in range(num_q_nodes)]
    mae_errors = [[] for _ in range(num_q_nodes)]
    std_ae_errors = [[] for _ in range(num_q_nodes)]
    mape_errors = [[] for _ in range(num_q_nodes)]
    
    date_range = pd.date_range(start=test_day, end=end_day)
    
    for j in range(forecast):
        for i, station in enumerate(q_stations):
        
            rmse_scores[i].append(rmse(y_true[i][:,j], y_pred[i][:,j]))
            r2_scores[i].append(r2_score(y_true[i][:,j], y_pred[i][:,j]))
            max_errors[i].append(max_error(y_true[i][:,j], y_pred[i][:,j]))
            ind = np.argmax(np.abs(y_true[i][:,j] - y_pred[i][:,j]))
            max_errors_dates[i].append(date_range[ind])
            max_errors_true[i].append(y_true[i][ind,j])
            max_errors_pred[i].append(y_pred[i][ind,j])
            std_ae_errors[i].append(np.std(np.abs(y_true[i][:,j] - y_pred[i][:,j])))
            mae_errors[i].append(mean_absolute_error(y_true[i][:,j], y_pred[i][:,j]))
            mape_errors[i].append(mean_absolute_percentage_error(y_true[i][:,j], y_pred[i][:,j]))

    # Calculate the 75th percentile for each station separately
    filtered_rmse_scores = [[[] for _ in range(forecast)] for _ in range(num_q_nodes)]
    filtered_mae_scores = [[[] for _ in range(forecast)] for _ in range(num_q_nodes)]
    filtered_mape_scores = [[[] for _ in range(forecast)] for _ in range(num_q_nodes)]
    filtered_r2_scores = [[[] for _ in range(forecast)] for _ in range(num_q_nodes)]

    for i, station in enumerate(q_stations):
        for j in range(forecast):
            station_true_values = y_true[i][:, j]
            station_pred_values = y_pred[i][:, j]
            
            # Calculate the 75th percentile of true values for this station and day
            station_75th_percentile = np.percentile(station_true_values, 75)
            # print(station_75th_percentile)

            # Filter true and predicted values where true values are above the 75th percentile
            filter_mask = station_true_values > station_75th_percentile
            filtered_true = station_true_values[filter_mask]
            filtered_pred = station_pred_values[filter_mask]

            # Calculate errors only for filtered values for each forecast day
            filtered_rmse_scores[i][j] = rmse(filtered_true, filtered_pred) if len(filtered_true) > 0 else np.nan
            filtered_mae_scores[i][j] = mean_absolute_error(filtered_true, filtered_pred) if len(filtered_true) > 0 else np.nan
            filtered_mape_scores[i][j] = mean_absolute_percentage_error(filtered_true, filtered_pred) if len(filtered_true) > 0 else np.nan
            filtered_r2_scores[i][j] = r2_score(filtered_true, filtered_pred) if len(filtered_true) > 0 else np.nan

        # Print the original errors and filtered errors for each station
        print(station)
        print("RMSE Scores:", rmse_scores[i])
        print("R2 Scores:", r2_scores[i])
        print("Max Errors:", max_errors[i])
        print("MAE Errors:", mae_errors[i])
        print("MAPE errors:", mape_errors[i])   

        # Print the filtered errors for each day
        print(f"Filtered RMSE for true values above 75th percentile: {filtered_rmse_scores[i]}")
        print(f"Filtered R2 for true values above 75th percentile: {filtered_r2_scores[i]}")
        print(f"Filtered MAE for true values above 75th percentile: {filtered_mae_scores[i]}")
        print(f"Filtered MAPE for true values above 75th percentile: {filtered_mape_scores[i]}")
        
        
        
        data = {
        "Forecast Day": [j for j in range(1,forecast+1)],
        "Original MAE": mae_errors[i],
        "Std AE":std_ae_errors[i],
        "Original MAPE": mape_errors[i],
        "Original RMSE": rmse_scores[i],
        "Original R2": r2_scores[i],
        "Max Error": max_errors[i],
        "Max Error Dates": max_errors_dates[i],
        "Max Error True": max_errors_true[i],
        "Max Error Forecast": max_errors_pred[i], 
        "Filtered MAE (75th)": filtered_mae_scores[i],
        "Filtered MAPE (75th)": filtered_mape_scores[i],
        "Filtered RMSE (75th)": filtered_rmse_scores[i],
        "Filtered R2 (75th)": filtered_r2_scores[i],
        }
    
        # Create a DataFrame
        df = pd.DataFrame(data)

        # Format numerical values to 3 decimal places
        df = df.round(3)

        # Save DataFrame to a CSV file
        filename = f"rezultati/{station}_metrics_{dataset}.csv"
        # df.to_csv(filename, index=False)
       
    return y_pred, y_true

y_pred, y_true = test(model_tcn, test_batch, dataset = 'test')
