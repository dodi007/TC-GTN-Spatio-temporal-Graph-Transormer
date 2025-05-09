# Spatio-Temporal Graph Neural Network for Hydrological Forecasting

## Overview

This repository contains a **Graph Neural Network (GNN)** model for **multi-step hydrological forecasting**, specifically designed to predict river discharge values at multiple stations. The model combines:

- **Temporal processing** (using either TCN or LSTM architectures)
- **Spatial processing** (using Transformer-based graph neural networks)
- **Graph-based relationships** between meteorological and hydrological stations

Key features:
- Multi-day ahead forecasting (configurable forecast horizon)
- Handling of both directed and undirected relationships between stations
- Custom quantile loss function for robust training
- Comprehensive evaluation metrics including flood-focused metrics (75th percentile performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hydrological-forecasting-gnn.git
cd hydrological-forecasting-gnn
```
2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependecies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
  - Place your Excel data file in the root directory
  - Modify m_stations and q_stations lists in main_training.py with your station names
  - Update the file path and date parameters in main_training.py
2. Run the training:
```bash
python main_training.py
```   
3. Outputs
  - Model weights saved as best_model.pth
  - Visualizations can be saved also, edit test function
  - Performance metrics can be saved, edit test function
