"""
Matching prediction dataframe rows to event prefixes in the original event log.
"""
import os
import pandas as pd
import numpy as np
import pickle
from torch_geometric.data import Dataset, DataLoader
import argparse
from PGTNetutils import eventlog_class_provider, mean_cycle_norm_factor_provider

# To load and work with the graph dataset (a Pytorch Geometric data object)
class CustomDataset(Dataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

parser = argparse.ArgumentParser(description='Handle PGTNet results')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
parser.add_argument('--seed_number', type=str, help='Seed number')
parser.add_argument('--inference_config', type=str, help='Inference configuration')
args = parser.parse_args()
dataset_name = args.dataset_name
seed_number = args.seed_number
inference_config = args.inference_config
num_runs = 5

script_dir = os.path.dirname(os.path.abspath(__file__)) #Root directory for PGTNet
result_folder = os.path.join(script_dir, 'PGTNet results') #path for results of PGTNet for all datasets
dataset_result_folder = os.path.join(result_folder, dataset_name) #path to result for this dataset
seed_folder = 'seed ' + seed_number
dataset_result_seed_folder = os.path.join(dataset_result_folder, seed_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
if not os.path.exists(dataset_result_folder):
    os.makedirs(dataset_result_folder)
if not os.path.exists(dataset_result_seed_folder):
    os.makedirs(dataset_result_seed_folder)
csv_name = dataset_name + '-seed'+ seed_number+ '-PGTNet_results.csv'
csv_path = os.path.join(dataset_result_seed_folder, csv_name)

gps_directory = os.path.dirname(script_dir) #Root directory for GraphGPS
graph_dataset_class_name = eventlog_class_provider(dataset_name)
graph_dataset_folder = os.path.join(gps_directory, 'datasets', graph_dataset_class_name, 'raw')

inference_results_folder = os.path.join(gps_directory, 'results', inference_config)

# check whether there exist already the aggregated file in the relevant folder
if os.path.exists(csv_path):
    final_result_dataframe = pd.read_csv(csv_path)
    print('Aggregate file for the dataset:', dataset_name, 'and for the seed number:', seed_number, 'is already computed.')
else:
    # main body to create aggregated result
    
    #  Import PGTNet's prediction dataframe
    filename = f'{dataset_name}-pgtnet_prediction_dataframe.csv'
    file_path = os.path.join(inference_results_folder, filename)
    if os.path.exists(file_path):
        prediction_dataframe = pd.read_csv(file_path)
    else:
        print(f"File {filename} not found.")  
    
    # Import graph dataset that is used for training and evaluation
    # Load the pickle files and create instances of CustomDataset
    with open(os.path.join(graph_dataset_folder, "train.pickle"), "rb") as f:
        train_dataset = CustomDataset(root=graph_dataset_folder, data_list=pickle.load(f))
    with open(os.path.join(graph_dataset_folder, "val.pickle"), "rb") as f:
        val_dataset = CustomDataset(root=graph_dataset_folder, data_list=pickle.load(f))
    with open(os.path.join(graph_dataset_folder, "test.pickle"), "rb") as f:
        test_dataset = CustomDataset(root=graph_dataset_folder, data_list=pickle.load(f))
    combined_dataset = train_dataset + val_dataset + test_dataset
    
    # use graph dataset to create a pandas dataframe
    num_node_list = []
    num_edge_list = []
    real_cycle_time_list = []
    cid_list = []
    pl_list = []
    for graph in combined_dataset:
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        real_cycle_time = graph.y.item()  # Assuming y is a single value tensor
        cid = graph.cid
        pl = graph.pl
        num_node_list.append(num_nodes)
        num_edge_list.append(num_edges)
        real_cycle_time_list.append(real_cycle_time)
        cid_list.append(cid)
        pl_list.append(pl)
    data = {"num_node": num_node_list, "num_edge": num_edge_list, "real_cycle_time": real_cycle_time_list,
            "cid": cid_list, "pl": pl_list}
    initial_dataframe = pd.DataFrame(data)
    
    # match predictions and data used for training. 
    final_result_dataframe = pd.DataFrame(columns=list(initial_dataframe.columns) + ['predicted_cycle_time', 'MAE-days'])
    # Iterate through the initial DataFrame
    for index, initial_row in initial_dataframe.iterrows():
        num_node = initial_row['num_node']
        num_edge = initial_row['num_edge']
        real_cycle_time_init = initial_row['real_cycle_time']    
        # Filter the prediction DataFrame based on "num_node" and "num_edge"
        filtered_prediction = prediction_dataframe[(prediction_dataframe['num_node'] == num_node) & (prediction_dataframe['num_edge'] == num_edge)]    
        if not filtered_prediction.empty:
            # Find the closest match based on "real_cycle_time" within the filtered subset
            closest_match = filtered_prediction.iloc[np.argmin(np.abs(filtered_prediction['real_cycle_time'] - real_cycle_time_init))] 
            # Add the matched row to the final result DataFrame with "predicted_cycle_time"
	    # if you are using older version of pandas you might need to replace "_append" by "append" in the following lines.
            match_data = initial_row._append(pd.Series([closest_match['predicted_cycle_time']], index=['predicted_cycle_time']))
            match_data = initial_row._append(pd.Series([closest_match['MAE-days']], index=['MAE-days']))
            final_result_dataframe = final_result_dataframe._append(match_data, ignore_index=True)
    
    final_result_dataframe.to_csv(csv_path, index=False)






