import yaml
import os
import os.path as osp
import numpy as np
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import bisect
import random
from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data
import pickle
import argparse
from PGTNetutils import eventlog_class_provider


# Read user inputs from .yml file
def read_user_inputs(file_path):
    with open(file_path, 'r') as f:
        user_inputs = yaml.safe_load(f)
    return user_inputs

# Add string "case:" to categorical and numerical case attributes
def case_full_func(instance_att, instance_num_att):
    case_attributes_full, case_num_ful = [], []
    for case_attribute in instance_att:
        case_attribute = 'case:' + case_attribute
        case_attributes_full.append(case_attribute)
    for case_attribute in instance_num_att:
        case_attribute = 'case:' + case_attribute
        case_num_ful.append(case_attribute)        
    return case_attributes_full, case_num_ful
        
# Get the number of active cases (given a timestamp)
def ActiveCase(L1, L2, T):
    number_of_active_cases = bisect.bisect_right(L1, T) - bisect.bisect_right(L2, T)
    return number_of_active_cases

# Get train, validation, test data split, as well as duration of the longest case in training data
def train_test_split_func (log, event_log, split_ratio = [0.64, 0.16, 0.2]):
    start_dates, end_dates, durations, case_ids = [], [], [], []
    train_validation_index = int(len(log) * (split_ratio[0]+split_ratio[1]))
    train_index = int(len(log) * split_ratio[0])
    for i in range (len(log)):
        current_case = log[i]
        current_length = len(current_case)
        start_dates.append(current_case[0].get('time:timestamp'))
        end_dates.append(current_case[current_length-1].get('time:timestamp'))
        durations.append((current_case[current_length-1].get('time:timestamp') - 
                          current_case[0].get('time:timestamp')).total_seconds()/3600/24)
        case_ids.append(current_case.attributes.get('concept:name'))        
    combined_data = list(zip(start_dates, end_dates, durations, case_ids))
    sorted_data = sorted(combined_data, key=lambda x: x[0])
    sorted_start_dates, sorted_end_dates, sorted_durations, sorted_case_ids = zip(*sorted_data)        
    train_case_ids = sorted_case_ids[:train_index]
    validation_case_ids = sorted_case_ids[train_index:train_validation_index]
    train_validation_case_ids = sorted_case_ids[:train_validation_index]
    test_case_ids = sorted_case_ids[train_validation_index:]
    train_validation_durations = sorted_durations[:train_validation_index]        
    max_case_duration = max(train_validation_durations)
    training_dataframe = pm4py.filter_trace_attribute_values(event_log, 'case:concept:name',
                                                             train_case_ids, 
                                                             case_id_key='case:concept:name')
    validation_dataframe = pm4py.filter_trace_attribute_values(event_log, 'case:concept:name',
                                                               validation_case_ids,
                                                               case_id_key='case:concept:name')
    test_dataframe = pm4py.filter_trace_attribute_values(event_log, 'case:concept:name',
                                                         test_case_ids,
                                                         case_id_key='case:concept:name')
    train_validation_dataframe = pm4py.filter_trace_attribute_values(event_log, 'case:concept:name',
                                                                     train_validation_case_ids,
                                                                     case_id_key='case:concept:name')
    training_event_log = pm4py.convert_to_event_log(training_dataframe)
    validation_event_log = pm4py.convert_to_event_log(validation_dataframe)
    test_event_log = pm4py.convert_to_event_log(test_dataframe)
    train_validation_event_log = pm4py.convert_to_event_log(train_validation_dataframe)
    return training_dataframe, validation_dataframe, test_dataframe, train_validation_dataframe,\
        training_event_log, validation_event_log, test_event_log, train_validation_event_log,\
            max_case_duration, sorted_start_dates, sorted_end_dates


# Get other relevant global information from training data + required one hot encoders
def get_global_stat_func (training_validation_log, training_validation_df, case_num_ful,
                          event_num_att, event_attributes, case_attributes_full, event_log, log):
    """
    node_class_dict (activity classes), keys: tuple of activity/lifecycle, value: integer representation
    max_case_df: maximum number of same df relationship in one case
    max_active_cases: maximum number of concurrent active cases in training-validation sets
    """
    max_case_df = 0  
    node_class_dict = {}
    node_class_rep = 0
    start_dates, end_dates = [], []
    # Get node_class_dict, max_case_df
    for case_counter in range (len(log)):
        df_dict = {} # Dict: all activity-class df relationships and their frequencies
        current_case = log[case_counter]
        case_length = len(current_case)
        if case_length > 1:
            # iterate over all events of the case, collect df information
            for event_counter in range(case_length-1): 
                source_class = (current_case[event_counter].get('concept:name'), 
                                current_case[event_counter].get('lifecycle:transition'))
                target_class = (current_case[event_counter+1].get('concept:name'), 
                                current_case[event_counter+1].get('lifecycle:transition'))
                df_class = (source_class, target_class)
                if df_class in df_dict:
                    df_dict[df_class] += 1
                else:
                    df_dict[df_class] = 1
                if not (source_class in node_class_dict):
                    node_class_dict[source_class] = node_class_rep
                    node_class_rep += 1                                 
            if max((df_dict).values()) > max_case_df:
                max_case_df = max((df_dict).values())
    
    # Iterate over train-val data, get list of start and end dates, use them to get max_active_cases
    for case_counter in range (len(training_validation_log)):
        current_case = log[case_counter]
        case_length = len(current_case)
        start_dates.append(current_case[0].get('time:timestamp'))
        end_dates.append(current_case[case_length-1].get('time:timestamp'))          
    sorted_start_dates = sorted(start_dates)
    sorted_end_dates = sorted(end_dates)
    max_active_cases = 0
    unique_timestamps = list(training_validation_df['time:timestamp'].unique())
    for any_time in unique_timestamps:
        cases_in_system = ActiveCase(sorted_start_dates, sorted_end_dates, any_time)
        if cases_in_system > max_active_cases:
            max_active_cases = cases_in_system
    
    # obtain number of numerical case attributes + 3 lists for min/max/avg values for each attribute
    min_num_list, max_num_list, avg_num_list = [], [], []
    case_num_card = len(case_num_ful)
    for num_att in case_num_ful:
        unique_values = training_validation_df[num_att].dropna().tolist()
        unique_values_float = [float(val) for val in unique_values]
        min_num_list.append(float(min(unique_values_float)))
        max_num_list.append(float(max(unique_values_float)))
        avg_num_list.append(float(sum(unique_values_float)/len(unique_values_float)))
    # obtain number of numerical event attributes + 3 lists for min/max/avg values for each attribute
    event_min_num_list, event_max_num_list, event_avg_num_list = [], [], []
    event_num_card = len(event_num_att)
    for num_att in event_num_att:
        unique_values = training_validation_df[num_att].dropna().tolist()
        unique_values_float = [float(val) for val in unique_values]
        event_min_num_list.append(float(min(unique_values_float)))
        event_max_num_list.append(float(max(unique_values_float)))
        event_avg_num_list.append(float(sum(unique_values_float)/len(unique_values_float)))        
    
    # List of one-hot encoders: categorical event attributes of intrest
    attribute_encoder_list = []
    attribute_cardinality = 0
    for event_attribute in event_attributes:
        unique_values = list(event_log[event_attribute].unique())
        att_array = np.array(unique_values)
        att_enc = OneHotEncoder(handle_unknown='ignore')
        att_enc.fit(att_array.reshape(-1, 1))
        attribute_encoder_list.append(att_enc)
        attribute_cardinality += len(unique_values)
    # List of one-hot encoders (for case attributes of intrest)
    case_encoder_list = []
    case_cardinality = 0
    for case_attribute in case_attributes_full:
        unique_values = list(event_log[case_attribute].unique())
        att_array = np.array(unique_values)
        att_enc = OneHotEncoder(handle_unknown='ignore')
        att_enc.fit(att_array.reshape(-1, 1))
        case_encoder_list.append(att_enc)
        case_cardinality += len(unique_values)     
           
    # Get node and edge dimensions
    node_dim = len(node_class_dict.keys()) # size for node featuers
    edge_dim  = attribute_cardinality + case_cardinality + case_num_card + event_num_card + 7 
    return node_class_dict, max_case_df, max_active_cases, min_num_list, max_num_list,\
        event_min_num_list, event_max_num_list, attribute_encoder_list, case_encoder_list,\
            node_dim, edge_dim, avg_num_list, event_avg_num_list

# main function for converting an event log into graph dataset
def graph_conversion_func (split_log, removed_cases, idx, data_list, case_attributes, case_encoder_list,
                           case_num_att, min_num_list, max_num_list, event_attributes, event_num_att,
                           target_normalization, max_time_norm, node_class_dict, edge_dim, max_case_df,
                           sorted_start_dates, sorted_end_dates, max_active_cases,
                           attribute_encoder_list, event_min_num_list, event_max_num_list, avg_num_list,
                           event_avg_num_list):
    
    # iterate over cases, and transform them if they have at least three events
    for case_counter in range(len(split_log)):
        current_case = split_log[case_counter]
        case_id = split_log[case_counter].attributes.get('concept:name')   
        case_length = len(current_case)
        if case_length < 3:
            removed_cases.append(case_id)
        else:
            
            case_level_feat = np.empty((0,)) # collect all case-level information
            
            # first categorical attributes
            for att_index in range(len(case_attributes)):
                case_att = split_log[case_counter].attributes.get(case_attributes[att_index])
                case_att = str(case_att)
                #case_att_enc = case_encoder_list[att_index].transform(np.array(case_att).reshape(-1, 1)).toarray()
                case_att_enc = case_encoder_list[att_index].transform([[case_att]]).toarray()
                case_att_enc = case_att_enc.reshape(-1)
                case_level_feat = np.append(case_level_feat, case_att_enc)  
            
            # now, numerical attributes
            for att_index in range(len(case_num_att)):
                case_att = float(split_log[case_counter].attributes.get(case_num_att[att_index]))
                # impute NaN values with average value for that attribute!
                if np.isnan(case_att):
                    case_att_normalized = (avg_num_list[att_index] - min_num_list[att_index])/(max_num_list[att_index]- min_num_list[att_index])
                else:
                    case_att_normalized = (case_att - min_num_list[att_index])/(max_num_list[att_index]- min_num_list[att_index])
                case_level_feat = np.append(case_level_feat, np.array(case_att_normalized)) 
            
            # collect all events of the case, compute case start, end time
            case_events = split_log[case_counter][:]  
            case_start = split_log[case_counter][0].get('time:timestamp') 
            case_end = split_log[case_counter][case_length-1].get('time:timestamp')
        
            # collect activity classes, timestamps, and all attributes of intrest for each event
            collection_lists = [[] for _ in range(len(event_attributes)+len(event_num_att)+2)]
            for event_index in range(case_length):            
                current_event = case_events[event_index]
                collection_lists[0].append((current_event.get('concept:name'),current_event.get('lifecycle:transition')))
                collection_lists[1].append(current_event.get('time:timestamp'))
                for attribute_counter in range (2,len(event_attributes)+2):
                    collection_lists[attribute_counter].append(current_event.get(event_attributes[attribute_counter-2]))
                for attribute_counter in range (len(event_attributes)+2,len(event_attributes)+len(event_num_att)+2):
                    collection_lists[attribute_counter].append(current_event.get(event_num_att[attribute_counter-len(event_attributes)-2]))
        
            # for each prefix create a graph by iterating over all possible prefix lengthes
            for prefix_length in range (2, case_length):
                # prefix_event_classes is a list of tuples representing class = (act, life) of the relevant event.
                prefix_event_classes = collection_lists[0][:prefix_length]
                prefix_classes = list(set(prefix_event_classes)) # only includes unique classes
                prefix_times = collection_lists[1][:prefix_length]
            
                # create target based on the normalization option for user
                if target_normalization:
                    target_cycle = np.array((case_end - collection_lists[1][prefix_length-1]).total_seconds()/3600/24/max_time_norm)
                else:    
                    target_cycle = np.array((case_end - collection_lists[1][prefix_length-1]).total_seconds()/3600/24)
                y = torch.from_numpy(target_cycle).float()
                
                # collect information about nodes
                # define zero array to collect node features of the graph associated to this prefix  
                node_feature = np.zeros((len(prefix_classes), 1), dtype=np.int64)
                # collect node type by iteration over all nodes in the graph.
                for prefix_class in prefix_classes:
                    # get index of the relevant prefix class, and update its row in node feature matirx         
                    node_feature[prefix_classes.index(prefix_class)] = node_class_dict[prefix_class]
                x = torch.from_numpy(node_feature).long()
            
                # Compute edge index list.
                # Each item in pair_result: tuple of tuples representing df between two activity classes 
                pair_result = list(zip(prefix_event_classes , prefix_event_classes [1:]))
                pair_freq = {}            
                for item in pair_result:
                    source_index = prefix_classes.index(item[0])
                    target_index = prefix_classes.index(item[1])
                    if ((source_index, target_index) in pair_freq):
                        pair_freq[(source_index, target_index)] += 1
                    else:
                        pair_freq[(source_index, target_index)] = 1
                edges_list = list(pair_freq.keys())
                edge_index = torch.tensor(edges_list, dtype=torch.long)
            
                # Compute edge attributes
                edge_feature = np.zeros((len(edge_index), edge_dim), dtype=np.float64) # initialize edge feature matrix
                edge_counter = 0
                for edge in edge_index:
                    source_indices = [i for i, x in enumerate(prefix_event_classes) if x == prefix_classes[edge[0]]]
                    target_indices = [i for i, x in enumerate(prefix_event_classes) if x == prefix_classes[edge[1]]]
                    acceptable_indices = [(x, y) for x in source_indices for y in target_indices if x + 1 == y]
                    special_feat = np.empty((0,)) # collect all special features
                    # Add edge weights to the special feature vector
                    num_occ = len(acceptable_indices)/max_case_df
                    special_feat = np.append(special_feat, np.array(num_occ))
                    # Add temporal features to the special feature vector
                    sum_dur = 0
                    for acceptable_index in acceptable_indices:
                        last_dur = (prefix_times[acceptable_index[1]]- prefix_times[acceptable_index[0]]).total_seconds()/3600/24/max_time_norm
                        sum_dur += last_dur
                    special_feat = np.append(special_feat, np.array(last_dur))
                    special_feat = np.append(special_feat, np.array(sum_dur))
                    if acceptable_indices[-1][1] == prefix_length-1: # only meaningful for the latest event in prefix
                        temp_feat1 = (prefix_times[acceptable_indices[-1][1]]-case_start).total_seconds()/3600/24/max_time_norm
                        temp_feat2 = prefix_times[acceptable_indices[-1][1]].hour/24 + prefix_times[acceptable_indices[-1][1]].minute/60/24 + prefix_times[acceptable_indices[-1][1]].second/3600/24
                        temp_feat3 = (prefix_times[acceptable_indices[-1][1]].weekday() + temp_feat2)/7
                    else:
                        temp_feat1 = temp_feat2 = temp_feat3 = 0
                    special_feat = np.append(special_feat, np.array(temp_feat1))
                    special_feat = np.append(special_feat, np.array(temp_feat2))
                    special_feat = np.append(special_feat, np.array(temp_feat3))
                    num_cases = ActiveCase(sorted_start_dates, sorted_end_dates, prefix_times[acceptable_indices[-1][1]])/max_active_cases
                    special_feat = np.append(special_feat, np.array(num_cases))
                    partial_edge_feature = np.append(special_feat, case_level_feat)
                    
                    # One-hot encoding for the target of last occurence + numerical event attributes
                    for attribute_counter in range (2,len(event_attributes)+2):
                        attribute_value = np.array(collection_lists[attribute_counter][acceptable_indices[-1][1]]).reshape(-1, 1)
                        if str(attribute_value[0][0]) == 'nan':
                            num_zeros = len(attribute_encoder_list[attribute_counter - 2].categories_[0])
                            onehot_att = np.zeros((len(attribute_value), num_zeros))
                        else:
                            onehot_att = attribute_encoder_list[attribute_counter-2].transform(attribute_value).toarray()
                        partial_edge_feature = np.append(partial_edge_feature, onehot_att)
                    
                    # Numerical event attributes
                    # imputation requires improvement: average rather than using zero values!
                    for attribute_counter in range (len(event_attributes)+2,len(event_attributes)+len(event_num_att)+2):
                        attribute_value = np.array(collection_lists[attribute_counter][acceptable_indices[-1][1]])
                        if np.isnan(attribute_value):
                            norm_att_val = (event_avg_num_list[attribute_counter-len(event_attributes)-2] - event_min_num_list[attribute_counter-len(event_attributes)-2])/(event_max_num_list[attribute_counter-len(event_attributes)-2]- event_min_num_list[attribute_counter-len(event_attributes)-2])
                            #norm_att_val = np.array(0)
                        else:
                            norm_att_val = (attribute_value - event_min_num_list[attribute_counter-len(event_attributes)-2])/(event_max_num_list[attribute_counter-len(event_attributes)-2]- event_min_num_list[attribute_counter-len(event_attributes)-2])
                        partial_edge_feature = np.append(partial_edge_feature, norm_att_val)
                    edge_feature[edge_counter, :] = partial_edge_feature
                    edge_counter += 1
                edge_attr = torch.from_numpy(edge_feature).float()
                graph = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y, cid=case_id, pl = prefix_length)
                #print(graph)
                data_list.append(graph)
                idx += 1
    return removed_cases, idx, data_list

def main(directory, yml_file, overwrite):
    try:
        yml_file_path = os.path.join(directory, yml_file)
        user_inputs = read_user_inputs(yml_file_path)
        
        # Check whether conversion is required or not.
        dataset_name = user_inputs.get('dataset_name')
        dataset_name_no_ext = os.path.splitext(dataset_name)[0] #dataset name: without .xes extension
        graph_dataset_class_name = eventlog_class_provider(dataset_name_no_ext)
        output_address_list = ['train.pickle', 'val.pickle', 'test.pickle']
        parent_directory = os.path.dirname(os.getcwd()) 
        datasets_directory =  os.path.join(parent_directory, "datasets") #path to dataset folder
        if not os.path.exists(datasets_directory):
            os.makedirs(datasets_directory)
        graph_dataset_path =  os.path.join(datasets_directory, graph_dataset_class_name)
        graph_dataset_path_raw =  os.path.join(graph_dataset_path, "raw")
        graph_dataset_path_processed =  os.path.join(graph_dataset_path, "processed")        
        out_add0 = os.path.join(graph_dataset_path_raw, output_address_list[0])
        out_add1 = os.path.join(graph_dataset_path_raw, output_address_list[1])
        out_add2 = os.path.join(graph_dataset_path_raw, output_address_list[2])        
        if not overwrite and os.path.exists(out_add0) and os.path.exists(out_add1) and os.path.exists(out_add2):
            print(f"For event log: '{dataset_name_no_ext}' conversion is already done and overwrite is set to false.")
            print("Stopping the code.")
            return
        
        # Import the event log
        root_directory = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(root_directory, 'raw_dataset', dataset_name)
        log = xes_importer.apply(dataset_path)   
        event_log = pm4py.read_xes(dataset_path)
        # Assigning values from .yml file to global variables in main
        split_ratio = user_inputs.get('train_val_test_ratio')
        event_attributes = user_inputs.get('event_attributes', [])
        event_num_att = user_inputs.get('event_num_att', [])
        case_attributes = user_inputs.get('case_attributes', [])
        case_num_att = user_inputs.get('case_num_att', [])
        target_normalization = user_inputs.get('target_normalization', True) 
        # add "case:" string to case attributes in .yml file
        case_attributes_full, case_num_ful = case_full_func(case_attributes,case_num_att)               
        # Split the event log into training, validation, and test sets
        train_df, val_df, test_df, train_val_df, train_log, val_log,test_log,\
            train_val_log, max_time_norm, sorted_start_dates,\
                sorted_end_dates = train_test_split_func(log, event_log, split_ratio)
        # Get global information required for transformation (only use training and validation sets)
        # We use all observed values for categorical attributes (one-hot encoding).
        node_class_dict, max_case_df, max_active_cases, min_num_list, max_num_list, event_min_num_list,\
            event_max_num_list,attribute_encoder_list,case_encoder_list, node_dim, edge_dim,\
                avg_num_list, event_avg_num_list = get_global_stat_func(train_val_log, train_val_df,
                                                                        case_num_ful, event_num_att,
                                                                        event_attributes, 
                                                                        case_attributes_full,
                                                                        event_log, log)
 
        # Now the main part for converting prefixes into directed attributed graphs
        removed_cases = [] # a list to collect removed cases (any case with length less than 3)
        idx = 0 # index for graphs
        data_list = [] # a list to collect all Pytorch geometric data objects.
        removed_cases, idx, data_list = graph_conversion_func (train_log, removed_cases, idx, data_list, 
                                                               case_attributes, case_encoder_list, 
                                                               case_num_att, min_num_list, max_num_list, 
                                                               event_attributes, event_num_att, 
                                                               target_normalization, max_time_norm, 
                                                               node_class_dict, edge_dim, max_case_df,
                                                               sorted_start_dates, sorted_end_dates,
                                                               max_active_cases, attribute_encoder_list,
                                                               event_min_num_list, event_max_num_list,
                                                               avg_num_list, event_avg_num_list)
        last_train_idx = idx
        removed_cases, idx, data_list = graph_conversion_func (val_log, removed_cases, idx, data_list, 
                                                               case_attributes, case_encoder_list, 
                                                               case_num_att, min_num_list, max_num_list, 
                                                               event_attributes, event_num_att, 
                                                               target_normalization, max_time_norm, 
                                                               node_class_dict, edge_dim, max_case_df,
                                                               sorted_start_dates, sorted_end_dates,
                                                               max_active_cases, attribute_encoder_list,
                                                               event_min_num_list, event_max_num_list,
                                                               avg_num_list, event_avg_num_list)
        last_val_idx = idx
        removed_cases, idx, data_list = graph_conversion_func (test_log, removed_cases, idx, data_list, 
                                                               case_attributes, case_encoder_list, 
                                                               case_num_att, min_num_list, max_num_list, 
                                                               event_attributes, event_num_att, 
                                                               target_normalization, max_time_norm, 
                                                               node_class_dict, edge_dim, max_case_df,
                                                               sorted_start_dates, sorted_end_dates,
                                                               max_active_cases, attribute_encoder_list,
                                                               event_min_num_list, event_max_num_list,
                                                               avg_num_list, event_avg_num_list)
        
        indices = list(range(len(data_list)))
        # data split based on the global graph list
        train_indices = indices[:last_train_idx]
        val_indices = indices[last_train_idx:last_val_idx]
        test_indices = indices[last_val_idx:] 
        data_train = [data_list[i] for i in train_indices]
        data_val = [data_list[i] for i in val_indices]
        data_test = [data_list[i] for i in test_indices]
        # shuffle the data in each split, to avoid order affect training process
        random.shuffle(data_train)
        random.shuffle(data_val)
        random.shuffle(data_test)
        # Save the training, validation, and test datasets
        file_save_list = [data_train, data_val, data_test]      
        if not os.path.exists(graph_dataset_path):
            os.makedirs(graph_dataset_path)
        if not os.path.exists(graph_dataset_path_raw):
            os.makedirs(graph_dataset_path_raw)
        if not os.path.exists(graph_dataset_path_processed):
            os.makedirs(graph_dataset_path_processed)           
        
        for address_counter in range(len(output_address_list)):
            save_address = osp.join(graph_dataset_path_raw, output_address_list[address_counter])
            save_flie = open(save_address, "wb")
            pickle.dump(file_save_list[address_counter], save_flie)
            save_flie.close()      
        
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except yaml.YAMLError as e:
        print("Error while parsing the .yml file.")
        print(e)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Converting event logs to graph datasets.")
    parser.add_argument("directory", type=str, help="Directory where the conversion's configuration file is located")
    parser.add_argument("yml_file", type=str, help="Name of the YAML file")
    parser.add_argument("--overwrite", type=lambda x: x.lower() == 'true', help="Boolean indicating whether to overwrite")
    
    args = parser.parse_args()
    main(args.directory, args.yml_file, args.overwrite)