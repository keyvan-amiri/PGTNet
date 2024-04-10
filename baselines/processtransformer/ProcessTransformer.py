"""
This script is based on the following source code:
    https://github.com/Zaharah/processtransformer
    https://gitlab.citius.usc.es/efren.rama/pmdlcompararator
We just adjusted some parts to efficiently use it in our study.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_value_
import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_import_factory
import os
import argparse
import json
from pathlib import Path
import pickle
import joblib
import random
from datetime import datetime, timedelta
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn import utils
from sklearn import preprocessing 
import tensorflow as tf
from itertools import chain
import warnings


##############################################################################
# Genral utility methods, and classes
##############################################################################
class Timestamp_Formats:
    TIME_FORMAT_PT = '%Y-%m-%d %H:%M:%S%z'
    TIME_FORMAT_PT2 = '%Y-%m-%d %H:%M:%S.%f%z' # Envpermit & all BPIC 2012 logs
    TIME_FORMAT_PT_list = [TIME_FORMAT_PT, TIME_FORMAT_PT2]


class XES_Fields:
    CASE_COLUMN = 'case:concept:name'
    ACTIVITY_COLUMN = 'concept:name'
    TIMESTAMP_COLUMN = 'time:timestamp'
    LIFECYCLE_COLUMN = 'lifecycle:transition'

# Custom function for LogCosh loss as suggested in original implementation
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
    
# Custom function for Mean Absolute Percentage Error (MAPE)
def mape(outputs, targets):
    return torch.mean(torch.abs((targets - outputs) / targets)) * 100 

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def delete_files(folder_path=None, substring=None, extension=None):
    files = os.listdir(folder_path)    
    for file in files:
        if (substring!= None) and (substring in file):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
        if (extension!= None) and (file.endswith(extension)):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

##############################################################################
# Data preprocessing utility methods, and classes
##############################################################################
# A helper function for remaining time prediction preprocessing
def _remaining_time_helper_func(df):
    case_id = XES_Fields.CASE_COLUMN
    event_name = XES_Fields.ACTIVITY_COLUMN
    event_time = XES_Fields.TIMESTAMP_COLUMN
    processed_df = pd.DataFrame(columns = [case_id, 'prefix', 'k',
                                           'time_passed', 'recent_time',
                                           'latest_time', 'next_act',
                                           'remaining_time_days'])
    idx = 0
    unique_cases = df[case_id].unique()
    for _, case in enumerate(unique_cases):
        act = df[df[case_id] == case][event_name].to_list()
        time = df[df[case_id] == case][event_time].str[:19].to_list()
        time_passed = 0
        latest_diff = timedelta()
        recent_diff = timedelta()
        # remove prefixes of length 1,n (fair comparison)
        for i in range(1, len(act)-1):
            prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
            latest_diff = datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S")
            if i > 1:
                recent_diff = datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")- \
                    datetime.strptime(time[i-2], "%Y-%m-%d %H:%M:%S")
            latest_time = np.where(i == 0, 0, latest_diff.days)
            recent_time = np.where(i <=1, 0, recent_diff.days)
            time_passed = time_passed + latest_time
            time_stamp = str(np.where(i == 0, time[0], time[i]))
            ttc = datetime.strptime(time[-1], "%Y-%m-%d %H:%M:%S") - \
                datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
            ttc = str(ttc.days)
            processed_df.at[idx, case_id]  = case
            processed_df.at[idx, 'prefix']  =  prefix
            processed_df.at[idx, 'k'] = i
            processed_df.at[idx, 'time_passed'] = time_passed
            processed_df.at[idx, 'recent_time'] = recent_time
            processed_df.at[idx, 'latest_time'] =  latest_time
            processed_df.at[idx, 'remaining_time_days'] = ttc
            idx = idx + 1
    processed_df_remaining_time = processed_df[[case_id, 'prefix', 'k', 
        'time_passed', 'recent_time', 'latest_time','remaining_time_days']]
    return processed_df_remaining_time         

# A method to tranform XES to CSV and execute some preprocessing steps
def xes_to_csv(file=None, output_folder=None, fill_na=None):
    xes_path = file
    csv_file = Path(file).stem.split('.')[0] + '.csv'
    dataset_name = Path(file).stem.split('.')[0]
    csv_path = os.path.join(output_folder, csv_file)
    log = xes_import_factory.apply(xes_path,
                                   parameters={'timestamp_sort': True})
    equivalent_dataframe = pm4py.convert_to_dataframe(log)
    equivalent_dataframe.to_csv(csv_path)
    pd_log = pd.read_csv(csv_path)
    if fill_na is not None:
        pd_log.fillna(fill_na, inplace=True)
        pd_log.replace("-", fill_na, inplace=True)
        pd_log.replace(np.nan, fill_na)
    # remove cases with wrong timstamp format for 2012, W, C, CW, A
    if 'BPI_2012' in dataset_name:
        counter_list = []
        for counter in range (len(pd_log)):
            for format_str in Timestamp_Formats.TIME_FORMAT_PT_list:
                try:
                    incr_timestamp = datetime.strptime(
                        str(pd_log.iloc[counter][
                            XES_Fields.TIMESTAMP_COLUMN]), format_str)  
                    if format_str == '%Y-%m-%d %H:%M:%S%z':
                        counter_list.append(counter)
                    break
                except ValueError:
                    continue
        pd_log = pd_log.drop(index=counter_list)
    # Use integers always for case identifiers.
    # We need this to make a split that is equal for every dataset
    pd_log[XES_Fields.CASE_COLUMN] = pd.Categorical(
        pd_log[XES_Fields.CASE_COLUMN])
    pd_log[XES_Fields.CASE_COLUMN] = pd_log[XES_Fields.CASE_COLUMN].cat.codes
    
    pd_log[XES_Fields.ACTIVITY_COLUMN] = pd_log[
        XES_Fields.ACTIVITY_COLUMN].str.lower()
    pd_log[XES_Fields.ACTIVITY_COLUMN] = pd_log[
        XES_Fields.ACTIVITY_COLUMN].str.replace(" ", "-")
    pd_log[XES_Fields.TIMESTAMP_COLUMN] = pd_log[
        XES_Fields.TIMESTAMP_COLUMN].str.replace("/", "-")
    # Set timestamp format to handle the event log.
    if ('env_permit' in dataset_name) or ('BPI_2012' in dataset_name):
        timestamp_format=Timestamp_Formats.TIME_FORMAT_PT2 
    else:      
        timestamp_format=Timestamp_Formats.TIME_FORMAT_PT
    pd_log[XES_Fields.TIMESTAMP_COLUMN]= pd.to_datetime(pd_log[
        XES_Fields.TIMESTAMP_COLUMN], format=timestamp_format).map(
            lambda x: x.strftime(timestamp_format)) 
    # write meta data to json file        
    activities = list(pd_log[XES_Fields.ACTIVITY_COLUMN].unique())
    keys = ["[PAD]", "[UNK]"]        
    keys.extend(activities)
    val_keys = range(len(keys))
    coded_activity = dict({"x_word_dict":dict(zip(keys, val_keys))})
    code_activity_normal = dict({"y_word_dict": dict(
        zip(activities,range(len(activities))))})
    coded_activity.update(code_activity_normal)
    coded_json = json.dumps(coded_activity)
    json_file = Path(file).stem.split(".")[0] + "-metadata.json"
    json_path = os.path.join(output_folder, json_file)
    with open(json_path, "w") as metadata_file:
        metadata_file.write(coded_json)
    # Execute initial preprocessing
    pd_log = _remaining_time_helper_func(pd_log)  
    pd_log.to_csv(csv_path, encoding="utf-8")
    return csv_file, csv_path, json_path

# A method to split the cases into training, validation, and test sets
def split_data(file=None, output_directory=None, case_column=None,
               train_val_test_ratio = [0.64, 0.16, 0.2]):
    # split data for cv
    pandas_init = pd.read_csv(file)
    pd.set_option('display.expand_frame_repr', False)
    groups = [pandas_df for _, pandas_df in \
              pandas_init.groupby(case_column, sort=False)]
    train_size = round(len(groups) * train_val_test_ratio[0])
    val_size = round(len(groups) * (train_val_test_ratio[0]+\
                                    train_val_test_ratio[1]))
    train_groups = groups[:train_size]
    val_groups = groups[train_size:val_size]
    test_groups = groups[val_size:]
    # Disable the sorting. Otherwise it would mess with the order of the timestamps
    train = pd.concat(train_groups, sort=False).reset_index(drop=True)
    val = pd.concat(val_groups, sort=False).reset_index(drop=True)
    test = pd.concat(test_groups, sort=False).reset_index(drop=True)
    train_hold_path = os.path.join(output_directory, "train_" + Path(file).stem + ".csv")
    val_hold_path = os.path.join(output_directory, "val_" + Path(file).stem + ".csv")
    test_hold_path = os.path.join(output_directory, "test_" + Path(file).stem + ".csv")
    train.to_csv(train_hold_path, index=False)
    val.to_csv(val_hold_path, index=False)
    test.to_csv(test_hold_path, index=False)
    return file, train_hold_path, val_hold_path, test_hold_path

# method to handle initial steps of preprocessing for ProcessTransformer
def data_handling(xes=None, output_folder=None):
    # create equivalent csv file, metadata in json file
    csv_file, csv_path, json_path = xes_to_csv(file=xes,
                                               output_folder=output_folder)  
    # execute data split
    (csv_path, train_path, val_path,
     test_path)= split_data(file=csv_path, output_directory=output_folder,
                            case_column=XES_Fields.CASE_COLUMN)     


class PTDataLoader:
    def __init__(self, name, output_folder):
        self.dataset_name = name
        self.train_name = 'train_' + name + '.csv'
        self.val_name = 'val_' + name  + '.csv'
        self.test_name = 'test_' + name  + '.csv'
        self.json_name = name  + '-metadata.json'
        self.train_path = os.path.join(output_folder, self.train_name)
        self.val_path = os.path.join(output_folder, self.val_name)
        self.test_path = os.path.join(output_folder, self.test_name)  
        self.json_path = os.path.join(output_folder, self.json_name)        
       
    def load_data(self):
        # Load csv files
        train_df = pd.read_csv(self.train_path)           
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)         
        # open meta-data json file
        with open(self.json_path) as json_file:
            metadata = json.load(json_file)
        # get word dictionaries
        x_word_dict = metadata["x_word_dict"]
        y_word_dict = metadata["y_word_dict"]
        max_case_length = self.get_max_case_length(train_df["prefix"].values)
        vocab_size = len(x_word_dict) 
        total_classes = len(y_word_dict)

        return (train_df, val_df, test_df, x_word_dict, y_word_dict, 
                max_case_length, vocab_size, total_classes)
    
    # in _remaining_time_helper_func, we excluded the last prefix
    # which is essentially the whole trace. Therefore, we need to add one
    # to have the correct number for max_case_length in the training dataset.
    def get_max_case_length(self, train_x):
        train_token_x = list()
        for _x in train_x:
            train_token_x.append(len(_x.split()))
        return max(train_token_x) + 1

    
    def prepare_data_remaining_time(self, df, x_word_dict, max_case_length, 
        time_scaler = None, y_scaler = None, shuffle = True):

        x = df["prefix"].values
        time_x = df[["recent_time",	"latest_time", 
            "time_passed"]].values.astype(np.float32)
        y = df["remaining_time_days"].values.astype(np.float32)

        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(
                time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(
                time_x).astype(np.float32)            

        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(
                y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(
                y.reshape(-1, 1)).astype(np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)
        
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        return token_x, time_x, y, time_scaler, y_scaler

    
# A method for ProcessTransformer preprocessing (pytorch tensors)
def pt_process(dataset_name=None, output_folder=None, n_splits=5): 
    # define file names, and paths    
    X_train_token_path = os.path.join(
        output_folder, "PT_X_train_token_"+dataset_name+".pt")
    X_train_time_path = os.path.join(
        output_folder, "PT_X_train_time_"+dataset_name+".pt")
    X_val_token_path = os.path.join(
        output_folder, "PT_X_val_token_"+dataset_name+".pt")
    X_val_time_path = os.path.join(
        output_folder, "PT_X_val_time_"+dataset_name+".pt")
    X_test_token_path = os.path.join(
        output_folder, "PT_X_test_token_"+dataset_name+".pt")
    X_test_time_path = os.path.join(
        output_folder, "PT_X_test_time_"+dataset_name+".pt")
    y_train_path = os.path.join(
        output_folder, "PT_y_train_"+dataset_name+".pt")
    y_val_path = os.path.join(
        output_folder, "PT_y_val_"+dataset_name+".pt")
    y_test_path = os.path.join(
        output_folder, "PT_y_test_"+dataset_name+".pt") 
    test_length_path = os.path.join(
        output_folder, "PT_test_length_list_"+dataset_name+".pkl") 
    scaler_path = os.path.join(
        output_folder, "PT_y_scaler_"+dataset_name+".pkl")
    vocab_size_path = os.path.join(
        output_folder, "PT_vocab_size_"+dataset_name+".pkl")
    max_len_path = os.path.join(
        output_folder, "PT_max_len_"+dataset_name+".pkl")
    # create a PTDataLoader object
    data_loader = PTDataLoader(name = dataset_name, output_folder=output_folder)
    (train_df, val_df, test_df, x_word_dict, y_word_dict, max_case_length, 
        vocab_size, num_output) = data_loader.load_data()    
    # Prepare training, validation, test examples as numpy array
    (train_token_x, train_time_x, train_y, time_scaler,
     y_scaler) = data_loader.prepare_data_remaining_time(
         train_df, x_word_dict, max_case_length)  
    (val_token_x, val_time_x, val_y,
     _, _) = data_loader.prepare_data_remaining_time(
         val_df, x_word_dict, max_case_length, time_scaler=time_scaler,
         y_scaler=y_scaler)
    (test_token_x, test_time_x, test_y,
     _, _) = data_loader.prepare_data_remaining_time(
         test_df, x_word_dict, max_case_length, time_scaler=time_scaler,
         y_scaler=y_scaler, shuffle = False)
    # get length of prefixes in the test set
    k_in_train_list = train_df['k'].tolist()
    k_in_val_list = val_df['k'].tolist()
    k_in_test_list = test_df['k'].tolist()
    train_lengths = [k + 1 for k in k_in_train_list] 
    valid_lengths = [k + 1 for k in k_in_val_list] 
    test_lengths = [k + 1 for k in k_in_test_list]   
    # convert numpy arrays to tensors
    train_token_x = torch.tensor(train_token_x)
    train_time_x = torch.tensor(train_time_x)
    train_y = torch.tensor(train_y)
    val_token_x = torch.tensor(val_token_x)
    val_time_x = torch.tensor(val_time_x)
    val_y = torch.tensor(val_y)
    test_token_x = torch.tensor(test_token_x)
    test_time_x = torch.tensor(test_time_x)
    test_y = torch.tensor(test_y)
    # save training, validation, test tensors   
    torch.save(train_token_x, X_train_token_path) 
    torch.save(train_time_x, X_train_time_path)
    torch.save(val_token_x, X_val_token_path)
    torch.save(val_time_x, X_val_time_path)
    torch.save(test_token_x, X_test_token_path)
    torch.save(test_time_x, X_test_time_path)
    torch.save(train_y, y_train_path)
    torch.save(val_y, y_val_path)
    torch.save(test_y, y_test_path)
    # save lengths, max length, and vocab size
    with open(test_length_path, 'wb') as file:
        pickle.dump(test_lengths, file)
    with open(vocab_size_path, 'wb') as file:
        pickle.dump(vocab_size, file)
    with open(max_len_path, 'wb') as file:
        pickle.dump(max_case_length, file)
    # save target scaler for inference:
    joblib.dump(y_scaler, scaler_path)
    # Delete csv files as they are not require anymore
    delete_files(folder_path=output_folder, extension='.csv')
    
    # Now, we create train, valid, test splits for cross-validation
    # Put all prefixes in one dataset
    total_token_x = torch.cat((train_token_x, val_token_x, test_token_x), dim=0)
    total_time_x = torch.cat((train_time_x, val_time_x, test_time_x), dim=0)
    total_y = torch.cat((train_y, val_y, test_y), dim=0)
    total_lengths = train_lengths + valid_lengths + test_lengths
    # get indices for train, validation, and test
    n_samples = total_token_x.shape[0]
    splits={}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf_split = kf.split(np.zeros(n_samples))
    for i, (_, ids) in enumerate(kf_split):
        splits[i] = ids.tolist()
    for split_key in range(n_splits):
        test_ids = splits[split_key]
        val_ids = splits[((split_key + 1) % n_splits)]      
        train_ids = []
        for fold in range(n_splits):
            if fold != split_key and fold != (split_key + 1) % n_splits: 
                train_ids.extend(splits[fold]) 
        # now get training, validation, and test prefixes
        train_token_x = total_token_x[train_ids]
        train_time_x = total_time_x[train_ids]
        train_y = total_y[train_ids]
        val_token_x = total_token_x[val_ids]
        val_time_x = total_time_x[val_ids]
        val_y = total_y[val_ids]
        test_token_x = total_token_x[test_ids]
        test_time_x = total_time_x[test_ids]
        test_y = total_y[test_ids]
        test_lengths = [total_lengths[i] for i in test_ids]        
        # define file names, and paths    
        X_train_token_path = os.path.join(
            output_folder,"PT_X_train_token_fold_"+str(split_key)+dataset_name+".pt")
        X_train_time_path = os.path.join(
            output_folder, "PT_X_train_time_fold_"+str(split_key)+dataset_name+".pt")
        X_val_token_path = os.path.join(
            output_folder, "PT_X_val_token_fold_"+str(split_key)+dataset_name+".pt")
        X_val_time_path = os.path.join(
            output_folder, "PT_X_val_time_fold_"+str(split_key)+dataset_name+".pt")
        X_test_token_path = os.path.join(
            output_folder, "PT_X_test_token_fold_"+str(split_key)+dataset_name+".pt")
        X_test_time_path = os.path.join(
            output_folder, "PT_X_test_time_fold_"+str(split_key)+dataset_name+".pt")
        y_train_path = os.path.join(
            output_folder, "PT_y_train_fold_"+str(split_key)+dataset_name+".pt")
        y_val_path = os.path.join(
            output_folder, "PT_y_val_fold_"+str(split_key)+dataset_name+".pt")
        y_test_path = os.path.join(
            output_folder, "PT_y_test_fold_"+str(split_key)+dataset_name+".pt") 
        test_length_path = os.path.join(
            output_folder, "PT_test_length_list_fold_"\
                                        +str(split_key)+dataset_name+".pkl")
        # save training, validation, test tensors   
        torch.save(train_token_x, X_train_token_path) 
        torch.save(train_time_x, X_train_time_path)
        torch.save(val_token_x, X_val_token_path)
        torch.save(val_time_x, X_val_time_path)
        torch.save(test_token_x, X_test_token_path)
        torch.save(test_time_x, X_test_time_path)
        torch.save(train_y, y_train_path)
        torch.save(val_y, y_val_path)
        torch.save(test_y, y_test_path)
        # save lengths
        with open(test_length_path, 'wb') as file:
            pickle.dump(test_lengths, file)  
    print('Preprocessing is done for both holdout and CV data split.')


##############################################################################
# Backbone ProcessTransformer model for remaining time prediction
##############################################################################
class PTModel(nn.Module):
    def __init__(self, max_len=None, vocab_size=None, embed_dim=None,
                 num_heads=None, ff_dim=None, temp_proj_size=None,
                 last_layer_size=None, dropout=True, p_fix=0.1):        
        super(PTModel, self).__init__()
        self.dropout = dropout
        self.max_len = max_len
        self.att = nn.TransformerEncoderLayer(d_model=embed_dim,
                                              nhead=num_heads,
                                              dim_feedforward=ff_dim,
                                              dropout=p_fix,
                                              layer_norm_eps=1e-6, 
                                              batch_first=True)
        self.token_emb = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)
        self.pos_emb = nn.Embedding(num_embeddings=max_len,
                                    embedding_dim=embed_dim)
        self.linear1 = nn.Linear(3, temp_proj_size) # handle temporal features
        self.dropout_layer = nn.Dropout(p=p_fix)
        self.linear2 = nn.Linear(temp_proj_size+embed_dim, last_layer_size)
        self.linear3 = nn.Linear(last_layer_size, 1)
        '''
        ARGUMENTS:
        max_len: maximum length for prefixes in the dataset
        vocab_size: vocabulary size for activity identifiers
        embed_dim: Embedding Size
        num_heads: number of heads in self-attention layers
        ff_dim: Size of the first layer in feed-forward network
        temp_proj_size: size to project temporal features into it
        last_layer_size: size of the prediction head
        dropout: apply dropout if "True", otherwise no dropout
        p_fix: dropout probability
        '''        
        
    def forward(self, x, x_t):
        #maxlen = x.size(1)  # Get the maximum sequence length
        positions = torch.arange(0, self.max_len,
                                 dtype=torch.long, device=x.device)
        positions = self.pos_emb(positions)  # Lookup position embeddings
        x = self.token_emb(x)  # Lookup token embeddings
        x = x + positions
        x = self.att(x) # self-attention (Transformer Block)
        x = torch.mean(x, dim=1) # global average pooling
        x_t = self.linear1(x_t)
        x_t = F.relu(x_t)
        x = torch.cat((x, x_t), dim=1)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.linear2(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout_layer(x)
        yhat = self.linear3(x)
        return yhat


##############################################################################
# Utlility functions for training and inference
##############################################################################
# function to set the optimizer object
def set_optimizer (model, optimizer_type, base_lr, eps, weight_decay):
    if optimizer_type == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':   
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'Adam':   
        optimizer = optim.Adam(model.parameters(), lr=base_lr, eps=eps,
                               weight_decay=weight_decay)         
    return optimizer

# function to handle training the model
def train_model(model=None, train_loader=None, val_loader=None, criterion=None,
                optimizer=None, scheduler=None, device=None, num_epochs=None,
                early_patience=None, min_delta=None, clip_grad_norm=None,
                clip_value=None, processed_data_path=None, data_split=None,
                seed=None):
    print('Now start training for {} data slit.'.format(data_split))
    checkpoint_path = os.path.join(
        processed_data_path,'{}_seed_{}_best_model.pt'.format(data_split, seed))     
    #Training loop
    current_patience = 0
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        # training
        model.train()
        for batch in train_loader:
            # Forward pass
            token_x_batch, time_x_batch, y_batch = batch
            token_x_batch = token_x_batch.to(device).long()
            time_x_batch = time_x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad() # Resets the gradients
            outputs = model(token_x_batch, time_x_batch)
            loss = criterion(outputs, y_batch)
            # Backward pass and optimization
            loss.backward()
            if clip_grad_norm: # if True: clips gradient at specified value
                clip_grad_value_(model.parameters(), clip_value=clip_value)
            optimizer.step()        
        # Validation
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0
            for batch in val_loader:
                token_x_batch, time_x_batch, y_batch = batch
                token_x_batch = token_x_batch.to(device).long()
                time_x_batch = time_x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(token_x_batch, time_x_batch)
                valid_loss = criterion(outputs, y_batch)
                total_valid_loss += valid_loss.item()                    
            average_valid_loss = total_valid_loss / len(val_loader)
        # print the results       
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {average_valid_loss}')
        # save the best model
        if average_valid_loss < best_valid_loss - min_delta:
            best_valid_loss = average_valid_loss
            current_patience = 0
            checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'best_valid_loss': best_valid_loss            
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            current_patience += 1
            # Check for early stopping
            if current_patience >= early_patience:
                print('Early stopping: Val loss has not improved for {} epochs.'.format(early_patience))
                break
        
        # Update learning rate if there is any scheduler
        if scheduler is not None:
           scheduler.step(average_valid_loss)


# function to handle inference with trained model
def test_model(model=None, test_loader=None, test_original_lengths=None,
               y_scaler=None, processed_data_path=None, data_split=None,
               seed=None, device=None):
    start=datetime.now()
    print('Now start inference for {} data slit.'.format(data_split))
    checkpoint_path = os.path.join(
        processed_data_path,'{}_seed_{}_best_model.pt'.format(data_split, seed)) 
    report_path = os.path.join(processed_data_path,
                               '{}_seed_{}_report_.txt'.format(
                                   data_split,seed))    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    all_results = {'GroundTruth': [], 'Prediction': [], 'Prefix_length': [],
                   'Absolute_error': [], 'Absolute_percentage_error': []}
    absolute_error = 0
    absolute_percentage_error = 0
    length_idx = 0 
    model.eval()
    with torch.no_grad():
        for index, test_batch in enumerate(test_loader):
            token_x_batch, time_x_batch, targets = test_batch
            token_x_batch = token_x_batch.to(device).long()
            time_x_batch = time_x_batch.to(device)
            targets = targets.to(device)
            outputs = model(token_x_batch, time_x_batch)
            batch_size = token_x_batch.shape[0]
            # Original implementation ProcessTransformer: normalization            
            _y_truth = y_scaler.inverse_transform(targets.detach().cpu())
            _y_pred = y_scaler.inverse_transform(outputs.detach().cpu())
            # convert them again to tensors
            _y_truth = torch.tensor(_y_truth, device=device)
            _y_pred = torch.tensor(_y_pred, device=device)                       
            #if _y_pred.dim() == 0:
                #print('strange condition: check it out!')
                #_y_pred = _y_pred.unsqueeze(0)
            # Compute batch loss
            absolute_error += F.l1_loss(_y_pred, _y_truth).item()
            absolute_percentage_error += mape(_y_pred, _y_truth).item()
            # Detach predictions and ground truths (np arrays)
            _y_truth = _y_truth.detach().cpu().numpy()
            _y_pred = _y_pred.detach().cpu().numpy()
            mae_batch = np.abs(_y_truth - _y_pred)
            mape_batch = (mae_batch/_y_truth*100)
            # collect inference result in all_result dict.
            all_results['GroundTruth'].extend(_y_truth.tolist())
            all_results['Prediction'].extend(_y_pred.tolist())
            pre_lengths = \
                test_original_lengths[length_idx:length_idx+batch_size]
            length_idx+=batch_size
            prefix_lengths = (np.array(pre_lengths).reshape(-1, 1)).tolist()
            all_results['Prefix_length'].extend(prefix_lengths)
            all_results['Absolute_error'].extend(mae_batch.tolist())
            all_results['Absolute_percentage_error'].extend(mape_batch.tolist())
           
        num_test_batches = len(test_loader)    
        absolute_error /= num_test_batches    
        absolute_percentage_error /= num_test_batches
    print('Test - MAE: {:.3f}, '
                  'MAPE: {:.3f}'.format(
                      round(absolute_error, 3),
                      round(absolute_percentage_error, 3))) 
    inference_time = (datetime.now()-start).total_seconds() 
    instance_inference = inference_time / len (test_original_lengths) * 1000
    with open(report_path, 'a') as file:
        file.write('Inference time- in seconds: {}\n'.format(inference_time))
        file.write(
            'Inference time for each instance- in miliseconds: {}\n'.format(
                instance_inference))
        file.write('Test - MAE: {:.3f}, '
                      'MAPE: {:.3f}'.format(
                          round(absolute_error, 3),
                          round(absolute_percentage_error, 3)))
    
    # get instance-level predictions in a csv file
    all_results_flat = {key: list(chain.from_iterable(value))
                        for key, value in all_results.items()}
    results_df = pd.DataFrame(all_results_flat)
    csv_filename = os.path.join(
        processed_data_path,'{}_seed_{}_inference_result_.csv'.format(
            data_split,seed)) 
    results_df.to_csv(csv_filename, index=False)
    
##############################################################################
# Main function for the whole pipeline
##############################################################################

def main():
    warnings.filterwarnings('ignore')
    # Parse arguments for training and inference
    parser = argparse.ArgumentParser(description='ProcessTransformer Baseline')
    parser.add_argument('--dataset',
                        help='Raw dataset to predict remaining time for')
    parser.add_argument('--seed', help='Random seed to use')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    args = parser.parse_args()
    # set device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    # set random seed
    seed = int(args.seed)
    set_random_seed(seed)
    # set important file names and paths
    dataset_name = args.dataset
    current_directory = os.getcwd()
    #raw_data_dir = os.path.join(current_directory, 'raw_datasets')
    parent_directory = os.path.dirname(current_directory)
    raw_data_dir = os.path.join(parent_directory, 'GGNN', 'raw_datasets')
    dataset_file = dataset_name+'.xes'
    path = os.path.join(raw_data_dir, dataset_file)
    processed_data_path = os.path.join(current_directory, dataset_name)
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    
    # define important hyperparameters
    n_splits = 5
    training_batch_size = 12
    evaluation_batch_size = 12
    embed_dim = 36
    num_heads = 4 # number of heads in self-attention mechanism
    ff_dim = 64 # size of feedforward network in Transfomer block
    temp_size = 32 # size to project time features
    last_size = 128 # size for the last dense layer
    dropout = True # whether to apply dropout
    drop_prob = 0.1 
    max_epochs = 100 #100
    early_stop_patience = 20
    early_stop_min_delta = 0
    clip_grad_norm = False # if True: clips gradient at specified value
    clip_value = 1.0 # value to clip gradient at
    optimizer_type = 'Adam'
    base_lr = 0.001 # base learning rate 
    eps = 1e-7 # epsilon parameter for Adam 
    weight_decay = 0.0  # weight decay for Adam  
    
    print('Pipline for:', dataset_name, 'seed:', seed)
    
    ##########################################################################
    # Preprocessing process
    ##########################################################################
    # we only do pre-processing for the first seed (no redundant computation)
    if seed == 42:
        data_handling(xes=path, output_folder=processed_data_path)
        pt_process(dataset_name=dataset_name, output_folder=processed_data_path)
    else:
        print('Preprocessing is already done!')
    
    ##########################################################################
    # training and evaluation for holdout data split
    ##########################################################################
    if seed == 42:
        start=datetime.now()    
        # Load tensors, and length lists
        X_train_token_path = os.path.join(
            processed_data_path, "PT_X_train_token_"+dataset_name+".pt")
        X_train_time_path = os.path.join(
            processed_data_path, "PT_X_train_time_"+dataset_name+".pt")
        X_val_token_path = os.path.join(
            processed_data_path, "PT_X_val_token_"+dataset_name+".pt")
        X_val_time_path = os.path.join(
            processed_data_path, "PT_X_val_time_"+dataset_name+".pt")
        X_test_token_path = os.path.join(
            processed_data_path, "PT_X_test_token_"+dataset_name+".pt")
        X_test_time_path = os.path.join(
            processed_data_path, "PT_X_test_time_"+dataset_name+".pt")
        y_train_path = os.path.join(
            processed_data_path, "PT_y_train_"+dataset_name+".pt")
        y_val_path = os.path.join(
            processed_data_path, "PT_y_val_"+dataset_name+".pt")
        y_test_path = os.path.join(
            processed_data_path, "PT_y_test_"+dataset_name+".pt") 
        test_length_path = os.path.join(
            processed_data_path, "PT_test_length_list_"+dataset_name+".pkl") 
        scaler_path = os.path.join(
            processed_data_path, "PT_y_scaler_"+dataset_name+".pkl")
        vocab_size_path = os.path.join(
            processed_data_path, "PT_vocab_size_"+dataset_name+".pkl")
        max_len_path = os.path.join(
            processed_data_path, "PT_max_len_"+dataset_name+".pkl")    
        train_token_x = torch.load(X_train_token_path)
        train_time_x = torch.load(X_train_time_path)
        val_token_x = torch.load(X_val_token_path)
        val_time_x = torch.load(X_val_time_path)
        test_token_x = torch.load(X_test_token_path)
        test_time_x = torch.load(X_test_time_path)
        train_y = torch.load(y_train_path)
        val_y = torch.load(y_val_path)
        test_y = torch.load(y_test_path)
        with open(test_length_path, 'rb') as f:
            test_lengths =  pickle.load(f)
        # input_size corresponds to vocab_size
        with open(vocab_size_path, 'rb') as f:
            input_size =  pickle.load(f)
        with open(max_len_path, 'rb') as f:
            max_len =  pickle.load(f)        
        y_scaler = joblib.load(scaler_path)
    
        # define training, validation, test datasets                    
        train_dataset = TensorDataset(train_token_x, train_time_x, train_y)
        val_dataset = TensorDataset(val_token_x, val_time_x, val_y)
        test_dataset = TensorDataset(test_token_x, test_time_x, test_y)
        # define training, validation, test data loaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=training_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=training_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=evaluation_batch_size, shuffle=False) 
    
        # training for holdout data split
        # define loss function
        criterion = LogCoshLoss()
        # define the model
        model = PTModel(max_len=max_len, vocab_size=input_size,
                        embed_dim=embed_dim, num_heads=num_heads,
                        ff_dim=ff_dim, temp_proj_size=temp_size,
                        last_layer_size=last_size, dropout=dropout,
                        p_fix=drop_prob).to(device)
        # define optimizer
        optimizer = set_optimizer(model, optimizer_type,
                                  base_lr, eps, weight_decay)          
        # define scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5) 
        # execute training:       
        train_model(model=model, train_loader=train_loader,
                    val_loader=val_loader, criterion=criterion,
                    optimizer=optimizer, scheduler=scheduler, device=device,
                    num_epochs=max_epochs, early_patience=early_stop_patience,
                    min_delta=early_stop_min_delta, clip_grad_norm=clip_grad_norm,
                    clip_value=clip_value,
                    processed_data_path=processed_data_path,
                    data_split='holdout', seed=seed)     
        training_time = (datetime.now()-start).total_seconds()
        report_path = os.path.join(processed_data_path,
                                   'holdout_seed_{}_report_.txt'.format(seed))
        with open(report_path, 'w') as file:
            file.write('Training time- in seconds: {}\n'.format(training_time)) 
        print('Training for holout data slit is done!')
    
        # now inference for holdout data split
        # execute inference
        test_model(model=model, test_loader=test_loader,
                   test_original_lengths=test_lengths,
                   y_scaler=y_scaler, processed_data_path= processed_data_path,
                   data_split = 'holdout', seed=seed, device=device)
    
    ##########################################################################
    # training and evaluation for cross-validation data split
    ##########################################################################                
    for fold in range(n_splits):
        start=datetime.now()
        data_split_name = 'cv_' + str(fold)
        # Load relevant preprocessed data
        X_train_token_path = os.path.join(
            processed_data_path, "PT_X_train_token_fold_"+str(fold)+dataset_name+".pt")                
        X_train_time_path = os.path.join(
            processed_data_path, "PT_X_train_time_fold_"+str(fold)+dataset_name+".pt")
        X_val_token_path = os.path.join(
            processed_data_path, "PT_X_val_token_fold_"+str(fold)+dataset_name+".pt")
        X_val_time_path = os.path.join(
            processed_data_path, "PT_X_val_time_fold_"+str(fold)+dataset_name+".pt")
        X_test_token_path = os.path.join(
            processed_data_path, "PT_X_test_token_fold_"+str(fold)+dataset_name+".pt")
        X_test_time_path = os.path.join(
            processed_data_path, "PT_X_test_time_fold_"+str(fold)+dataset_name+".pt")
        y_train_path = os.path.join(
            processed_data_path, "PT_y_train_fold_"+str(fold)+dataset_name+".pt")
        y_val_path = os.path.join(
            processed_data_path, "PT_y_val_fold_"+str(fold)+dataset_name+".pt")
        y_test_path = os.path.join(
            processed_data_path, "PT_y_test_fold_"+str(fold)+dataset_name+".pt") 
        test_length_path = os.path.join(
            processed_data_path, "PT_test_length_list_fold_"+str(fold)+dataset_name+".pkl") 
        scaler_path = os.path.join(
            processed_data_path, "PT_y_scaler_"+dataset_name+".pkl")
        vocab_size_path = os.path.join(
            processed_data_path, "PT_vocab_size_"+dataset_name+".pkl")
        max_len_path = os.path.join(
            processed_data_path, "PT_max_len_"+dataset_name+".pkl")  
        train_token_x = torch.load(X_train_token_path)
        train_time_x = torch.load(X_train_time_path)
        val_token_x = torch.load(X_val_token_path)
        val_time_x = torch.load(X_val_time_path)
        test_token_x = torch.load(X_test_token_path)
        test_time_x = torch.load(X_test_time_path)
        train_y = torch.load(y_train_path)
        val_y = torch.load(y_val_path)
        test_y = torch.load(y_test_path)
        with open(test_length_path, 'rb') as f:
            test_lengths =  pickle.load(f)
        # input_size corresponds to vocab_size
        with open(vocab_size_path, 'rb') as f:
            input_size =  pickle.load(f)
        with open(max_len_path, 'rb') as f:
            max_len =  pickle.load(f)        
        y_scaler = joblib.load(scaler_path)
        
        # define training, validation, test datasets                    
        train_dataset = TensorDataset(train_token_x, train_time_x, train_y)
        val_dataset = TensorDataset(val_token_x, val_time_x, val_y)
        test_dataset = TensorDataset(test_token_x, test_time_x, test_y)
        # define training, validation, test data loaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=training_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=training_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=evaluation_batch_size, shuffle=False) 
        # training for cv data split
        # define loss function
        criterion = LogCoshLoss()
        # define the model
        model = PTModel(max_len=max_len, vocab_size=input_size,
                        embed_dim=embed_dim, num_heads=num_heads,
                        ff_dim=ff_dim, temp_proj_size=temp_size,
                        last_layer_size=last_size, dropout=dropout,
                        p_fix=drop_prob).to(device)
        # define optimizer
        optimizer = set_optimizer(model, optimizer_type,
                                  base_lr, eps, weight_decay)          
        # define scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5) 
        # execute training:       
        train_model(model=model, train_loader=train_loader,
                    val_loader=val_loader, criterion=criterion,
                    optimizer=optimizer, scheduler=scheduler, device=device,
                    num_epochs=max_epochs, early_patience=early_stop_patience,
                    min_delta=early_stop_min_delta, clip_grad_norm=clip_grad_norm,
                    clip_value=clip_value,
                    processed_data_path=processed_data_path,
                    data_split=data_split_name, seed=seed)     
        training_time = (datetime.now()-start).total_seconds()
        report_path = os.path.join(
            processed_data_path,
            '{}_seed_{}_report_.txt'.format(data_split_name,seed))
        with open(report_path, 'w') as file:
            file.write('Training time- in seconds: {}\n'.format(training_time)) 
    
        # now inference for cross=validation data split
        # execute inference
        test_model(model=model, test_loader=test_loader,
                   test_original_lengths=test_lengths,
                   y_scaler=y_scaler, processed_data_path= processed_data_path,
                   data_split = data_split_name, seed=seed, device=device)
    
    # delete all preprocessed data, after saving all the results.
    # only after the last seed
    if seed == 79:
        delete_files(folder_path=processed_data_path, substring="PT_")  
    
if __name__ == '__main__':
    main()