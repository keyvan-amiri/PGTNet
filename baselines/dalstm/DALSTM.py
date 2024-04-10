"""
This script is based on the following source code:
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
import yaml
import sys
import argparse
from pathlib import Path
import pickle
import random
from datetime import datetime
import time
import torch.optim as optim
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence
import warnings

##############################################################################
# Genral utility methods, and classes
##############################################################################

class Timestamp_Formats:
    TIME_FORMAT_DALSTM = "%Y-%m-%d %H:%M:%S"
    #TIME_FORMAT_DALSTM2 = '%Y-%m-%d %H:%M:%S.%f%z' 
    TIME_FORMAT_DALSTM2 = '%Y-%m-%d %H:%M:%S%z' # all BPIC 2012 logs
    TIME_FORMAT_DALSTM_list = [TIME_FORMAT_DALSTM, TIME_FORMAT_DALSTM2]
    
class XES_Fields:
    CASE_COLUMN = 'case:concept:name'
    ACTIVITY_COLUMN = 'concept:name'
    TIMESTAMP_COLUMN = 'time:timestamp'
    LIFECYCLE_COLUMN = 'lifecycle:transition'
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            sys.exit(f"Error reading YAML file: {e}")

def buildOHE(index, n):
    L = [0] * n
    L[index] = 1
    return L

def delete_files(folder_path=None, substring=None, extension=None):
    files = os.listdir(folder_path)    
    for file in files:
        if (substring!= None) and (substring in file):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
        if (extension!= None) and (file.endswith(extension)):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            
# Custom function for Mean Absolute Percentage Error (MAPE)
def mape(outputs, targets):
    return torch.mean(torch.abs((targets - outputs) / targets)) * 100 

##############################################################################
# Data preprocessing utility methods, and classes
##############################################################################

def select_columns(file, input_columns, category_columns, timestamp_format,
                   output_columns, categorize=False, fill_na=None,
                   save_category_assignment=None):

    dataset = pd.read_csv(file)
    if fill_na is not None:
        dataset = dataset.fillna(fill_na)
    if input_columns is not None:
        dataset = dataset[input_columns]
    timestamp_column = XES_Fields.TIMESTAMP_COLUMN
    dataset[timestamp_column] = pd.to_datetime(
        dataset[timestamp_column], utc=True)
    dataset[timestamp_column] = dataset[
        timestamp_column].dt.strftime(timestamp_format)
    if categorize:
        for category_column in category_columns:
            if category_column == XES_Fields.ACTIVITY_COLUMN:
                category_list = dataset[
                    category_column].astype("category").cat.categories.tolist()
                category_dict = {c : i for i, c in enumerate(category_list)}
                if save_category_assignment is None:
                    print("Activity assignment: ", category_dict)
                else:
                    file_name = Path(file).name
                    with open(os.path.join(
                            save_category_assignment, file_name), "w") as fw:
                        fw.write(str(category_dict))
            dataset[category_column] = dataset[
                category_column].astype("category").cat.codes
    if output_columns is not None:
        dataset.rename(
            output_columns,
            axis="columns",
            inplace=True)
    dataset.to_csv(file, sep=",", index=False)

def reorder_columns(file, ordered_columns):
    df = pd.read_csv(file)
    df = df.reindex(columns=(ordered_columns + list(
        [a for a in df.columns if a not in ordered_columns])))
    df.to_csv(file, sep=",", index=False)
    
# A method to tranform XES to CSV and execute some preprocessing steps
def xes_to_csv(file, output_folder, perform_lifecycle_trick=True,
                       fill_na=None):
    
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
    if 'BPI_2012' in dataset_name:
        counter_list = []
        for counter in range (len(pd_log)):
            for format_str in Timestamp_Formats.TIME_FORMAT_DALSTM_list:
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
    # lifecycle_trick: ACTIVITY NAME + LIFECYCLE-TRANSITION
    unique_lifecycle = pd_log[XES_Fields.LIFECYCLE_COLUMN].unique()
    if len(unique_lifecycle) > 1 and perform_lifecycle_trick:
        pd_log[XES_Fields.ACTIVITY_COLUMN] = pd_log[
            XES_Fields.ACTIVITY_COLUMN].astype(str) + "+" + pd_log[
                XES_Fields.LIFECYCLE_COLUMN]       
    pd_log.to_csv(csv_path, encoding="utf-8")

    return csv_file, csv_path

    
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


# method to handle initial steps of preprocessing for DALSTM
def data_handling(xes=None, output_folder=None, cfg=None):
    
    # create equivalent csv file
    csv_file, csv_path = xes_to_csv(file=xes, output_folder=output_folder) 
    # Define relevant attributes
    dataset_name = Path(xes).stem.split('.')[0] 
    attributes = cfg[dataset_name]['event_attributes']
    if dataset_name == "Traffic_Fine":
        attributes.remove('dismissal')     
    if (dataset_name == "BPI_2012" or dataset_name == "BPI_2012W" 
        or dataset_name == "BPI_2013_I"):
        attributes.append(XES_Fields.LIFECYCLE_COLUMN)
    # select related columns
    if 'BPI_2012' in dataset_name:
        #selected_timestamp_format =Timestamp_Formats.TIME_FORMAT_DALSTM2
        selected_timestamp_format =Timestamp_Formats.TIME_FORMAT_DALSTM
    else:
        selected_timestamp_format =Timestamp_Formats.TIME_FORMAT_DALSTM
    select_columns(csv_path, input_columns=[XES_Fields.CASE_COLUMN,
                                            XES_Fields.ACTIVITY_COLUMN,
                                            XES_Fields.TIMESTAMP_COLUMN] + attributes,
                   category_columns=None, 
                   timestamp_format=selected_timestamp_format,
                   output_columns=None, categorize=False) 
    # Reorder columns   
    reorder_columns(csv_path, [XES_Fields.CASE_COLUMN,
                               XES_Fields.ACTIVITY_COLUMN,
                               XES_Fields.TIMESTAMP_COLUMN])     
    
    # execute data split
    split_data(file=csv_path, output_directory=output_folder,
               case_column=XES_Fields.CASE_COLUMN)   
                            

# A method for DALSTM preprocessing (output: Pytorch tensors for training)
def dalstm_load_dataset(filename, prev_values=None):
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    dataframe = pd.read_csv(filename, header=0)
    dataframe = dataframe.replace(r's+', 'empty', regex=True)
    dataframe = dataframe.replace("-", "UNK")
    dataframe = dataframe.fillna(0)

    dataset = dataframe.values
    if prev_values is None:
        values = []
        for i in range(dataset.shape[1]):
            try:
                values.append(len(np.unique(dataset[:, i])))  # +1
            except:
                dataset[:, i] = dataset[:, i].astype(str)       
                values.append(len(np.unique(dataset[:, i])))  # +1

        # output is changed to handle prefix lengths
        #print(values)
        return (None, None, None), values 
    else:
        values = prev_values

    #print("Dataset: ", dataset)
    #print("Values: ", values)

    datasetTR = dataset

    def generate_set(dataset):

        data = []
        # To collect prefix lengths (required for earliness analysis)
        original_lengths = []  
        newdataset = []
        temptarget = []
        
        # analyze first dataset line
        caseID = dataset[0][0]
        starttime = datetime.fromtimestamp(
            time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
        lastevtime = datetime.fromtimestamp(
            time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
        t = time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")
        midnight = datetime.fromtimestamp(
            time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = (
            datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
        n = 1
        temptarget.append(
            datetime.fromtimestamp(time.mktime(
                time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S"))))
        a = [(datetime.fromtimestamp(
            time.mktime(time.strptime(
                dataset[0][2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
        a.append((datetime.fromtimestamp(
            time.mktime(time.strptime(
                dataset[0][2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
        a.append(timesincemidnight)
        a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
        a.extend(buildOHE(
            one_hot(dataset[0][1], values[1], split="|")[0], values[1]))

        field = 3
        for i in dataset[0][3:]:
            if not np.issubdtype(dataframe.dtypes[field], np.number):
                a.extend(buildOHE(one_hot(
                    str(i), values[field], split="|")[0], values[field]))
                #print(field, values[field])
            else:
                #print('numerical', field)
                a.append(i)
            field += 1
        newdataset.append(a)
        #line_counter = 1
        for line in dataset[1:, :]:
            #print(line_counter)
            case = line[0]
            if case == caseID:
                # continues the current case
                t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                midnight = datetime.fromtimestamp(time.mktime(t)).replace(
                    hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (datetime.fromtimestamp(
                    time.mktime(t)) - midnight).total_seconds()
                temptarget.append(datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                a = [(datetime.fromtimestamp(
                    time.mktime(time.strptime(
                        line[2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
                a.append((datetime.fromtimestamp(
                    time.mktime(time.strptime(
                        line[2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                lastevtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                a.extend(buildOHE(one_hot(
                    line[1], values[1], filters=[], split="|")[0], values[1]))

                field = 3
                for i in line[3:]:
                    if not np.issubdtype(
                            dataframe.dtypes[field], np.number):
                        a.extend(buildOHE(
                            one_hot(str(i), values[field], filters=[],
                                    split="|")[0], values[field]))
                    else:
                        a.append(i)
                    field += 1
                newdataset.append(a)
                n += 1
                finishtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            else:
                caseID = case
                # Exclude prefix of length one: the loop range is changed.
                # +1 not adding last case. target is 0, not interesting. era 1
                for i in range(2, len(newdataset)): 
                    data.append(newdataset[:i])
                    # Keep track of prefix lengths (earliness analysis)
                    original_lengths.append(i) 
                    # print newdataset[:i]
                newdataset = []
                starttime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
                lastevtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                midnight = datetime.fromtimestamp(
                    time.mktime(t)).replace(
                        hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (
                    datetime.fromtimestamp(
                        time.mktime(t)) - midnight).total_seconds()

                a = [(datetime.fromtimestamp(
                    time.mktime(time.strptime(
                        line[2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
                a.append((datetime.fromtimestamp(
                    time.mktime(time.strptime(
                        line[2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                a.extend(buildOHE(one_hot(line[1], values[1], split="|")[0], values[1]))

                field = 3
                for i in line[3:]:
                    if not np.issubdtype(dataframe.dtypes[field], np.number):
                        a.extend(buildOHE(
                            one_hot(str(i), values[field],
                                    split="|")[0], values[field]))
                    else:
                        a.append(i)
                    field += 1
                newdataset.append(a)
                for i in range(n):  
                    # try-except: error handling of the original implementation.
                    try:
                        temptarget[-(i + 1)] = (
                            finishtime - temptarget[-(i + 1)]).total_seconds()
                    except UnboundLocalError:
                        # Set target value to zero if finishtime is not defined
                        # The effect is negligible as only for one dataset,
                        # this exception is for one time executed
                        print('one error in loading dataset is observed', i, n)
                        temptarget[-(i + 1)] = 0
                # Remove the target attribute for the prefix of length one
                if n > 1:
                    temptarget.pop(0-n)
                temptarget.pop()  # remove last element with zero target
                temptarget.append(
                    datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], "%Y-%m-%d %H:%M:%S"))))
                finishtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                n = 1
            #line_counter += 1
        # last case
        # To exclude prefix of length 1: the loop range is adjusted.
        # + 1 not adding last event, target is 0 in that case. era 1
        for i in range(2, len(newdataset)):  
            data.append(newdataset[:i])
            original_lengths.append(i) # Keep track of prefix lengths
            # print newdataset[:i]
        for i in range(n):  # era n.
            temptarget[-(i + 1)] = (
                finishtime - temptarget[-(i + 1)]).total_seconds()
            # print temptarget[-(i + 1)]
        # Remove the target attribute for the prefix of length one
        if n > 1:
            temptarget.pop(0-n)
        temptarget.pop()  # remove last element with zero target

        # print temptarget
        print("Generated dataset with n_samples:", len(temptarget))
        assert (len(temptarget) == len(data))
        #Achtung! original_lengths is added to output
        return data, temptarget, original_lengths 

    return generate_set(datasetTR), values

# A method for DALSTM preprocessing (prepoessing actions on pytorch tensors)
def dalstm_process(dataset_name=None, output_folder=None, normalization=False,
                   n_splits=5):
    # define important file names and paths
    full_dataset_name = dataset_name + '.csv'
    full_dataset_path = os.path.join(output_folder, full_dataset_name)
    train_dataset_name = 'train_' + dataset_name + '.csv'
    train_dataset_path = os.path.join(output_folder, train_dataset_name)
    val_dataset_name = 'val_' + dataset_name + '.csv'
    val_dataset_path = os.path.join(output_folder, val_dataset_name)
    test_dataset_name = 'test_' + dataset_name + '.csv'
    test_dataset_path = os.path.join(output_folder, test_dataset_name) 
    X_train_path = os.path.join(
        output_folder, "DALSTM_X_train_"+dataset_name+".pt")
    X_val_path = os.path.join(
        output_folder, "DALSTM_X_val_"+dataset_name+".pt")
    X_test_path = os.path.join(
        output_folder, "DALSTM_X_test_"+dataset_name+".pt")
    y_train_path = os.path.join(
        output_folder, "DALSTM_y_train_"+dataset_name+".pt")
    y_val_path = os.path.join(
        output_folder, "DALSTM_y_val_"+dataset_name+".pt")
    y_test_path = os.path.join(
        output_folder, "DALSTM_y_test_"+dataset_name+".pt") 
    test_length_path = os.path.join(
        output_folder, "DALSTM_test_length_list_"+dataset_name+".pkl")    
    scaler_path = os.path.join(
        output_folder, "DALSTM_max_train_val_"+dataset_name+".pkl")
    input_size_path = os.path.join(
        output_folder, "DALSTM_input_size_"+dataset_name+".pkl")
    max_len_path = os.path.join(
        output_folder, "DALSTM_max_len_"+dataset_name+".pkl")    
    
    # call dalstm_load_dataset for the whole dataset
    (_, _, _), values = dalstm_load_dataset(full_dataset_path)
    # call dalstm_load_dataset for training, validation, and test sets
    (X_train, y_train, train_lengths), _ =  dalstm_load_dataset(
        train_dataset_path, values)
    (X_val, y_val, valid_lengths), _ = dalstm_load_dataset(
        val_dataset_path, values)
    (X_test, y_test, test_lengths), _ = dalstm_load_dataset(
        test_dataset_path, values)
        
    # normalize input data
    # compute the normalization values only on training set
    max = [0] * len(X_train[0][0])
    for a1 in X_train:
        for s in a1:
            for i in range(len(s)):
                if s[i] > max[i]:
                    max[i] = s[i]
    # normalization for train, validation, and test sets
    for a1 in X_train:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]
    for a1 in X_val:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]
    for a1 in X_test:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]
    
    # convert the results to numpy arrays
    X_train = np.asarray(X_train, dtype='object')
    X_val = np.asarray(X_val, dtype='object')
    X_test = np.asarray(X_test, dtype='object')
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    # execute padding, and error handling for BPIC13I
    if dataset_name == 'BPI_2013_I':
        X_train = sequence.pad_sequences(X_train, dtype="int16")
        X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1], 
                                        dtype="int16")
        X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1],
                                       dtype="int16")
    else:
        X_train = sequence.pad_sequences(X_train)
        X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1])
        X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1])
    # Convert target attribute to days
    y_train /= (24*3600) 
    y_val /= (24*3600) 
    y_test /= (24*3600) 
    # Target attribute normalization
    if normalization:
        max_y_train = np.max(y_train)
        max_y_val = np.max(y_val)
        max_train_val = np.max([max_y_train, max_y_val])
        #print(max_train_val)
        y_train /= max_train_val
        y_val /= max_train_val
        y_test /= max_train_val
    else:
        max_train_val = None
    # convert numpy arrays to tensors
    # manage disk space for huge event logs
    if (('BPIC15' in dataset_name) or (dataset_name== 'Traffic_Fine') or
        (dataset_name== 'Hospital')):
        X_train = torch.tensor(X_train).type(torch.bfloat16)
        X_val = torch.tensor(X_val).type(torch.bfloat16)
        X_test = torch.tensor(X_test).type(torch.bfloat16)
    else:
        X_train = torch.tensor(X_train).type(torch.float)
        X_val = torch.tensor(X_val).type(torch.float)
        X_test = torch.tensor(X_test).type(torch.float)
    y_train = torch.tensor(y_train).type(torch.float)
    y_val = torch.tensor(y_val).type(torch.float)
    y_test = torch.tensor(y_test).type(torch.float)
    input_size = X_train.size(2)
    max_len = X_train.size(1) 
    # save training, validation, test tensors
    torch.save(X_train, X_train_path)                    
    torch.save(X_val, X_val_path)
    torch.save(X_test, X_test_path)                      
    torch.save(y_train, y_train_path)
    torch.save(y_val, y_val_path)
    torch.save(y_test, y_test_path)
    # save test prefix lengths, normalization constat, max_len, input_size
    with open(test_length_path, 'wb') as file:
        pickle.dump(test_lengths, file)
    with open(scaler_path, 'wb') as file:
        pickle.dump(max_train_val, file)
    with open(input_size_path, 'wb') as file:
        pickle.dump(input_size, file)
    with open(max_len_path, 'wb') as file:
        pickle.dump(max_len, file)
    # Delete csv files as they are not require anymore
    delete_files(folder_path=output_folder, extension='.csv')
    # Now, we create train, valid, test splits for cross-validation
    # Put all prefixes in one dataset
    X_total = torch.cat((X_train, X_val, X_test), dim=0)
    y_total = torch.cat((y_train, y_val, y_test), dim=0)
    total_lengths = train_lengths + valid_lengths + test_lengths
    # get indices for train, validation, and test
    n_samples = X_total.shape[0]
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
        X_train = X_total[train_ids]
        y_train = y_total[train_ids]
        X_val = X_total[val_ids]
        y_val = y_total[val_ids]
        X_test = X_total[test_ids]
        y_test = y_total[test_ids]
        test_lengths = [total_lengths[i] for i in test_ids]          
        # define file names, and paths 
        X_train_path = os.path.join(
            output_folder,"DALSTM_X_train_fold_"+str(split_key)+dataset_name+".pt")
        X_val_path = os.path.join(
            output_folder, "DALSTM_X_val_fold_"+str(split_key)+dataset_name+".pt")
        X_test_path = os.path.join(
            output_folder, "DALSTM_X_test_fold_"+str(split_key)+dataset_name+".pt")
        y_train_path = os.path.join(
            output_folder, "DALSTM_y_train_fold_"+str(split_key)+dataset_name+".pt")
        y_val_path = os.path.join(
            output_folder, "DALSTM_y_val_fold_"+str(split_key)+dataset_name+".pt")
        y_test_path = os.path.join(
            output_folder, "DALSTM_y_test_fold_"+str(split_key)+dataset_name+".pt")        
        test_length_path = os.path.join(
            output_folder, "DALSTM_test_length_list_fold_"\
                                        +str(split_key)+dataset_name+".pkl")
        # save training, validation, test tensors   
        torch.save(X_train, X_train_path) 
        torch.save(X_val, X_val_path)
        torch.save(X_test, X_test_path)
        torch.save(y_train, y_train_path)
        torch.save(y_val, y_val_path)
        torch.save(y_test, y_test_path)
        # save lengths
        with open(test_length_path, 'wb') as file:
            pickle.dump(test_lengths, file)  
    print('Preprocessing is done for both holdout and CV data split.')


##############################################################################
# Backbone Data-aware LSTM model for remaining time prediction
##############################################################################
class DALSTMModel(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 max_len=None, dropout=True, p_fix=0.2):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        max_len: maximum length for prefixes in the dataset
        dropout: apply dropout if "True", otherwise no dropout
        p_fix: dropout probability
        '''
        super(DALSTMModel, self).__init__()
        
        self.n_layers = n_layers 
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(p=p_fix)
        self.batch_norm1 = nn.BatchNorm1d(max_len)
        self.linear1 = nn.Linear(hidden_size, 1) 
        
    def forward(self, x):
        x = x.float() # if tensors are saved in a different format
        x, (hidden_state,cell_state) = self.lstm1(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.batch_norm1(x)
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                x, (hidden_state,cell_state) = self.lstm2(
                    x, (hidden_state,cell_state))
                if self.dropout:
                    x = self.dropout_layer(x)
                x = self.batch_norm1(x)
        yhat = self.linear1(x[:, -1, :]) # only the last one in the sequence 
        return yhat.squeeze(dim=1) 

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
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            optimizer.zero_grad() # Resets the gradients
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                outputs = model(inputs)
                valid_loss = criterion(outputs, targets)
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
               seed=None, device=None, normalization=False):
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
            inputs = test_batch[0].to(device)
            _y_truth = test_batch[1].to(device)
            batch_size = inputs.shape[0]
            _y_pred = model(inputs)
            # convert tragets, outputs in case of normalization
            if normalization:
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred        
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
    
    flattened_list = [item for sublist in all_results['Prefix_length'] 
                      for item in sublist]
    all_results['Prefix_length'] = flattened_list
    results_df = pd.DataFrame(all_results)
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
    parser = argparse.ArgumentParser(description='DALSTM Baseline')
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
    preprocessing_cfg = read_config('preprocessing_config.yaml')  
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    
    # define important hyperparameters
    n_splits = 5
    n_nuerons = 150
    n_layers = 2 # number of LSTM layers
    dropout = True # whether to apply dropout
    drop_prob = 0.2
    max_epochs = 500
    early_stop_patience = 20
    early_stop_min_delta = 0
    clip_grad_norm = False # if True: clips gradient at specified value
    clip_value = 1.0 # value to clip gradient at   
    optimizer_type = 'NAdam'
    base_lr = 0.001 # base learning rate 
    eps = 1e-7 # epsilon parameter for Adam 
    weight_decay = 0.0  # weight decay for Adam 
    normalization = False # whether to normalized target attribute or not
    
    print('Pipline for:', dataset_name, 'seed:', seed)
    
    ##########################################################################
    # Preprocessing process
    ##########################################################################
    if seed == 42:
        data_handling(xes=path, output_folder=processed_data_path,
                      cfg=preprocessing_cfg)
        
        dalstm_process(dataset_name=dataset_name,
                       output_folder=processed_data_path,
                       normalization=normalization)
    else:
        print('Preprocessing is already done!')
    
    ##########################################################################
    # training and evaluation for holdout data split
    ##########################################################################
    if seed == 42:
        start=datetime.now()    
        # Load tensors, and length lists
        X_train_path = os.path.join(
            processed_data_path, "DALSTM_X_train_"+dataset_name+".pt")
        X_val_path = os.path.join(
            processed_data_path, "DALSTM_X_val_"+dataset_name+".pt")
        X_test_path = os.path.join(
            processed_data_path, "DALSTM_X_test_"+dataset_name+".pt")
        y_train_path = os.path.join(
            processed_data_path, "DALSTM_y_train_"+dataset_name+".pt")
        y_val_path = os.path.join(
            processed_data_path, "DALSTM_y_val_"+dataset_name+".pt")
        y_test_path = os.path.join(
            processed_data_path, "DALSTM_y_test_"+dataset_name+".pt") 
        test_length_path = os.path.join(
            processed_data_path, "DALSTM_test_length_list_"+dataset_name+".pkl")    
        scaler_path = os.path.join(
            processed_data_path, "DALSTM_max_train_val_"+dataset_name+".pkl")
        input_size_path = os.path.join(
            processed_data_path, "DALSTM_input_size_"+dataset_name+".pkl")
        max_len_path = os.path.join(
            processed_data_path, "DALSTM_max_len_"+dataset_name+".pkl")         
        X_train = torch.load(X_train_path)
        X_val = torch.load(X_val_path)
        X_test = torch.load(X_test_path)
        y_train = torch.load(y_train_path)
        y_val = torch.load(y_val_path)
        y_test = torch.load(y_test_path)        
        with open(test_length_path, 'rb') as f:
            test_lengths =  pickle.load(f)
        # input_size corresponds to vocab_size
        with open(input_size_path, 'rb') as f:
            input_size =  pickle.load(f)
        with open(max_len_path, 'rb') as f:
            max_len =  pickle.load(f) 
        with open(scaler_path, 'rb') as f:
            max_train_val =  pickle.load(f)  
        # define training, validation, test datasets                    
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        # define training, validation, test data loaders
        train_loader = DataLoader(train_dataset, batch_size=max_len, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=max_len, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=max_len, shuffle=False)
        
        # training for holdout data split
        # define loss function
        criterion = nn.L1Loss()
        # define the model
        model = DALSTMModel(input_size=input_size, hidden_size=n_nuerons,
                                n_layers=n_layers, max_len=max_len,
                                dropout=dropout, p_fix=drop_prob).to(device)
        # define optimizer
        optimizer = set_optimizer(model, optimizer_type, base_lr, eps,
                                  weight_decay)          
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
                   y_scaler=max_train_val,
                   processed_data_path= processed_data_path,
                   data_split = 'holdout', seed=seed, device=device,
                   normalization=normalization)
    
    ##########################################################################
    # training and evaluation for cross-validation data split
    ##########################################################################
    for fold in range(n_splits):
        start=datetime.now()
        data_split_name = 'cv_' + str(fold)
        # Load tensors, and length lists
        X_train_path = os.path.join(
            processed_data_path, "DALSTM_X_train_fold_"+str(fold)+dataset_name+".pt")
        X_val_path = os.path.join(
            processed_data_path, "DALSTM_X_val_fold_"+str(fold)+dataset_name+".pt")
        X_test_path = os.path.join(
            processed_data_path, "DALSTM_X_test_fold_"+str(fold)+dataset_name+".pt")
        y_train_path = os.path.join(
            processed_data_path, "DALSTM_y_train_fold_"+str(fold)+dataset_name+".pt")
        y_val_path = os.path.join(
            processed_data_path, "DALSTM_y_val_fold_"+str(fold)+dataset_name+".pt")
        y_test_path = os.path.join(
            processed_data_path, "DALSTM_y_test_fold_"+str(fold)+dataset_name+".pt") 
        test_length_path = os.path.join(
            processed_data_path, "DALSTM_test_length_list_fold_"+str(fold)+dataset_name+".pkl")    
        scaler_path = os.path.join(
            processed_data_path, "DALSTM_max_train_val_"+dataset_name+".pkl")
        input_size_path = os.path.join(
            processed_data_path, "DALSTM_input_size_"+dataset_name+".pkl")
        max_len_path = os.path.join(
            processed_data_path, "DALSTM_max_len_"+dataset_name+".pkl")  
        X_train = torch.load(X_train_path)
        X_val = torch.load(X_val_path)
        X_test = torch.load(X_test_path)
        y_train = torch.load(y_train_path)
        y_val = torch.load(y_val_path)
        y_test = torch.load(y_test_path)        
        with open(test_length_path, 'rb') as f:
            test_lengths =  pickle.load(f)
        # input_size corresponds to vocab_size
        with open(input_size_path, 'rb') as f:
            input_size =  pickle.load(f)
        with open(max_len_path, 'rb') as f:
            max_len =  pickle.load(f) 
        with open(scaler_path, 'rb') as f:
            max_train_val =  pickle.load(f) 
        # define training, validation, test datasets                    
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        # define training, validation, test data loaders
        train_loader = DataLoader(train_dataset, batch_size=max_len, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=max_len, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=max_len, shuffle=False)
        # training for holdout data split
        # define loss function
        criterion = nn.L1Loss()
        # define the model
        model = DALSTMModel(input_size=input_size, hidden_size=n_nuerons,
                                n_layers=n_layers, max_len=max_len,
                                dropout=dropout, p_fix=drop_prob).to(device)
        # define optimizer
        optimizer = set_optimizer(model, optimizer_type, base_lr, eps,
                                  weight_decay)          
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
                   y_scaler=max_train_val,
                   processed_data_path= processed_data_path,
                   data_split = data_split_name, seed=seed, device=device,
                   normalization=normalization)
        
    # delete all preprocessed data, after saving all the results.
    # only after the last seed
    if seed == 79:
        delete_files(folder_path=processed_data_path, substring="DALSTM_")     
        
if __name__ == '__main__':
    main()