"""
This script is based on the following source code:
    https://github.com/duongtoan261196/RemainingCycleTimePrediction
We just adjusted some parts to efficiently use it in our study.
"""

import os
import pickle
import copy
import random
import argparse
import warnings
import time
import datetime
import pm4py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from ax.service.managed_loop import optimize

##############################################################################
# Genral utility methods, and classes
##############################################################################

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def delete_files_with_string(folder_path, substring):
    files = os.listdir(folder_path)    
    for file in files:
        if substring in file:
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

##############################################################################
# Data preprocessing utility methods, and classes
##############################################################################

# A method to extract trace and compute the 4 timed features for each event
def get_features(tab, case_identifier='case:concept:name',
                  activity_identifier = 'concept:name',
                  time_identifier = 'time:timestamp'):    
    lastcase = ''
    firstLine = True
    casestarttime, lasteventtime = None, None
    line, lines = [] , [] 
    lines_t, lines_t2, lines_t3, lines_t4 = [], [], [], []
    times, times2, times3, times4 = [], [], [], []    
    for i in range(len(tab)):
        # get the unix timestamp of the event        
        t = tab[time_identifier].iloc[i].timestamp()      
        if tab[case_identifier].iloc[i] != lastcase: # if its a new case
            casestarttime, lasteventtime = t, t
            lastcase = tab[case_identifier].iloc[i]
            if not firstLine: # add the previous case
                lines.append(line)
                lines_t.append(times)
                lines_t2.append(times2)
                lines_t3.append(times3)
                lines_t4.append(times4)
            line, times, times2, times3, times4  = [], [], [], [], []
        line.append(tab[activity_identifier].iloc[i])
        timesincelastevent = t - lasteventtime
        timesincecasestart = t - casestarttime
        midnight = datetime.datetime.fromtimestamp(t).replace(
            hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = (
            datetime.datetime.fromtimestamp(t)-midnight).total_seconds()
        dayinweek = datetime.datetime.fromtimestamp(t).weekday() 
        times.append(timesincelastevent)
        times2.append(timesincecasestart)
        times3.append(timesincemidnight)
        times4.append(dayinweek)
        lasteventtime = t
        firstLine = False
    # add the last case
    lines.append(line)
    lines_t.append(times)
    lines_t2.append(times2)
    lines_t3.append(times3)
    lines_t4.append(times4)
    return lines, lines_t, lines_t2, lines_t3, lines_t4

# A method to extract prefixes and target attributes
def Extract_prefix(lines, lines_t, lines_t2, lines_t3, lines_t4):
    step = 1
    sentences, prefix_lengths = [], []
    next_ope, next_ope_t, end_ope_t = [], [], []
    sentences_t, sentences_t2, sentences_t3, sentences_t4 = [], [], [], []
    for line, line_t, line_t2, line_t3, line_t4 in zip(lines,
                                                       lines_t,
                                                       lines_t2,
                                                       lines_t3,
                                                       lines_t4):
        # Keep only prefix with length min = 2
        for i in range(2, len(line), step): 
            prefix_lengths.append(i) # prefix length for earliness analysis
            sentences.append(line[0: i])
            sentences_t.append(line_t[0:i])
            sentences_t2.append(line_t2[0:i])
            sentences_t3.append(line_t3[0:i])
            sentences_t4.append(line_t4[0:i])
            next_ope.append(line[i])
            next_ope_t.append(line_t[i])
            end_ope_t.append(line_t2[-1] - line_t2[i-1])
    sentence_list = [sentences, sentences_t, sentences_t2, sentences_t3,
                     sentences_t4]
    ope_list = [next_ope, next_ope_t, end_ope_t]
    return sentence_list, ope_list, prefix_lengths

# A method to for graph dataset creation
def Get_dataset(tab, list_activities, encoder, divisor_list):
    divisor = divisor_list[0]
    divisor2 = divisor_list[1]
    divisor_rt = divisor_list[2]    
    lines, lines_t, lines_t2, lines_t3, lines_t4 = get_features(tab)
    prefixes, outputs, prefix_lengths = Extract_prefix(lines, lines_t,
                                                       lines_t2, lines_t3,
                                                       lines_t4)
    list_features, list_edge_idx, list_edge_weight, list_rt = [], [], [], []
    for i, sentence in enumerate(prefixes[0]):
        list_rt.append(outputs[2][i]/divisor_rt)
        x = torch.zeros(len(sentence), len(list_activities)+5)
        edge_weight, edge_idx = [], []
        sentence_t = prefixes[1][i]
        sentence_t2 = prefixes[2][i]
        sentence_t3 = prefixes[3][i]
        sentence_t4 = prefixes[4][i]
        for j, char in enumerate(sentence):
            x[j, :len(list_activities)] = torch.from_numpy(
                encoder.transform(np.array([[char]])).toarray()[0]) 
            x[j, len(list_activities)] = j+1
            x[j, len(list_activities)+1] = sentence_t[j]/divisor
            x[j, len(list_activities)+2] = sentence_t2[j]/divisor2
            x[j, len(list_activities)+3] = sentence_t3[j]/86400
            x[j, len(list_activities)+4] = sentence_t4[j]/7
            if j < len(sentence)-1:
                if sentence[j] == sentence[j+1]:
                    weight = 0
                elif sentence[j+1] in sentence[:j+1]:
                    weight = -1
                else:
                    weight = 1
                edge_weight.append(weight)
                edge_idx.append([j, j+1])
        list_features.append(x)
        list_edge_idx.append(torch.tensor(edge_idx).t().to(torch.long))
        list_edge_weight.append(torch.tensor(edge_weight).t())
    return [list_features, list_edge_idx, list_edge_weight], torch.tensor(list_rt), prefix_lengths

# method to handle GGNN preprocessing (from event log to graph dataset)
def GGNN_process(datset_path=None, case_identifier='case:concept:name',
                  activity_identifier = 'concept:name',
                  time_identifier = 'time:timestamp',
                  split_ratio = [0.64, 0.16, 0.20]):
    
    # read the dataset  
    log = pm4py.read_xes(datset_path)
    
    # drop unnecessary columns
    columns_to_keep = [case_identifier, activity_identifier, time_identifier] 
    common_columns = list(set(log.columns).intersection(columns_to_keep))
    log = log[common_columns]
    
    # sort event log first by cases, and within each case by timestamps
    log = log.sort_values([case_identifier,
                           time_identifier]).reset_index(drop = True)
    
    # get train, validation, and test dataframes 
    tv_ratio = split_ratio[0]+split_ratio[1]
    tr_ratio = split_ratio[0]/tv_ratio
    first_act_tab = log.groupby(
        case_identifier).first().sort_values(time_identifier).reset_index()
    first_act_tab = first_act_tab[
        ~first_act_tab.duplicated(subset=[case_identifier,
                                          activity_identifier], keep = "first")]
    first_act_tab = first_act_tab.reset_index(drop = True)
    list_train_valid_cases = list(
        first_act_tab[: int(tv_ratio*len(first_act_tab))][case_identifier].unique())
    list_test_cases = list(
        first_act_tab[int(tv_ratio*len(first_act_tab)):][case_identifier].unique())
    list_train_cases = list_train_valid_cases[
        :int(len(list_train_valid_cases)*tr_ratio)]
    list_valid_cases = list_train_valid_cases[
        int(len(list_train_valid_cases)*tr_ratio):]
    log_train = log[
        log[case_identifier].isin(list_train_cases)].reset_index(drop = True)
    log_valid = log[
        log[case_identifier].isin(list_valid_cases)].reset_index(drop = True)
    log_test = log[
        log[case_identifier].isin(list_test_cases)].reset_index(drop = True)
    
    # create one-hot-encoder for activity identifier
    encoder = OneHotEncoder(handle_unknown='ignore')    
    list_activities = list(log[activity_identifier].unique())
    encoder.fit(np.array(list_activities).reshape((len(list_activities), 1)))
      
    # get normalization coefficients
    lines, lines_t, lines_t2, lines_t3, lines_t4 = get_features(log_train)
    #average time between events
    divisor = np.mean([item for sublist in lines_t for item in sublist]) 
    #average time between current and first events
    divisor2 = np.mean([item for sublist in lines_t2 for item in sublist]) 
    _, outputs, _ = Extract_prefix(lines, lines_t, lines_t2, lines_t3,
                                       lines_t4)
    divisor_rt = np.mean(outputs[2])
    log_list = [log_train, log_valid, log_test]
    divisor_list = [divisor, divisor2, divisor_rt] 
    return log_list, divisor_list, list_activities, encoder

##############################################################################
# Graph representation utility methods, and classes
##############################################################################

class EventLogData(Dataset):
    def __init__ (self, input_x, output):
        self.X = input_x[0]
        self.A = input_x[1]
        self.V = input_x[2]
        self.y = output
        self.y = self.y.to(torch.float32)
        self.y = self.y.reshape((len(self.y),1))

    #get the number of rows in the dataset
    def __len__(self):
        return len(self.X)

    #get a row at a particular index in the dataset
    def __getitem__ (self,idx):
        return [[self.X[idx], self.A[idx], self.V[idx]],self.y[idx]]
    
     # get the indices for the train and test rows
    def get_splits(self, n_valid = 0.2):
        train_idx,valid_idx = train_test_split(list(range(len(self.X))),
                                               test_size = n_valid,
                                               shuffle = True)
        train = Subset(self, train_idx)
        valid = Subset(self, valid_idx)
        return train, valid
    
def my_collate(batch):
    data = [item[0] for item in batch]
    Y = [item[1] for item in batch]
    return [data, Y]

##############################################################################
# Backbone gated graph neural network (GGNN): one gor HPO one for predcitions
##############################################################################
class GGNN_model(nn.Module):
    def __init__(self, parameterization):
        super(GGNN_model, self).__init__()        
        self.ggnn_dim = parameterization.get("neurons", 15)
        self.num_layers = parameterization.get("layers", 1) 
        self.droppout_prob = parameterization.get("dropout", 0.2)
        
        self.ggnn = GatedGraphConv(self.ggnn_dim, num_layers=self.num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(p = self.droppout_prob),
            nn.Linear(self.ggnn_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(p = self.droppout_prob),
            nn.Linear(256,1),
        )
    
    # Progresses data across layers    
    def forward(self, x):
        x = [self.ggnn(X, A.to(torch.long), V) for i, (X, A, V ) in enumerate(x)]
        x = torch.stack([global_mean_pool(single_x, batch = None) for single_x in x])
        x = x.squeeze(1)
        out = self.fc(x)
        return out.squeeze(1)

# Creating the model class
class GGNN_model2(nn.Module):
    def __init__(self, ggnn_dim, num_layers, droppout_prob):
        super(GGNN_model2, self).__init__()        
        self.ggnn_dim = ggnn_dim
        self.num_layers = num_layers
        self.droppout_prob = droppout_prob
        
        self.ggnn = GatedGraphConv(self.ggnn_dim, num_layers=self.num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(p = self.droppout_prob),
            nn.Linear(self.ggnn_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(p = self.droppout_prob),
            nn.Linear(256,1),
        )
    
    # Progresses data across layers    
    def forward(self, x):
        x = [self.ggnn(X, A.to(torch.long), V) for i, (X, A, V ) in enumerate(x)]
        x = torch.stack([global_mean_pool(single_x, batch = None) for single_x in x])
        x = x.squeeze(1)
        out = self.fc(x)
        return out.squeeze(1)
    
    
##############################################################################
# Training and evaluation methods
##############################################################################
# training the network for hyper-parameter tuning
def net_train(net, train_loader, valid_loader, parameters, dtype, device,
              early_stop_patience):
    net.to(dtype=dtype, device=device)
    min_delta = 0
    # Define loss and optimizer
    criterion = nn.L1Loss()
    # 0.001 is used if no lr is specified
    optimizer = optim.Adam(net.parameters(), lr=parameters.get("lr", 0.001))     
    num_epochs = 100 
    
    # Train Network
    not_improved_count = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        net.train()
        training_loss = 0
        num_train = 0
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = [[sub_item.to(device=device) for sub_item in item]
                      for item in inputs]
            labels = torch.tensor(labels).to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            output = net(inputs)
            loss = criterion(output.reshape((1,-1)),labels.reshape((1,-1)))
            # back prop
            loss.backward()
            # optimize
            optimizer.step()
            training_loss+= loss.item()
            num_train+=1
        with torch.no_grad():
            net.eval()
            num_valid = 0
            validation_loss = 0
            for i,(inputs,targets) in enumerate(valid_loader):
                inputs = [[sub_item.to(device=device) for sub_item in item]
                          for item in inputs]
                targets = torch.tensor(targets).to(device=device)
                yhat_valid = net(inputs)
                loss_valid = criterion(yhat_valid.reshape(
                    (1,-1)),targets.reshape((1,-1)))
                validation_loss+= loss_valid.item()
                num_valid+= 1
        avg_training_loss = training_loss/num_train
        avg_validation_loss = validation_loss/num_valid        
        print('Epoch: {}, Training MAE : {}, Validation loss : {}'.format(
            epoch,avg_training_loss,avg_validation_loss))
        if (epoch==0): 
            best_loss = avg_validation_loss
            best_model = copy.deepcopy(net)
        else:
            if (best_loss - avg_validation_loss >= min_delta):
                best_model = copy.deepcopy(net)
                best_loss = avg_validation_loss
                not_improved_count = 0
            else:
                not_improved_count += 1
        # Early stopping
        if not_improved_count == early_stop_patience:
            print('Validation performance didn\'t improve for {} epochs. '
                            'Training stops.'.format(early_stop_patience))
            break
    training_time = time.time() - start_time
    print('Training time:', training_time)
    return best_model

# evaluation of the model for hyper-parameter tuning
def model_evaluate(net, data_loader, dtype, device):
    criterion = nn.L1Loss()
    net.eval()
    loss = 0
    total = 0
    with torch.no_grad():
        for i,(inputs,targets) in enumerate(data_loader):
            # move data to proper dtype and device
            inputs = [[sub_item.to(dtype=dtype, device=device)
                       for sub_item in item] for item in inputs]
            targets = torch.tensor(targets).to(device=device)
            outputs = net(inputs)
            loss += criterion(outputs,targets)
            total += 1
    return loss.item() / total

# training the predictive model for remaining time.
def training_method (train_loader=None, valid_loader=None, ggnn_dim=None,
                     num_layers=None, droppout_prob=None, lr_value=None,
                     device=None, processed_data_path=None, seed=None,
                     data_split=None):    
    num_epochs = 100
    early_stop_patience = 20
    min_delta = 0    
    start=datetime.datetime.now()
    model = GGNN_model2(ggnn_dim, num_layers, droppout_prob)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr_value)
    model = model.to(device)
    epochs_plt = []
    mae_plt = []
    valid_loss_plt = []
    not_improved_count = 0
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0
        num_train = 0
        for i, (inputs,targets) in enumerate(train_loader):
            inputs = [[sub_item.to(device=device) for sub_item in item]
                      for item in inputs]
            targets = torch.tensor(targets).to(device=device)
            
            optimizer.zero_grad() # Clearing the gradients
            yhat = model(inputs)
            loss = criterion(yhat.reshape((1,-1)),targets.reshape((1,-1)))
            loss.backward()
            optimizer.step()

            training_loss+= loss.item()
            num_train+=1
        with torch.no_grad():
            model.eval()
            num_valid = 0
            validation_loss = 0
            for i,(inputs,targets) in enumerate(valid_loader):            
                inputs = [[sub_item.to(device=device) for sub_item in item]
                          for item in inputs]
                targets = torch.tensor(targets).to(device=device)
                yhat_valid = model(inputs)           
                loss_valid = criterion(
                    yhat_valid.reshape((1,-1)),targets.reshape((1,-1)))
                validation_loss+= loss_valid.item()
                num_valid+= 1
        avg_training_loss = training_loss/num_train
        avg_validation_loss = validation_loss/num_valid
        print(
            "Epoch: {}, Training MAE : {}, Validation loss : {}".format(
                epoch,avg_training_loss,avg_validation_loss))
        epochs_plt.append(epoch+1)
        mae_plt.append(avg_training_loss)
        valid_loss_plt.append(avg_validation_loss)
        if (epoch==0): 
            best_loss = avg_validation_loss
            torch.save(model.state_dict(),
                       os.path.join(processed_data_path,
                                    '{}_seed_{}_best_model.pt'.format(
                                        data_split,seed)))
        else:
            if (best_loss - avg_validation_loss >= min_delta):
                torch.save(model.state_dict(),
                           os.path.join(processed_data_path,
                                        '{}_seed_{}_best_model.pt'.format(
                                            data_split,seed)))
                best_loss = avg_validation_loss
                not_improved_count = 0
            else:
                not_improved_count += 1
        # Early stopping
        if not_improved_count == early_stop_patience:
            print('Validation performance didn\'t improve for {} epochs. '
                            'Training stops.'.format(early_stop_patience))
            break
    training_time = (datetime.datetime.now()-start).total_seconds()
    report_path = os.path.join(processed_data_path,
                               '{}_seed_{}_report_.txt'.format(
                                   data_split,seed))
    with open(report_path, 'w') as file:
        for item in zip(epochs_plt,mae_plt,valid_loss_plt):
            file.write('{}\n'.format(item))
        file.write('Training time- in seconds: {}\n'.format(training_time))
    return

# Evaluating the predictive model for remaining time.
def inference_method (model=None, test_loader=None, divisor_rt=None, 
                      dtype=None, device=None, processed_data_path=None,
                      data_split=None, seed=None):
    start=datetime.datetime.now()
    all_results = {'GroundTruth': [], 'Prediction': [], 'Prefix_length': [],
                   'Absolute_error': [], 'Absolute_percentage_error': []}
    absolute_error = 0
    absolute_percentage_error = 0
    with torch.no_grad():
        model.eval()
        for i,(inputs,targets) in enumerate(test_loader):
            prefix_len = inputs[0][0].size(0)
            inputs = [[sub_item.to(dtype=dtype, device=device)
                       for sub_item in item] for item in inputs]
            targets = torch.tensor(targets).to(device=device)
            yhat = model(inputs)
            loss_mape = (torch.abs((targets - yhat)/targets)*100).item()
            criterion = nn.L1Loss()
            loss_mae = criterion(yhat,targets).item()
            # scale the loss!
            loss_mae = loss_mae * divisor_rt/86400
            absolute_error += loss_mae
            absolute_percentage_error += loss_mape
            _y_truth = targets.detach().cpu().numpy()
            _y_pred = yhat.detach().cpu().numpy()
            # scale the ground_truth and prediction!
            _y_truth = _y_truth * divisor_rt/86400
            _y_pred = _y_pred * divisor_rt/86400
            all_results['GroundTruth'].extend(_y_truth)
            all_results['Prediction'].extend(_y_pred)
            all_results['Prefix_length'].extend(np.array([prefix_len]))
            all_results['Absolute_error'].extend(np.array([loss_mae]))
            all_results['Absolute_percentage_error'].extend(np.array([loss_mape]))
        num_test_batches = len(test_loader)    
        absolute_error /= num_test_batches    
        absolute_percentage_error /= num_test_batches
    inference_time = (datetime.datetime.now()-start).total_seconds()
    # Only in case size of test batch is one (in mili seconds)
    instance_inference_time = (inference_time/num_test_batches)*1000
    report_path = os.path.join(processed_data_path,
                               '{}_seed_{}_report_.txt'.format(
                                   data_split,seed))
    with open(report_path, 'a') as file:
        file.write('Inference time- in seconds: {}\n'.format(inference_time))
        file.write(
            'Inference time for each instance- in miliseconds: {}\n'.format(
                instance_inference_time))
        file.write(
            'Inference time for each instance- in miliseconds: {}\n'.format(
                instance_inference_time))
        file.write('Test - MAE: {}, '
                   'MAPE: {}\n'.format(absolute_error,
                                       absolute_percentage_error))
    results_df = pd.DataFrame(all_results)
    csv_filename = os.path.join(
        processed_data_path,'{}_seed_{}_inference_result_.csv'.format(
            data_split,seed))
    results_df.to_csv(csv_filename, index=False)
    return 

##############################################################################
# Main function for the whole pipeline
##############################################################################

def main():
    warnings.filterwarnings('ignore')
    n_splits = 5 # number of splits for cross-validation
    dtype = torch.float
    parser = argparse.ArgumentParser(description='GGNN-remaining_time_prediction')
    parser.add_argument('--dataset',
                        help='Raw dataset to predict remaining time for')
    parser.add_argument('--seed', help='Random seed to use')    
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--trial', type=int, default=100, help='trials for HPO')    
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
    raw_data_dir = os.path.join(current_directory, 'raw_datasets')
    dataset_file = dataset_name+'.xes'
    path = os.path.join(raw_data_dir, dataset_file)
    processed_data_path = os.path.join(current_directory, dataset_name)
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    # set number of trials
    num_trial = args.trial
    
    print('Pipline for:', dataset_name, 'seed:', seed)
    ##########################################################################
    # a function for HPO
    ##########################################################################
    def train_evaluate(parameterization):

        # constructing a new training data loader allows us to tune the batch size
        train_loader = DataLoader(EventLogData(X_train,Y_train),
                                  batch_size=parameterization.get('batchsize', 32),
                                  shuffle=True, collate_fn=my_collate)    
        # Get neural net
        untrained_net = GGNN_model(parameterization)
        # train
        trained_net = net_train(net=untrained_net,
                                train_loader=train_loader,
                                valid_loader = valid_loader,
                                parameters=parameterization, dtype=dtype,
                                device=device, early_stop_patience = 10)
        
        # return the accuracy of the model as it was trained in this run
        return model_evaluate(
            net=trained_net,
            data_loader=valid_loader,
            dtype=dtype,
            device=device,)
        
    ##########################################################################
    # Preprocessing process
    ##########################################################################
    # we only do pre-processing for the first seed (no redundant computation)
    if seed == 42:
        # Handle preprocessing step
        log_list, divisor_list, act_list, encoder = GGNN_process(datset_path=path)
        # get the training, validation, test datasets for holdout data aplit
        X_train, Y_train, train_lengths = Get_dataset(log_list[0], act_list, 
                                                      encoder, divisor_list)
        X_valid, Y_valid, valid_lengths = Get_dataset(log_list[1], act_list,
                                                      encoder, divisor_list)
        X_test, Y_test, test_lengths = Get_dataset(log_list[2], act_list,
                                                   encoder, divisor_list)
        train_save_name = 'GGNN_'+ dataset_name + '_train.pkl'
        val_save_name = 'GGNN_'+ dataset_name + '_valid.pkl'
        test_save_name = 'GGNN_'+ dataset_name + '_test.pkl' 
        devisor_save_name = 'GGNN_'+ dataset_name + '_normalization.pkl'
        with open(os.path.join(processed_data_path, train_save_name), 'wb') as f:
            pickle.dump([X_train, Y_train, train_lengths], f)
        with open(os.path.join(processed_data_path, val_save_name), 'wb') as f:
            pickle.dump([X_valid, Y_valid, valid_lengths], f)
        with open(os.path.join(processed_data_path, test_save_name), 'wb') as f:
            pickle.dump([X_test, Y_test, test_lengths], f)
        with open(os.path.join(processed_data_path, devisor_save_name), 'wb') as f:
            pickle.dump(divisor_list, f)
        # get the training, validation, test datasets for CV data aplit
        # Put all prefixes in one dataset
        X_total = [[], [], []] # X is a list of lists
        total_lengths = [] # prefix lengths are stored in a list
        splits={}
        for counter in range (3):
            X_total[counter] = X_train[counter] + X_valid[counter] + X_test[counter]
        Y_total = torch.cat((Y_train, Y_valid, Y_test), dim=0) # Y is save as s tensor
        total_lengths = train_lengths + valid_lengths + test_lengths
        # get indices for train, validation, and test
        n_samples = len(X_total[0])
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
            for counter in range (3):        
                X_train[counter] = [X_total[counter][i] for i in train_ids]        
            Y_train = Y_total[train_ids]
            train_lengths = [total_lengths[i] for i in train_ids]
            for counter in range (3):        
                X_valid[counter] = [X_total[counter][i] for i in val_ids]        
            Y_valid = Y_total[val_ids]
            valid_lengths = [total_lengths[i] for i in val_ids]
            for counter in range (3):        
                X_test[counter] = [X_total[counter][i] for i in test_ids]        
            Y_test = Y_total[test_ids]
            test_lengths = [total_lengths[i] for i in test_ids]
            train_save_name = 'GGNN_'+ dataset_name + '_fold' + str(split_key) + '_train.pkl'
            val_save_name = 'GGNN_'+ dataset_name + '_fold' + str(split_key) + '_valid.pkl'
            test_save_name = 'GGNN_'+ dataset_name + '_fold' + str(split_key) + '_test.pkl' 
            with open(os.path.join(processed_data_path, train_save_name), 'wb') as f:
                pickle.dump([X_train, Y_train, train_lengths], f)
            with open(os.path.join(processed_data_path, val_save_name), 'wb') as f:
                pickle.dump([X_valid, Y_valid, valid_lengths], f)
            with open(os.path.join(processed_data_path, test_save_name), 'wb') as f:
                pickle.dump([X_test, Y_test, test_lengths], f)
        print('Preprocessing is done for both holdout and CV data split.') 
    else:
        print('Preprocessing is already done!')        
    
    ##########################################################################
    # Hyper-parameter optimization
    ##########################################################################
    # Load preprocessed data for holdout data split
    train_load_name = 'GGNN_'+ dataset_name + '_train.pkl'
    val_load_name = 'GGNN_'+ dataset_name + '_valid.pkl'
    test_load_name = 'GGNN_'+ dataset_name + '_test.pkl' 
    devisor_load_name = 'GGNN_'+ dataset_name + '_normalization.pkl'
    with open(os.path.join(processed_data_path, train_load_name), 'rb') as f:
        X_train, Y_train, train_lengths =  pickle.load(f)
    with open(os.path.join(processed_data_path, val_load_name), 'rb') as f:
        X_valid, Y_valid, valid_lengths =  pickle.load(f)
    with open(os.path.join(processed_data_path, test_load_name), 'rb') as f:
        X_test, Y_test, test_lengths =  pickle.load(f)
    with open(os.path.join(processed_data_path, devisor_load_name), 'rb') as f:
        divisor_list = pickle.load(f)
    # define valid and test loaders
    valid_loader = DataLoader(EventLogData(X_valid, Y_valid),
                              batch_size=len(X_valid[0]),
                              shuffle=False, collate_fn=my_collate)
    # In original implementation test batch sizde is always 1
    test_loader = DataLoader(EventLogData(X_test, Y_test), batch_size=1,
                             shuffle=False, collate_fn=my_collate)
    # check on available best parameters
    if os.path.exists(os.path.join(processed_data_path, 'best_arm.pkl')):
        print('HyperParameter Oprtimization is already done!')
        with open(os.path.join(processed_data_path, 'best_arm.pkl'), 'rb') as f:
            best_arm = pickle.load(f)
    else:   
        # main HPO process using AX library
        print('Start HyperParameter Oprtimization.')
        # for larger number of activities a seperate HP space is required!
        # number of neurons in gated graph convolution layer should be larger
        # than the number of activities in the event log
        if not ('BPIC15' in dataset_name):        
            best_parameters, values, experiment, model = optimize(
                parameters=[
                    {'name': 'neurons', 'type': 'choice', 
                     'values': [40, 60, 80, 100], 'value_type': 'int'},
                    {'name': 'layers', 'type': 'choice', 'values': [3, 4, 5],
                     'value_type': 'int'},
                    {'name': 'lr', 'type': 'range', 'bounds': [1e-4, 0.01],
                     'value_type': 'float', 'log_scale': True},
                    {'name': 'dropout', 'type': 'range', 'bounds': [0, 0.5],
                     'value_type': 'float'},
                    {'name': 'batchsize', 'type': 'choice',
                     'values': [16, 32, 64], 'value_type': 'int'}],
                evaluation_function=train_evaluate, objective_name='MAE loss',
                minimize = True, random_seed = 123, total_trials = num_trial)
        else:
            best_parameters, values, experiment, model = optimize(
                parameters=[
                    {'name': 'neurons', 'type': 'choice', 
                     'values': [420, 440], 'value_type': 'int'},
                    {'name': 'layers', 'type': 'choice', 'values': [3, 4, 5],
                     'value_type': 'int'},
                    {'name': 'lr', 'type': 'range', 'bounds': [1e-4, 0.01],
                     'value_type': 'float', 'log_scale': True},
                    {'name': 'dropout', 'type': 'range', 'bounds': [0, 0.5],
                     'value_type': 'float'},
                    {'name': 'batchsize', 'type': 'choice',
                     'values': [16, 32, 64], 'value_type': 'int'}],
                evaluation_function=train_evaluate, objective_name='MAE loss',
                minimize = True, random_seed = 123, total_trials = num_trial)            
        print('Best parameters are:', best_parameters)
        data = experiment.fetch_data()
        df = data.df
        best_arm_name = df.arm_name[df['mean'] == df['mean'].min()].values[0]
        best_arm = experiment.arms_by_name[best_arm_name]
        print('Best arm is:', best_arm)
        # save the best arm
        with open(os.path.join(processed_data_path, 'best_arm.pkl'), 'wb') as f:
            pickle.dump(best_arm, f)
    
    batch_size = best_arm.parameters['batchsize']
    ggnn_dim = best_arm.parameters['neurons']
    num_layers = best_arm.parameters['layers']
    lr_value = best_arm.parameters['lr']
    droppout_prob = best_arm.parameters['dropout']
    
    ##########################################################################
    # training and evaluation for holdout data split
    ##########################################################################
    # we only train the model once for holdout (seed=42)
    if seed == 42:
        train_loader = DataLoader(EventLogData(X_train,Y_train),
                                 batch_size=batch_size, shuffle=True,
                                 collate_fn=my_collate)
        # Training part
        training_method(train_loader=train_loader, valid_loader=valid_loader,
                        ggnn_dim=ggnn_dim, num_layers=num_layers,
                        droppout_prob=droppout_prob, lr_value=lr_value,
                        device=device, processed_data_path=processed_data_path,
                        seed=seed, data_split='holdout')
        # Inference part
        trained_model = GGNN_model2(ggnn_dim, num_layers, droppout_prob)
        trained_model = trained_model.to(device)
        best_mode_path = os.path.join(
            processed_data_path, 'holdout_seed_{}_best_model.pt'.format(seed))
        trained_model.load_state_dict(
            torch.load(best_mode_path, map_location=torch.device(device)))
        inference_method(model=trained_model, test_loader=test_loader,
                         divisor_rt=divisor_list[2], dtype=dtype,
                         device=device, processed_data_path=processed_data_path,
                         data_split='holdout',seed=seed) 
        print('Training and inference is done for Holdout')
    ##########################################################################
    # training and evaluation for CV data split
    ##########################################################################
    for fold in range(n_splits):
        data_split_name = 'cv_' + str(fold)
        # Load relevant preprocessed data
        train_load_name = 'GGNN_'+ dataset_name + '_fold' + str(fold) + '_train.pkl'
        val_load_name = 'GGNN_'+ dataset_name + '_fold' + str(fold) + '_valid.pkl'
        test_load_name = 'GGNN_'+ dataset_name + '_fold' + str(fold) + '_test.pkl' 
        with open(os.path.join(processed_data_path, train_load_name), 'rb') as f:
            X_train, Y_train, train_lengths =  pickle.load(f)
        with open(os.path.join(processed_data_path, val_load_name), 'rb') as f:
            X_valid, Y_valid, valid_lengths =  pickle.load(f)
        with open(os.path.join(processed_data_path, test_load_name), 'rb') as f:
            X_test, Y_test, test_lengths =  pickle.load(f)
        # create loaders
        train_loader = DataLoader(EventLogData(X_train,Y_train),
                                  batch_size=batch_size, shuffle=True,
                                  collate_fn=my_collate)
        valid_loader = DataLoader(EventLogData(X_valid, Y_valid),
                                  batch_size=batch_size,
                                  shuffle=False, collate_fn=my_collate)
        test_loader = DataLoader(EventLogData(X_test, Y_test), batch_size=1,
                                 shuffle=False, collate_fn=my_collate)
        # train the model
        training_method(train_loader=train_loader, valid_loader=valid_loader,
                        ggnn_dim=ggnn_dim, num_layers=num_layers,
                        droppout_prob=droppout_prob, lr_value=lr_value,
                        device=device, processed_data_path=processed_data_path,
                        seed=seed, data_split=data_split_name)
        # now do the inference
        trained_model = GGNN_model2(ggnn_dim, num_layers, droppout_prob)
        trained_model = trained_model.to(device)
        best_mode_path = os.path.join(
            processed_data_path, '{}_seed_{}_best_model.pt'.format(
                data_split_name,seed))
        trained_model.load_state_dict(
            torch.load(best_mode_path, map_location=torch.device(device)))
        inference_method(model=trained_model, test_loader=test_loader,
                         divisor_rt=divisor_list[2], dtype=dtype,
                         device=device, processed_data_path=processed_data_path,
                         data_split=data_split_name, seed=seed)
        print('Training and inference is done for CV- fold: ', fold)

    # delete all preprocessed data, after saving all the results.
    # only after the last seed
    if seed == 79:
        delete_files_with_string(processed_data_path, "GGNN")    
if __name__ == '__main__':
    main()

    
    
    
    


