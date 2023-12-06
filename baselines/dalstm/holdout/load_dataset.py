"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import pandas
from datetime import datetime
import time
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot


def buildOHE(index, n):
    L = [0] * n
    L[index] = 1
    return L

def load_dataset(filename, prev_values=None):
    dataframe = pandas.read_csv(filename, header=0)
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
    else:
        values = prev_values

    print("Dataset: ", dataset)
    print("Values: ", values)


    datasetTR = dataset

    # trick empty column siav log
    # datasetTR=datasetTR[:,:8]
    # datasetTS=datasetTS[:,:8]

    # print len(values)
    # print dataset[0]
    def generate_set(dataset):

        data = []
        original_lengths = []  # Achtung! to collect prefix lengths (required for earliness analysis)
        newdataset = []
        temptarget = []
        # analyze first dataset line
        caseID = dataset[0][0]
        event = dataset[0][1]
        starttime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
        lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
        t = time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
        n = 1
        temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S"))))
        a = [(datetime.fromtimestamp(
            time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
        a.append((datetime.fromtimestamp(
            time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
        a.append(timesincemidnight)
        a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
        a.extend(buildOHE(one_hot(dataset[0][1], values[1], split="|")[0], values[1]))

        field = 3
        for i in dataset[0][3:]:
            if not np.issubdtype(dataframe.dtypes[field], np.number):
                # print field
                a.extend(buildOHE(one_hot(str(i), values[field], split="|")[0], values[field]))
            else:
                a.append(i)
            field += 1
        newdataset.append(a)
        for line in dataset[1:, :]:
            # print line
            case = line[0]
            if case == caseID:
                # print "case", case
                # continues the current case

                t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
                temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                a = [(datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
                a.append((datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                lastevtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                a.extend(buildOHE(one_hot(line[1], values[1], split="|")[0], values[1]))

                field = 3
                for i in line[3:]:
                    if not np.issubdtype(dataframe.dtypes[field], np.number):
                        # print "object", field
                        #print("------")
                        #print("OH: ", one_hot(str(i), values[field], split="|"))
                        #print("Values field: ", values[field])
                        #print("Field: ", field)
                        #print("str(i): ", str(i))
                        #print("OH2",  one_hot(str(i), values[field], split="|")[0])
                        #print("------")
                        a.extend(buildOHE(one_hot(str(i), values[field], split="|")[0], values[field]))
                    else:
                        a.append(i)
                    field += 1
                newdataset.append(a)
                n += 1
                finishtime = datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            else:
                caseID = case
                # Achtung! to exclude prefix of length one: the loop range is changed.
                for i in range(2, len(newdataset)):  # +1 not adding last case. target is 0, not interesting. era 1
                    data.append(newdataset[:i])
                    original_lengths.append(i) # Achtung! added to keep track of prefix lengths (earliness analysis)
                    # print newdataset[:i]
                newdataset = []
                starttime = datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
                lastevtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()

                a = [(datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
                a.append((datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                a.extend(buildOHE(one_hot(line[1], values[1], split="|")[0], values[1]))

                field = 3
                for i in line[3:]:
                    if not np.issubdtype(dataframe.dtypes[field], np.number):
                        a.extend(buildOHE(one_hot(str(i), values[field], split="|")[0], values[field]))
                    else:
                        a.append(i)
                    field += 1
                newdataset.append(a)
                for i in range(n):  
                    # Achtung! the following try-except is added for error handling of the original implementation.
                    try:
                        temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
                    except UnboundLocalError:
                        # Set target value to zero if finishtime is not defined
                        # The effect is negligible as only for one dataset, this exception is for one time executed
                        print('one error in loading dataset is observed', i, n)
                        temptarget[-(i + 1)] = 0
                # Achtung! the following condition is added to remove the target attribute for the prefix of length one
                if n > 1:
                    temptarget.pop(0-n)
                temptarget.pop()  # remove last element with zero target
                temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                finishtime = datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                n = 1

        # last case
        # Achtung! to exclude prefix of length 1: the loop range is adjusted.
        for i in range(2, len(newdataset)):  # + 1 not adding last event, target is 0 in that case. era 1
            data.append(newdataset[:i])
            original_lengths.append(i) # Achtung! added to keep track of prefix lengths
            # print newdataset[:i]
        for i in range(n):  # era n. rimosso esempio con singolo evento
            temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
            # print temptarget[-(i + 1)]
        # Achtung! the condition is added to remove the target attribute for the prefix of length one
        if n > 1:
            temptarget.pop(0-n)
        temptarget.pop()  # remove last element with zero target

        # print temptarget
        print("Generated dataset with n_samples:", len(temptarget))
        assert (len(temptarget) == len(data))
        # print temptarget
        return data, temptarget, original_lengths #Achtung! original_lengths is added to output

    return generate_set(datasetTR), values
