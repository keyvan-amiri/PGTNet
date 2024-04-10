This directory presents our implementation of GGNN baseline. 
The original implementation includes a hyper-parameter search using AX library. You also need to have pytorch, and pytorch geometric installed on your machine. In general, we suggest to create a separate conda environment for experimenting with GGNN. In such a separate environment, install the relevant libraries. You can follow instruction for installation provided [here](https://github.com/duongtoan261196/RemainingCycleTimePrediction?tab=readme-ov-file#installation). Once you are done, you can simply run our implementation using the following arguments:
```
python GGNN.py --dataset HelpDesk --seed 42 --device 7 --trial 100
```
In our pipeline, we only implemented preprocessing and holdout data split (training and inference) for seed 42. We also searched for the best hyperparameters for holdout data split just like the original implementation. We used the same best hyper-parameters for cross-fold validation data split. For this data split, we used three different random seeds to have fair comparison. If you want to use this code in other context or settings, do not remember to adjust the code. 
