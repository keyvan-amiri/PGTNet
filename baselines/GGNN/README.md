This directory presents our implementation of GGNN baseline. 
The original implementation includes a hyper-parameter search using AX library. You also need to have pytorch, and pytorch geometric installed in your machine. In general, we suggest to create a separate conda environment for this experiment, and install the relevant libraries. You can simply follow instruction for installation provided [here](https://github.com/duongtoan261196/RemainingCycleTimePrediction?tab=readme-ov-file#installation). Once you are done, you can simply run our implementation using the following arguments:
```
python GGNN.py --dataset HelpDesk --seed 42 --device 7 --trial 100
```
Note that in our implementation, we only applied preprocessing, and holdout data split to seed number 42. But for cross-fold validation we used three different random seeds to have fair comparison. If you want to use this code in other context or settings, remember to adjust the code. 

