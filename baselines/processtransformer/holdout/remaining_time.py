import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('INFO')  # Increase the log level to get more information
from sklearn import metrics 

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

parser = argparse.ArgumentParser(description="Process Transformer - Remaining Time Prediction.")

parser.add_argument("--dataset", required=True, type=str, help="dataset name")

parser.add_argument("--model_dir", default="./models", type=str, help="model directory")

parser.add_argument("--result_dir", default="./results", type=str, help="results directory")

parser.add_argument("--task", type=constants.Task, 
    default=constants.Task.REMAINING_TIME,  help="task name")

parser.add_argument("--epochs", default=10, type=int, help="number of total epochs")

parser.add_argument("--batch_size", default=12, type=int, help="batch size")

parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="learning rate")

parser.add_argument("--gpu", default=0, type=int, 
                    help="gpu id")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Achtung! this method is added to adjust performance evaluation: weighted rather than simple average!
def weighted_average(numbers, weights):
    if len(numbers) != len(weights):
        raise ValueError("Number of elements in 'numbers' and 'weights' lists must be the same.")
    
    total_weight = sum(weights)
    weighted_sum = sum(num * weight for num, weight in zip(numbers, weights))
    
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero.")
    
    return weighted_sum / total_weight

if __name__ == "__main__":
    # Create directories to save the results and models
    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = f"{model_path}/remaining_time_ckpt"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/results"

    # Load data
    data_loader = loader.LogsDataLoader(name = args.dataset)
    
    (train_df, test_df, x_word_dict, y_word_dict, max_case_length, 
        vocab_size, num_output) = data_loader.load_data(args.task)
    
    # Prepare training examples for next time prediction task
    (train_token_x, train_time_x, train_token_y, time_scaler, 
     y_scaler) = data_loader.prepare_data_remaining_time(train_df, x_word_dict, max_case_length)
    
    # Create and train a transformer model
    transformer_model = transformer.get_remaining_time_model(
        max_case_length=max_case_length, 
        vocab_size=vocab_size)

    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.LogCosh())

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor="loss", save_best_only=True)

    transformer_model.fit([train_token_x, train_time_x], train_token_y, 
        epochs=args.epochs, batch_size=args.batch_size, 
        verbose=2, callbacks=[model_checkpoint_callback]) #shuffle=True, 

    # Achtung! and here more changes to adjust performance evaluation
    """
    num_in_k is added to keep track of frequency gor each length
    i starts from 1 which is equivalent to prefix of length 2. The prefix length ends at n-1, this 
    is equivalent to having information about all events except the last one in the trace, and 
    trying to predict the remaining time of the trace. We excluded predixes of length 1,n from the
    original implementation of the ProcessTransformer to have a fair comparison.
    """
    # Evaluate over all the prefixes (k) and save the results
    k, maes, mses, rmses, num_in_k = [],[],[],[],[]
    for i in range(1,max_case_length-1):
        test_data_subset = test_df[test_df["k"]==i]
        if len(test_data_subset) > 0:
            test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_remaining_time(
                    test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False) 
            y_pred = transformer_model.predict([test_token_x, test_time_x])
            _test_y = y_scaler.inverse_transform(test_y)
            _y_pred = y_scaler.inverse_transform(y_pred)
            
            k.append(i)
            maes.append(metrics.mean_absolute_error(_test_y, _y_pred))
            mses.append(metrics.mean_squared_error(_test_y, _y_pred))
            rmses.append(np.sqrt(metrics.mean_squared_error(_test_y, _y_pred)))
            num_in_k.append(len(test_data_subset))

    # Compute simple and weighted average, and total number of examples
    wmaes = weighted_average(maes, num_in_k)
    wmses = weighted_average(mses, num_in_k)
    wrmses = weighted_average(rmses, num_in_k)
    amaes = np.mean(maes)
    amses = np.mean(mses)
    armses = np.mean(rmses)
    total_examples = np.sum(np.array(num_in_k))
    
    # Append the result list with averages that are computed in previous lines.
    k.append(i + 1)
    num_in_k.append(total_examples)
    maes.append(amaes)
    mses.append(amses)
    rmses.append(armses)
    k.append(i + 2)
    num_in_k.append(total_examples)
    maes.append(wmaes) 
    mses.append(wmses)     
    rmses.append(wrmses)       
    print('Simplle Average MAE across all prefixes:', amaes)
    print('Weighted Average MAE across all prefixes:', wmaes)        
    print('Simplle Average MSE across all prefixes:', amses)
    print('Weighted Average MSE across all prefixes:', wmses)
    print('Simplle Average RMSE across all prefixes:', armses)
    print('Weighted Average RMSE across all prefixes:', wrmses)
    
    results_df = pd.DataFrame({"k":k, "mean_absolute_error":maes, "mean_squared_error":mses,
                               "root_mean_squared_error":rmses, "num_in_k":num_in_k})
    results_df.to_csv(result_path+"_remaining_time.csv", index=False)
        