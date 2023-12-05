**Implementation Overview:**

This section of the implementation focuses on the conversion of an event log into a graph dataset. . However, we have provided our source code for the sake of transparency and to encourage the use of our proposal with other datasets or even different graph representation of event prefixes.

**Instructions:**

Name of the dataset must be the same one that is assigned to a variable called `dataset_name`  in the relevant configuration (.yml).
7. Run the`GTconvertor.py`  file using the relevant configuration file. For instance: `python GTconvertor.py bpic15m1.yaml`
The resultant graph dataset will be saved in a seperate folder for each event log in a directory called `transformation`.

**Configuration Files:**

Each configuration file defines global variables specific to the dataset, including:
1. `raw_dataset`: name of the raw dataset (i.e., event log).
2. `event_attributes`, `event_num_att`, `case_attributes`, `case_num_att`: Categorical and numerical attribute names at both the event-level and case-level. The implementation provides the opportunity to experiment with different combinations for these variables. Therefore, it is easy to conduct ablation studies or investigate contribution of different attributes to the accuracy of predictions. 
3. `train_val_test_ratio`: Training, validation, and test data ratio.
4. A boolean attribute called `target_normalization`.
By default, we use a 0.64-0.16-0.20 data split ratio. When `target_normalization` is set to True (the default value), the target attribute is normalized based on the duration of the longest case, ensuring values fall within the range of zero to one. This normalization proved to be helpful because the target attribuite often has a highly skewed distribution: see [Target attribute: histogram visualization](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/target%20attribute%20distribution).

**Dataset Structure:**

For each dataset, three separate files are generated for the training, validation, and test sets. These files are formatted as graph dataset objects compatible with PyTorch Geometric library. Loading a graph dataset to train a GPS graph transformer is done using one single zip file including all these three parts. While our evaluation relies on cross-validation data split, we initially create separate graph dataset files for direct use in a holdout approach. Modifying data split approach can be easily done by using a variable called `split_mode` in the relevant  `.yml` file. 

**Additional Outputs:**

Running the GTconvertor.py file produces several additional output files, including:
1. Encoders: One-hot encoders for both case-level and event-level attributes, implemented using scikit-learn.
2. Activity Classes Dictionary: A dictionary that defines activity classes.
3. Filtered Cases: A list of case IDs for cases that do not have at least three events.
4. Histogram Visualization: A PNG file that visualizes the distribution of target attribute values. All produced files of this type can be found in [Target attribute: histogram visualization](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/target%20attribute%20distribution).

**Note:** We provide additional text files describing general statistics for different graph datasets. See: [General statistics for graph datasets](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/statistics).


  
