**Implementation Overview:**

This section of the implementation focuses on the conversion of an event log into a graph dataset. This data transformation is not mandatory for training the downstream GPS graph transformer network because we already uploaded the resultant graph dataset in this repository. However, we have provided our source code for the sake of transparency and to encourage the use of our proposal with other datasets or even different graph representation of event prefixes.

**Instructions:**

To get started, follow these steps:
1. Clone this repository.
2. Navigate to the `conversion` directory.
3. In `conversion` directory there is another directory called `raw_dataset`. Download the relevant dataset and copy it into `raw_dataset`. All datasets are available on ... Name of the dataset should be changed based on the variable called `dataset_name`  the relevant configuration (.yml) file or name of this variable should be adjusted based on the name of the relevant dataset.
4. Run the`GTconvertor.py`  file using the relevant configuration file. For instance: `python GTconvertor.py bpic15m1.yaml`
The resulting graph dataset will be saved in a seperate folder for each event log in a directory called `transformation`.

**Configuration Files:**

Each configuration file defines global variables specific to the dataset, including:
1. Categorical and numerical attribute names at both the event-level and case-level.
2. Dataset name.
3. Training, validation, and test data ratio.
4. A boolean attribute called `target_normalization`.
By default, we use a 0.64-0.16-0.20 data split ratio. When `target_normalization` is set to True (the default value), the target attribute is normalized based on the duration of the longest case, ensuring values fall within the range of zero to one.

**Dataset Structure:**

For each dataset, three separate files are generated for the training, validation, and test sets. These files are formatted as graph dataset objects compatible with PyTorch Geometric library. While our evaluation relies on cross-validation data splitting, we initially create separate graph dataset files for direct use in a holdout approach. Modifying the data split can be easily accomplished by adjusting a variable in the configuration file. For further details, please refer to [link to be added].

**Additional Outputs:**

Running the GTconvertor.py file produces several additional output files, including:
1. Encoders: One-hot encoders for both case-level and event-level attributes, implemented using scikit-learn.
2. Activity Classes Dictionary: A dictionary that defines activity classes.
3. Filtered Cases: A list of case IDs for cases that do not have at least three events.
4. Statistics: A text file describing general statistics for each graph dataset. All produced files of this type in [General statistics for graph datasets](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/statistics).
5. Histogram Visualization: A PNG file that visualizes the distribution of target attribute values. All produced files of this type in [Target attribute: histogram visualization](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/target%20attribute%20distribution).


  
