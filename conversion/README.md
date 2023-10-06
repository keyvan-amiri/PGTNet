**Implementation Overview**
The main goal of this part of the implementation is to convert an event log into a graph dataset.
Conversion is not required for training GPS graph transformer networks, as we already uplaoded graph datasets used in our study.
Yet, we provide our source code for transparency and to encourage using our proposal for other datasets.

Instructions:
1. Clone this repository.
2. Navigate to the `conversion` directory.
3. Run `GTconvertor.py` file using the relevant configuration file for instance: `python GTconvertor.py bpic15m1.yaml`
4. Graph dataset which is obtained by running this command will be saved in a seperate folder for each event log in a directory called `transformation`.

Each configuration file defines global variables for the relevant data set. This include name of categorical and numerical attributes in both event-level and case-level. It also includes name of the dataset, training, validation, test ratio and a boolean attribute called `target_normalization`. By default we are using 0.64-0.16-0.20 ratio. If target_normalization is set True (the default value), the target attribute is normalized by duration of the longest case, and it will always get a value between zero and one.

For each dataset: three seperate files will be created for training, validation, and test set. Each of these files are a part from a graph dataset object in Pytorch geometric library. While our evaluation is based on cross-validation data split, we initially create each graph dataset in three seperate files which can be used for holdout approach directly. Using this approach changing the data split can be easily done by defining a variable called `???` in a configuration file. For more information, see: ???.

For each dataset: additional outputs are created by running `GTconvertor.py` file. These additional files include the following:
  A) Encoders (one-hot encoders in scikitlearn library) that are used for case_level and event_level attribute.
  B) A dictionary which defines activity classes.
  C) list of cases that are filtered (case IDs for cases that do not have at least three events)
  D) A text file describing general statistics for each graph dataset. You can find all produced files of this type in ???
  E) A PNG file visualizing histogram for distribution of the target attribute values. You can find all produced files of this type in ???
  
