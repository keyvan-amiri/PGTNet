import os
import urllib.request
import zipfile
import yaml
import pm4py
import pandas as pd
import gzip

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data if data else {}

def save_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def download_and_process_dataset(url_file, directory, downloaded_file):
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if the downloaded file tracking exists
    downloaded_urls = load_yaml(downloaded_file)

    # Read URLs from the file
    with open(url_file, 'r') as file:
        urls = yaml.safe_load(file)

    # Process each URL
    for key, url in urls.items():
        if key != "file_name_mapping":
            # Check if the URL has already been processed
            if key in downloaded_urls:
                print(f"Skipping already processed URL: {url}")
                continue

            # Extract the filename from the URL
            zip_filename = os.path.join(directory, key + ".zip")

            # Download the file
            urllib.request.urlretrieve(url, zip_filename)

            print(f"File downloaded successfully to {zip_filename}")

            # Extract the contents of the zip file
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(directory)

            print("Zip file extracted successfully")

            # Delete the original zip file
            os.remove(zip_filename)
            print("Original zip file deleted")

            # Delete files that do not end with .xes, .xes.gz, or .csv (if key contains "Helpdesk")
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if not (file.endswith(('.xes', '.xes.gz')) or (key.lower().find("helpdesk") != -1 and file.endswith('.csv'))):
                        os.remove(file_path)
                        print(f"File deleted: {file_path}")

        # Add the URL to the downloaded file tracking
        downloaded_urls[key] = url

    # Write the updated downloaded URLs to the tracking file
    save_yaml(downloaded_file, downloaded_urls)


def rename_files(directory, name_mapping):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return
    
    # Iterate over the dictionary items and rename files
    for old_name, new_name in name_mapping.items():
        old_path = os.path.join(os.getcwd(), directory, old_name) # os.getcwd() provides current directory
        new_path = os.path.join(os.getcwd(), directory, new_name) # os.getcwd() provides current directory

        try:
            os.rename(old_path, new_path)
            #print(f"File '{old_name}' renamed to '{new_name}' successfully.")
        except FileNotFoundError:
            #print(f"File '{old_name}' not found. Skipping renaming.")
            pass
            
def find_files_with_string(directory, search_string):
    matching_files = []

    # Iterate over files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if search_string.lower() in file.lower():
                matching_files.append(os.path.join(root, file))

    return matching_files  

def find_files_with_format(directory, format_string):
    files_with_format = []

    # Iterate over files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(format_string):
                file_path = os.path.join(root, file)
                files_with_format.append((file, file_path))

    return files_with_format

def drop_special_extension(file_name, extension_string):
    base_name, extension = os.path.splitext(file_name)
    if extension.lower() == extension_string:
        return base_name
    else:
        return file_name

def decompress_gz_file(gz_file_path, output_file_path):
    with gzip.open(gz_file_path, 'rt', encoding='utf-8') as gz_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(gz_file.read())

# Downloal event logs from data.4tu.nl
url_file = "4TU-links.yaml"  
directory = "raw_dataset"
downloaded_file = "downloaded-datasets.yaml"

download_and_process_dataset(url_file, directory, downloaded_file)

# Change event logs' names to those used in our paper.
# Load the file_name_mapping from the YAML file
file_name_mapping = load_yaml(url_file).get("file_name_mapping", {})
# Rename the files using the loaded mapping
rename_files(directory, file_name_mapping)
# Handel the case of environement event log due to its strange name.
search_string = "Receipt phase of an environmental permit application process"
result = find_files_with_string(directory, search_string)
if result:
    for old_path in result:
        new_path = os.path.join(os.getcwd(), directory, "env_permit.xes.gz") # os.getcwd() provides current directory
        os.rename(old_path, new_path)
        print("Handle rename of envpermit")
        
# Our implemetation supports .xes .xes.gz and .csv file formats. We convert all files to .xes format. 
# Handle files in .xes.gz format
format_string = '.xes.gz'
extension_string = '.gz'
result = find_files_with_format(directory, format_string)
if result:
    for file_name, file_path in result:
        foldername, old_file_name = os.path.split(file_path)
        new_file_name = drop_special_extension(file_name, extension_string)
        new_path = os.path.join(foldername, new_file_name)
        decompress_gz_file(file_path, new_path)        
        os.remove(file_path)
# Handle files in .csv format
format_string = '.csv'
result = find_files_with_format(directory, format_string)
if result:
    for file_name, file_path in result:
        foldername, old_file_name = os.path.split(file_path)
        if file_name == "finale.csv":
            new_file_name = "HelpDesk.xes"
            dataframe = pd.read_csv(file_path, sep=',')
            dataframe = pm4py.format_dataframe(dataframe, case_id='Case ID', 
                                               activity_key='Activity', timestamp_key='Complete Timestamp')
            event_log = pm4py.convert_to_event_log(dataframe)
        else:
            # the following lines shoule be adjusted in case of experiment with other logs
            new_file_name = drop_special_extension(file_name, format_string) + ".xes"
            dataframe = pd.read_csv(file_path, sep=',')
            try:
                dataframe = pm4py.format_dataframe(dataframe, case_id='case:concept:name',
                                                   activity_key='concept:name',
                                                   timestamp_key='time:timestamp')
            except:
                print('Error! Adjust the names of the mandatory attributes for', file_name)
                break
            event_log = pm4py.convert_to_event_log(dataframe)
        new_path = os.path.join(foldername, new_file_name)
        pm4py.write_xes(event_log, new_path)
        os.remove(file_path)

# Generating additional event logs for BPIC12
current_directory = os.getcwd()
bpic12_path = os.path.join(current_directory, directory, "BPI_Challenge_2012.xes")
bpic12_event_log = pm4py.read_xes(bpic12_path)
# get only events having lifecycle:transition = COMPLETE
bpic12c_event_log = pm4py.filter_event_attribute_values(bpic12_event_log,
                                                          "lifecycle:transition", ["COMPLETE"], 
                                                          level="event", retain=True)
# get only events related to applications
bpic12a_event_log = pm4py.filter_event_attribute_values(bpic12_event_log,
                                                          "concept:name", ['A_PREACCEPTED', 
                                                                           'A_REGISTERED', 
                                                                           'A_SUBMITTED', 
                                                                           'A_FINALIZED', 
                                                                           'A_PARTLYSUBMITTED', 
                                                                           'A_CANCELLED', 
                                                                           'A_ACTIVATED', 
                                                                           'A_APPROVED', 
                                                                           'A_DECLINED', 
                                                                           'A_ACCEPTED'], 
                                                          level="event", retain=True)
# get only events related to offers
bpic12o_event_log = pm4py.filter_event_attribute_values(bpic12_event_log,
                                                          "concept:name", ['O_SENT',
                                                                           'O_ACCEPTED', 
                                                                           'O_SENT_BACK', 
                                                                           'O_DECLINED', 
                                                                           'O_CANCELLED', 
                                                                           'O_CREATED', 
                                                                           'O_SELECTED'], 
                                                          level="event", retain=True)
# get only events related to works
bpic12w_event_log = pm4py.filter_event_attribute_values(bpic12_event_log,
                                                          "concept:name", ['W_Nabellen incomplete dossiers',
                                                                           'W_Wijzigen contractgegevens',
                                                                           'W_Valideren aanvraag',
                                                                           'W_Beoordelen fraude',
                                                                           'W_Nabellen offertes',
                                                                           'W_Afhandelen leads',
                                                                           'W_Completeren aanvraag'], 
                                                          level="event", retain=True)
# get only events related to works and having lifecycle:transition = COMPLETE
bpic12cw_event_log = pm4py.filter_event_attribute_values(bpic12w_event_log,
                                                          "lifecycle:transition", ["COMPLETE"], 
                                                          level="event", retain=True)

save_path = os.path.join(current_directory, directory, "BPI_Challenge_2012C.xes")
pm4py.write_xes(bpic12c_event_log, save_path)
save_path = os.path.join(current_directory, directory, "BPI_Challenge_2012A.xes")
pm4py.write_xes(bpic12a_event_log, save_path)
save_path = os.path.join(current_directory, directory, "BPI_Challenge_2012O.xes")
pm4py.write_xes(bpic12o_event_log, save_path)
save_path = os.path.join(current_directory, directory, "BPI_Challenge_2012W.xes")
pm4py.write_xes(bpic12w_event_log, save_path)
save_path = os.path.join(current_directory, directory, "BPI_Challenge_2012CW.xes")
pm4py.write_xes(bpic12cw_event_log, save_path)
print('Additional event logs are generated for the BPIC2012 log.')