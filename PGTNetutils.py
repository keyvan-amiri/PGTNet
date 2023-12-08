# Provide the class name for graph dataset (see: PGTNetlogHandler.py)
def eventlog_class_provider(name_of_dataset):
    if name_of_dataset == "BPIC15_1":
        pyg_class_name = "EVENTBPIC15M1"
    elif name_of_dataset == "BPIC15_2":
        pyg_class_name = "EVENTBPIC15M2"
    elif name_of_dataset == "BPIC15_3":
        pyg_class_name = "EVENTBPIC15M3"
    elif name_of_dataset == "BPIC15_4":
        pyg_class_name = "EVENTBPIC15M4"
    elif name_of_dataset == "BPIC15_5":
        pyg_class_name = "EVENTBPIC15M5"
    elif name_of_dataset == "BPI_Challenge_2012":
        pyg_class_name = "EVENTBPIC12"
    elif name_of_dataset == "BPI_Challenge_2012A":
        pyg_class_name = "EVENTBPIC12A"
    elif name_of_dataset == "BPI_Challenge_2012O":
        pyg_class_name = "EVENTBPIC12O"
    elif name_of_dataset == "BPI_Challenge_2012W":
        pyg_class_name = "EVENTBPIC12W"
    elif name_of_dataset == "BPI_Challenge_2012C":
        pyg_class_name = "EVENTBPIC12C"
    elif name_of_dataset == "BPI_Challenge_2012CW":
        pyg_class_name = "EVENTBPIC12CW"
    elif name_of_dataset == "BPI_Challenge_2013C" or name_of_dataset == "2013C":
        pyg_class_name = "EVENTBPIC13C"
    elif name_of_dataset == "BPI_Challenge_2013I" or name_of_dataset == "2013I":
        pyg_class_name = "EVENTBPIC13I"
    elif name_of_dataset == "BPIC20_DomesticDeclarations" or name_of_dataset == "2020D":
        pyg_class_name = "EVENTBPIC20D"
    elif name_of_dataset == "BPIC20_InternationalDeclarations" or name_of_dataset == "2020I":
        pyg_class_name = "EVENTBPIC20I"
    elif name_of_dataset == "env_permit" or name_of_dataset.lower() == "envpermit":
        pyg_class_name = "EVENTEnvPermit"
    elif name_of_dataset == "HelpDesk" or name_of_dataset.lower() == "helpdesk":
        pyg_class_name = "EVENTHelpDesk"
    elif name_of_dataset == "Hospital" or name_of_dataset.lower() == "hospital":
        pyg_class_name = "EVENTHospital"
    elif name_of_dataset == "Sepsis" or name_of_dataset.lower() == "sepsis":
        pyg_class_name = "EVENTSepsis"
    elif name_of_dataset == "Traffic_Fines" or name_of_dataset.lower() == "trafficfines":
        pyg_class_name = "EVENTTrafficfines" 
    else:
        pyg_class_name = None
        print('Error! no Pytorch Geometric dataset class is defined for this event log') 
    return pyg_class_name


def mean_cycle_norm_factor_provider(dataset):
    
    if dataset.lower() == 'helpdesk' or ("EVENTHelpDesk" in dataset):
        mean_cycle = 40.90
        normalization_factor = 59.99496528
    elif dataset == "BPIC20_InternationalDeclarations" or dataset == "2020I" or ("EVENTBPIC20I" in dataset):
        mean_cycle = 86.50
        normalization_factor = 742
    elif dataset == "BPIC20_DomesticDeclarations" or dataset == "2020D" or ("EVENTBPIC20D" in dataset):
        mean_cycle = 11.50
        normalization_factor = 469.2363
    elif dataset.lower() == 'envpermit' or dataset == "env_permit" or ("EVENTEnvPermit" in dataset):
        mean_cycle = 5.41
        normalization_factor = 275.8396
    elif dataset == "BPI_Challenge_2013I" or dataset == '2013I' or ("EVENTBPIC13I" in dataset):
        mean_cycle = 12.08
        normalization_factor = 771.351770833333
    elif dataset == "BPI_Challenge_2013C" or dataset == '2013C' or ("EVENTBPIC13C" in dataset):
        mean_cycle = 178.88
        normalization_factor = 2254.84850694444 
    elif dataset == "BPI_Challenge_2012" or dataset == '2012' or ("EVENTBPIC12" in dataset):
        mean_cycle = 8.60
        normalization_factor = 137.22148162037
    elif dataset == "BPI_Challenge_2012C" or dataset == '2012C' or ("EVENTBPIC12C" in dataset):
        mean_cycle = 8.61
        normalization_factor = 91.4552796412037
    elif dataset == "BPI_Challenge_2012W" or dataset == '2012W' or ("EVENTBPIC12W" in dataset):
        mean_cycle = 11.70
        normalization_factor = 137.220982743055
    elif dataset == "BPI_Challenge_2012CW" or dataset == '2012CW' or ("EVENTBPIC12CW" in dataset):
        mean_cycle = 11.40
        normalization_factor = 91.040850324074
    elif dataset == "BPI_Challenge_2012O" or dataset == '2012O' or ("EVENTBPIC12O" in dataset):
        mean_cycle = 17.18
        normalization_factor = 89.5486824537037
    elif dataset == "BPI_Challenge_2012A" or dataset == '2012A' or ("EVENTBPIC12A" in dataset):
        mean_cycle = 8.08
        normalization_factor = 91.4552796412037
    elif dataset == "BPIC15_1" or dataset == '2015m1' or ("EVENTBPIC15M1" in dataset):
        mean_cycle = 95.90
        normalization_factor = 1486
    elif dataset == "BPIC15_2" or dataset == '2015m2' or ("EVENTBPIC15M2" in dataset):
        mean_cycle = 160.30
        normalization_factor = 1325.9583
    elif dataset == "BPIC15_3" or dataset == '2015m3' or ("EVENTBPIC15M3" in dataset):
        mean_cycle = 62.20
        normalization_factor = 1512
    elif dataset == "BPIC15_4" or dataset == '2015m4' or ("EVENTBPIC15M4" in dataset):
        mean_cycle = 116.90
        normalization_factor = 926.9583
    elif dataset == "BPIC15_5" or dataset == '2015m5' or ("EVENTBPIC15M5" in dataset):
        mean_cycle = 98
        normalization_factor = 1343.9583
    elif dataset.lower() == 'sepsis' or ("EVENTSepsis" in dataset):
        mean_cycle = 28.48
        normalization_factor = 422.323946759259
    elif dataset.lower() == 'trafficfines' or dataset == "Traffic_Fines" or  ("EVENTTrafficfines" in dataset):
        mean_cycle = 341.60
        normalization_factor = 4372
    elif dataset.lower() == 'hospital' or ("EVENTHospital" in dataset): 
        mean_cycle = 127.24
        normalization_factor = 1035.4212037037
        
    else:
        print('Dataset is not recognized')
        mean_cycle = None
        normalization_factor = None
        
    return normalization_factor, mean_cycle

def eventlog_name_provider(name_of_class):
    if name_of_class == "EVENTBPIC15M1":
        event_log_name = "BPIC15_1"
    elif name_of_class == "EVENTBPIC15M2":
        event_log_name = "BPIC15_2"
    elif name_of_class == "EVENTBPIC15M3":
        event_log_name = "BPIC15_3"
    elif name_of_class == "EVENTBPIC15M4":
        event_log_name = "BPIC15_4"
    elif name_of_class == "EVENTBPIC15M5":
        event_log_name = "BPIC15_5"
    elif name_of_class == "EVENTBPIC12":
        event_log_name = "BPI_Challenge_2012"
    elif name_of_class == "EVENTBPIC12A":
        event_log_name = "BPI_Challenge_2012A"
    elif name_of_class == "EVENTBPIC12O":
        event_log_name = "BPI_Challenge_2012O"
    elif name_of_class == "EVENTBPIC12W":
        event_log_name = "BPI_Challenge_2012W"
    elif name_of_class == "EVENTBPIC12C":
        event_log_name = "BPI_Challenge_2012C"
    elif name_of_class == "EVENTBPIC12CW":
        event_log_name = "BPI_Challenge_2012CW"
    elif name_of_class == "EVENTBPIC13C":
        event_log_name = "BPI_Challenge_2013C"
    elif name_of_class == "EVENTBPIC13I":
        event_log_name = "BPI_Challenge_2013I"
    elif name_of_class == "EVENTBPIC20D":
        event_log_name = "BPIC20_DomesticDeclarations"
    elif name_of_class == "EVENTBPIC20I":
        event_log_name = "BPIC20_InternationalDeclarations" 
    elif name_of_class == "EVENTEnvPermit":
        event_log_name = "env_permit"
    elif name_of_class == "EVENTHelpDesk":
        event_log_name = "HelpDesk"
    elif name_of_class == "EVENTHospital":
        event_log_name = "Hospital"
    elif name_of_class == "EVENTSepsis":
        event_log_name = "Sepsis"
    elif name_of_class == "EVENTTrafficfines":
        event_log_name = "Traffic_Fines"
    else:
        event_log_name = None
        print('Error! no event log is related to this pythorch geometric dataset class.') 
    return event_log_name