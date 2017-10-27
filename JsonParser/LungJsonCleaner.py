# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import pickle

"""
Created on Thu Oct 26 15:40:52 2017

@author: eduardofierro

@purpose: Clean json metada failes
          See 

@project: Capstone Project, cancer 

"""

### CHANGE DIR TO WHERE THE FILE IS LOCATED IN YOUR LOCAL MACHINE. 

# Where json is
jsondir = "/Users/eduardofierro/Google Drive/TercerSemetre/Capstone/DownloadFiles/Lung/"

# FileName
fnmae = "LUNG_metadata.cart.2017-10-06T17-57-09.290314.json"

# Directory to save to final pickle
savedir = "/Users/eduardofierro/Google Drive/TercerSemetre/Capstone/DownloadFiles/Lung/"

with open(jsondir + fnmae) as data_file:    
    json_orig = json.load(data_file)
    
json_data = pd.read_json(jsondir + "LUNG_metadata.cart.2017-10-06T17-57-09.290314.json")    


json_data = json_data[['file_name', 'cases']]


# Not so elegant, but works fine

new_data = json_data.copy()
new_data['gender'] = None
new_data['age_at_diagnosis'] = None
new_data['cigarettes_per_day'] = None

for row in range(0,json_data.shape[0]):
    
    current_cases = json_data.iloc[row].cases
    
    gender = current_cases[0].get('demographic').get('gender')
    age_at_diagnosis = current_cases[0].get('diagnoses')[0].get('age_at_diagnosis')
    cigarettes_per_day = current_cases[0].get('exposures')[0].get('cigarettes_per_day')

    new_data = new_data.set_value(row, 'gender', gender)
    new_data = new_data.set_value(row, 'age_at_diagnosis', age_at_diagnosis)
    new_data = new_data.set_value(row, 'cigarettes_per_day', cigarettes_per_day)

# Create dictionary
new_data.set_index(new_data.file_name, inplace=True)
new_data = new_data[['gender', 'age_at_diagnosis', 'cigarettes_per_day']]
new_data_dict = new_data.to_dict(orient="index")

pickle.dump(new_data_dict, open( savedir + "LungJsonData.p", "wb" ) )
