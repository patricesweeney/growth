#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:24:15 2023

@author: patricksweeney
"""

#%% Import


def import_data_local():
    import pandas as pd
    file_path = '/Users/patricksweeney/growth/01_Acquisition/06_PQLs/PQLs.xlsx'
    data = pd.read_excel(file_path)
    return data

data = import_data_local()



def find_missing_values(data):
    missing_values = data.isnull().sum()
    print("Features with missing values are...")
    print(missing_values)
    

find_missing_values(data)