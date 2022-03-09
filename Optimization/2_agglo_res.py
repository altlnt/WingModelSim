#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:21:14 2022

@author: alex
"""

import pandas as pd
import os
import glob

result = glob.glob( './datasets/*/opti_res/**.csv' )
target_res_files=[i for i in result if "acc" not in i]

def func(filepath):
    
    df=pd.read_csv(filepath)
    
    if "INTERM" in filepath:
        modtype="interm"
    if "FULL" in filepath:
        modtype="full"
    if "rnn" in filepath:
        modtype="rnn"
    
    df["model_type"]=modtype
    
    ds=filepath.split("/")[2]
    df["dataset"]=ds
    return df

final_df=pd.concat(list(map(func,target_res_files))).sort_values(['dataset',"model_type"])
final_df.to_csv(os.path.join(os.getcwd(),"ALL_RES.csv"),index=False)
    
