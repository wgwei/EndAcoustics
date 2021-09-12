# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:13:40 2019

@author: weigang.wei
"""

import pandas as pd
import os

def get_10th_LAFmax():
    dirs = os.listdir(os.getcwd())
    locs = []
    LAFmax10th = []
    for d in dirs:
        if "Pos" in d:
            print(d)
            
            data = pd.read_csv(d)
            LAmaxF = data["LAFmax"].sort_values(ascending=False)
            
            locs.append(d)
            LAFmax10th.append(LAmaxF.values[9])
    
    maxlevelsDF = pd.DataFrame({"Locatoin":locs, "10th LAFmax":LAFmax10th})
    print(maxlevelsDF)
    maxlevelsDF.to_csv("LAFmax10_ambient.csv")
    
if __name__=="__main__":
    get_10th_LAFmax()