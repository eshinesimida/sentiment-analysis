# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:25:49 2018

@author: qinxiaozhen
"""

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("./trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None, names=cols)
# above line will be different depending on where you saved your data, and your file name
df.head()

df.sentiment.value_counts()