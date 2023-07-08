'''
#receive train dataset processed (punc removed, sentence amd labels / row)
#return train and val

import pandas as pd
from sklearn.model_selection import train_test_split

def split(pro_train):
  pro_train, pro_val = train_test_split( pro_train, test_size=0.33, random_state=42)
  return pro_train, pro_val 
  

'''