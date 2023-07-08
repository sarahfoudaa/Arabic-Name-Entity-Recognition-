import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
import datasets
import utils

def read_dataset(path):
  ''' 
  Function that read the file in the path line by line, split each line with a ' ' and put each half ina DataFrame cell
  
  Pararmeters:
    path(string): a string of the path of the dataset

  Returns:
    dataset(DataFrame): a DataFrame of two columns, words and labels
    
  '''
  # Read the file into a list of lines
  with open(path, 'r') as file:
      lines = file.readlines()
  # Split each line into a list of words
  words = [line.strip().split(' ') for line in lines]
  # Create a dataframe from the list of words
  dataset = pd.DataFrame(words,columns=['words','labels'])
  return dataset

#test = read_dataset('/content/drive/MyDrive/RDI/task 1/ANERcorp-CamelLabSplits/ANERcorp-CamelLabSplits/ANERCorp_CamelLab_test.txt')
#train = read_dataset('/content/drive/MyDrive/RDI/task 1/ANERcorp-CamelLabSplits/ANERcorp-CamelLabSplits/ANERCorp_CamelLab_train.txt')

def dictionary(pro_train):
  ''' 
  Function that takes a Dataframe of the dataset split it into 2,train and valitation, then convert then to dictionary dataset and merge them
  
  Pararmeters:
    pro_train(DataFrame): A DataFrame of two columns, words and labels

  Returns:
    my_dataset_dict(DataSet Dictionary): dataset dictionary of two dictionaries, train and validation 
    
  '''
  pro_train, pro_val = train_test_split( pro_train, test_size=0.33, random_state=42)
  train_dataset = Dataset.from_dict(pro_train)
  val_dataset = Dataset.from_dict(pro_val)
  my_dataset_dict = datasets.DatasetDict({"train":train_dataset,'validation':val_dataset})
  return my_dataset_dict

def main():
  pass
  #pro_train = utils.preprocess_train(train)
  #pro_test = utils.preprocess_test(test)

#  pro_train, pro_val = train_test_split( pro_train, test_size=0.33, random_state=42)
 # train_dataset = Dataset.from_dict(pro_train)
  #val_dataset = Dataset.from_dict(pro_val)
  #my_dataset_dict = datasets.DatasetDict({"train":train_dataset,'validation':val_dataset})

  #print(pro_train)

if __name__ == '__main__':
  main()
  


