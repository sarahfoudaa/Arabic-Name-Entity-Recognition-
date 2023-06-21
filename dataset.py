#read dataset
import pandas as pd

def read_dataset(path):
  # Read the file into a list of lines
  with open(path, 'r') as file:
      lines = file.readlines()

  # Split each line into a list of words
  words = [line.strip().split(' ') for line in lines]

  # Create a dataframe from the list of words
  dataset = pd.DataFrame(words,columns=['words','labels'])

  return dataset

