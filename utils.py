#functions that will be called from other files
#preprocesssing 

import pandas as pd
import pytz
from datetime import datetime

def remove_punc(test):
  ''' 
  Function that removes punctuation from dataframe

  Parameters:
    test(DataFrame): dataframe of two columns ,words and labels, unprocessed and have punctuations in it
    
  Returns:
    test(DataFrame):  dataframe of two columns ,words and labels, unprocessed and without punctuations in it
  '''
  punc = '،-!"#$%&\'()*+,.-–/:;<=>?@[\ـ\]^_`{؛|}~«؟'
  punctuation = set(punc)
  # Create boolean mask to identify rows containing punctuation
  mask = test['words'].apply(lambda x: any(char in punctuation for char in x))
  test = test[mask==False].reset_index(drop=True)
  return test

def preprocess_test(test):
  ''' 
  Function to process the test dataset.
  The function removes the punctuation from the datframe, then concat the words into sentence(string) separat then with a ' ' corresponding to it their labels in  a list

  Parameters:
    test (DataFrame): A DataFrame of two columns, words(string) and labels(string), where each words has its label 

  Returns:
    sen_entities (DataFrame):A DataFrame of two columns, sentence(string) and list_entities(list of string) .
   '''
  test = remove_punc(test)
  entities = ['O', 'B-PERS', 'B-LOC', 'I-PERS', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'I-LOC']
  df_sen = [] #list of sentences
  s = '' #sentence
  df = pd.DataFrame(columns = ['labels'])#dataframe of lists of labels
  l = []#list of labels
  for i in range (len(test)):
    if test['words'][i]!='' and test['labels'][i] in entities:
      s = s +" "+ test['words'][i]
      l.append(test['labels'][i])
    else:
      df_sen.append(s[1:])
      s=''
      df.at[i, 'labels'] = l
      l=[]
  df_sen = pd.DataFrame(df_sen, columns=['sentence'])
  df = df.reset_index()
  sen_entities = pd.DataFrame().assign(sentence=df_sen['sentence'],list_entities = df['labels'])
  return sen_entities

#for train
def preprocess_train(test):
  ''' 
  Function to process the train dataset.
  The function removes the punctuation from the datframe, then collect the words into list corresponding to it their labels mapped to their IDs in  a list

  Parameters:
    test (DataFrame): A DataFrame of two columns, words(string) and labels(string), where each words has its label 

  Returns:
    sen_entities (DataFrame):A DataFrame of three columns,, sentence_id(int),sentence (list of strings) and list_entities(list of numbers) .
   '''
  test = remove_punc(test)
  test['labels'] = test['labels'].replace(['B-LOC','B-MISC','B-ORG','B-PERS','I-LOC','I-MISC','I-ORG','I-PERS','O'], [0,1,2,3,4,5,6,7,8])
  no_of_sen =0
  df_sen =  pd.DataFrame(columns=['text'])#dataframe for lists of words
  s = []#list of words
  entities_per_line = pd.DataFrame(columns = ['labels'])#dataframe for lists of laels
  ss = []#list of labels
  for i in range (len(test)):
    if test['words'][i]!='':
      s.append(test['words'][i])
      ss.append(test['labels'][i])
    else:
      df_sen.at[no_of_sen, 'text'] = s
      s=[]
      entities_per_line.at[no_of_sen, 'labels'] = ss
      ss=[]
      no_of_sen = no_of_sen+1
  #sen_entities = pd.DataFrame().assign(sentence_id = df_sen['sentence'].index.astype(str) ,sentence=df_sen['sentence'], entities=entities_per_line['entities'])
  sen_entities = pd.DataFrame().assign(sentence_id = df_sen['text'].index.astype(str) ,text=df_sen['text'], labels=entities_per_line['labels'])
  return sen_entities

def flatten_num(data):
  ''' 
  Function that flatten the list of lists of the labels to be 1Dim 
  
  Pararmeters:
    data(DataFrame of lists): lists of the labels of each sentence

  Returns:
    flat_no(list): flatten list of the entites after mapping to their IDs.
    flat(list): flatten list of the entites.
    
  '''
  irregular_list = data.tolist()
  # Using lambda arguments: expression
  flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]
  flat =  flatten_list(irregular_list)
  flat_no = pd.DataFrame(flat,columns=['labels'])
  flat_no['labels'] = flat_no['labels'].replace(['B-LOC','B-MISC','B-ORG','B-PERS','I-LOC','I-MISC','I-ORG','I-PERS','O'], [0,1,2,3,4,5,6,7,8])
  flat_no = flat_no['labels'].tolist()
  return flat_no,flat

def flatten(data):
  ''' 
  Function that flatten the list of lists of the labels to be 1Dim 
  
  Pararmeters:
    data(DataFrame of lists): lists of the labels of each sentence

  Returns:
    flat(list): flatten list of the entites.
  '''
  #irregular_list = data.tolist()
  irregular_list = data
  flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]
  flat =  flatten_list(irregular_list)
 
  return flat


def DT():
  ''' 
  Function that gers the date and timezone in Egypt

  Returns:
  egypt_time_str(string): A string that contains the date and time in Egypt
   '''
  egypt_tz = pytz.timezone('Egypt')
  egypt_time = datetime.now(egypt_tz)
  return egypt_time.strftime('D:%Y-%m-%dH-%H:%M:%S')
