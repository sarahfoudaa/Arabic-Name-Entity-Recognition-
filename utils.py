#functions that will be called from other files
#preprocesssing 

import pandas as pd

#for the test
def sentence(test):
  punc = '،-!"#$%&\'()*+,.-–/:;<=>?@[\ـ\]^_`{؛|}~«؟'
  punctuation = set(punc)
  # Create boolean mask to identify rows containing punctuation
  mask = test['words'].apply(lambda x: any(char in punctuation for char in x))

  # Filter original DataFrame using boolean mask
  new_df = test[mask]
  test_wo_punc = test[mask==False]
  test_wo_punc = test_wo_punc.reset_index(drop=True)
  test = test_wo_punc


  no_of_sen =0
  df_sen = []
  s = ''
  entities_per_line = []
  entities = ['O', 'B-PERS', 'B-LOC', 'I-PERS', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'I-LOC']
  ss = ''
  df = pd.DataFrame(columns = ['labels'])
  l = []
  for i in range (len(test)):
    if test['words'][i]!='':
      s = s +" "+ test['words'][i]
      no_of_sen = no_of_sen+1
      if test['labels'][i] in entities:
        ss = ss +" "+ test['labels'][i]
        l.append(test['labels'][i])
    else:
      df_sen.append(s)
      s=''
      entities_per_line.append(ss)
      ss=''
      #print("============",l)
      df.at[i, 'labels'] = l
      l=[]
  df_sen = pd.DataFrame(df_sen, columns=['sentence'])
  entities_per_line = pd.DataFrame(entities_per_line, columns=['entities'])

  df = df.reset_index()
  #print(df)
  sen_entities = pd.DataFrame().assign(sentence=df_sen['sentence'], entities=entities_per_line['entities'],list_entities = df['labels'])
  return sen_entities

  ''' sent flag in the parameters  
  if test return ...
  if train return ...
  '''


