from transformers import AutoTokenizer,AutoModelForTokenClassification, pipeline

import pandas as pd
from datasets import Dataset, DatasetDict

from sklearn.metrics import classification_report, confusion_matrix

import dataset_proc, infer, utils

import sys

args = sys.argv

def prediction(test_sent,nlp):
  ''' 
  Function that predicts the labels of the sentences anf append them into a dataframe and removes the extra words and labels after tokenization
  
  Pararmeters:
    test_sent(DataFrame): DataFrame that has 2 columns,sentence and labels.
    nlp: A callable object of pipeline() that represents a Named Entity Recognition (NER) pipeline in the Hugging Face Transformers library.

  Returns:
    pred(DataFrame): DataFrame of lists of labels of each sentence.
  '''
  pred = pd.DataFrame(columns=['pred_entites'])
  l = []
  for i in range(len(test_sent)): 
    ner_results = nlp(test_sent['sentence'][i])
    for j in range(len(ner_results)):
      if '##' not in ner_results[j]['word']:
        l.append(ner_results[j]['entity'])
    pred.at[i,'pred_entites'] = l
    l = []
  return pred

def matrix(test_sent, true_flat_ent, link_model_w, tokenizer_w,flag):
  ''' 
  Function that calculate and display the classification rreport and confussion matrix

  Pararmeters:
    test_sent(Dataset Dictionary): the test dataset 
    true_flat_ent(list): true labels flattened in a 1D list
    link_model_w(string): path to the model weigths 
    tokenizer_w(string): path to the tokenizer weights
    flag(string): decide if the paths sent was of the pretrained or of the baseline
  '''
  nlp = infer.init(link_model_w, tokenizer_w)
  pred  = prediction(test_sent,nlp)
  pred_flat_ent = utils.flatten(pred['pred_entites'])
  
  class_report = classification_report(true_flat_ent, pred_flat_ent, target_names=['B-LOC','B-MISC','B-ORG','B-PERS','I-LOC','I-MISC','I-ORG','I-PERS','O'])
  print("\nClassification report for "+flag + " \n" +class_report )
  #dataframe save csv   save stdout
  #report_df = pd.DataFrame(class_report).transpose()
  #report_df.to_csv('/content/drive/MyDrive/RDI/evaluations/report.csv', index=True)

  conf_m = confusion_matrix(true_flat_ent, pred_flat_ent)
  print("\nConfussion matrix for  "+flag + " \n" + str(conf_m))
  #conf_mat_df = pd.DataFrame(conf_m)
  #conf_mat_df.to_csv('/content/drive/MyDrive/RDI/evaluations/conf_mat.csv', index=False)

def main():
  test = dataset_proc.read_dataset('ANERcorp-CamelLabSplits/ANERCorp_CamelLab_test.txt')
  test_sent = utils.preprocess_test(test)
  true_flat_ent =  utils.flatten(test_sent['list_entities'])
  
  matrix(test_sent,true_flat_ent, sys.argv[1], sys.argv[2],sys.argv[3])

if __name__ == '__main__':
  main()

