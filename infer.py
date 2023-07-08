#receive sentence 
#return labels of the sentence

from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForTokenClassification
import sys

args = sys.argv
#arge parse
def remove_punc_input(sent):
  ''' 
  Function that removes punctuation from a string 

  Pararmeters:
    sent(string): user input sequence (sentence) that might have punc in it

  Returns:
    sent(string): same input but w/o punc
    test(list): sent in a list the separator is ' '
   '''
  punc = '،-!"#$%&\'()*+,.-–/:;<=>?@[\ـ\]^_`{؛|}~«؟'
  sent = sent.translate (str.maketrans('', '', punc))
  test = sent.split(' ')

  return sent, test


def init(Model,Tokenizer):
  ''' 
  Function that creats an instance of a pre-trained model and pre-trained tokenizer for token classification.
  
  Pararmeters:
    Model(string): path to the pre-trained model from huggingface
    Tokenizer(string): path to the pre-trained tokenizer from huggingface

  Returns:
    nlp: A callable object of pipeline() that represents a Named Entity Recognition (NER) pipeline in the Hugging Face Transformers library.
    
   '''
  model = AutoModelForTokenClassification.from_pretrained(Model)
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer)
  nlp = pipeline("ner", model=model, tokenizer=tokenizer,ignore_labels = [] )
  return nlp

def infer(nlp,sen):
  ''' 
  Function that predicts the labels of the inputed sentes
  This Function removes the extra words after tokenization return the list of words to be the same size as the original sentence
  
  Pararmeters:
    nlp: A callable object of pipeline() that represents a Named Entity Recognition (NER) pipeline in the Hugging Face Transformers library.
    sen(string):

  Returns:
    w(list): List of words
    l(list): List of labels
   '''
  sen,x = remove_punc_input(sen)
  ner_results = nlp(sen)
  w = sen.split(' ')
  w = list(filter(lambda x: x != '', w))
  l=[]
  for j in range(len(ner_results)):
    if '##' not in ner_results[j]['word']:
      l.append(ner_results[j]['entity'])
        
  return w,l

def main():
  nlp = init('/content/drive/MyDrive/RDI/weights/ner_model','/content/drive/MyDrive/RDI/weights/tokenizer')
  w,l = infer(nlp,sys.argv[1])
  print(w)
  print(l)

if __name__ == '__main__':
  main()
  

