import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import accelerate
from transformers import TrainingArguments, Trainer

from transformers import BertTokenizerFast, DataCollatorForTokenClassification, AutoModelForTokenClassification

from datasets import Dataset, DatasetDict
import datasets

import accelerate
from transformers import TrainingArguments, Trainer

import json

import utils,dataset_proc,tokenization_label

from model import BertClassifierModel



def compute_metrics(eval_preds):
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
    pred_logits, labels = eval_preds
    label_list =['B-LOC','B-MISC','B-ORG','B-PERS','I-LOC','I-MISC','I-ORG','I-PERS','O']
    pred_logits = np.argmax(pred_logits, axis=2)
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(pred_logits, labels)
   ]

    precision, recall, f1_score, _ = precision_recall_fscore_support(utils.flatten(true_labels), utils.flatten(predictions), average='weighted')
    accuracy = accuracy_score(utils.flatten(true_labels), utils.flatten(predictions))

    return {
      "precision": precision,
      "recall": recall,
      "f1": f1_score,
      "accuracy": accuracy,
  }

def data(raw_data, tokenizer):

  train = dataset_proc.read_dataset(raw_data)
  pro_train = utils.preprocess_train(train)
  my_dataset_dict = dataset_proc.dictionary(pro_train)
  tokenized_datasets = my_dataset_dict.map(tokenization_label.tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})
  return tokenized_datasets

def main():
  label_list =['B-LOC','B-MISC','B-ORG','B-PERS','I-LOC','I-MISC','I-ORG','I-PERS','O']
  
  tokenizer = BertTokenizerFast.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
  model = BertClassifierModel(num_classes=9)
  
  tokenized_datasets = data('ANERcorp-CamelLabSplits/ANERCorp_CamelLab_train.txt', tokenizer)

  egypt_time_str = utils.DT()
  print(egypt_time_str)

  args = TrainingArguments(
    "weigths/"+egypt_time_str+"/check",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    label_names = ['labels'],
    label_smoothing_factor = 0.001
  )
  
  data_collator = DataCollatorForTokenClassification(tokenizer,max_length = 512)

  trainer = Trainer(
    model,
    args,
   train_dataset=tokenized_datasets["train"],
   eval_dataset=tokenized_datasets["validation"],
   data_collator=data_collator,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics 
  )
  
  trainer.train()

  model.save_pretrained("weigths/"+egypt_time_str+"/ner_model")
  tokenizer.save_pretrained("weigths/"+egypt_time_str+"/tokenizer")


  id2label = {
      str(i): label for i,label in enumerate(label_list)
  }
  label2id = {
      label: str(i) for i,label in enumerate(label_list)
  }

  config = json.load(open("weigths/"+egypt_time_str+"/ner_model/config.json"))
  config["id2label"] = id2label
  config["label2id"] = label2id

  json.dump(config, open("weigths/"+egypt_time_str+"/ner_model/config.json","w"))

  model_fine_tuned = AutoModelForTokenClassification.from_pretrained("weigths/"+egypt_time_str+"/ner_model") 

  print("model saved succefully to " + os.getcwd() + "/weigths/"+egypt_time_str)

if __name__ == '__main__':
  main()
