#receives train and val sets
#receives instant of the model from file model.py
#receive paths to where the model and the tokenizer should be saved
import numpy as np

import accelerate
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizerFast, DataCollatorForTokenClassification, AutoModelForTokenClassification

from datasets import Dataset, DatasetDict
import datasets

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
    metric = datasets.load_metric("seqeval")

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(pred_logits, labels)
   ]
    results = metric.compute(predictions=predictions, references=true_labels)
    return {
   "precision": results["overall_precision"],
   "recall": results["overall_recall"],
   "f1": results["overall_f1"],
  "accuracy": results["overall_accuracy"],
  }
def data(raw_data, tokenizer):

  train = dataset_proc.read_dataset(raw_data)
  pro_train = utils.preprocess_train(train)
  my_dataset_dict = dataset_proc.dictionary(pro_train)
  print(my_dataset_dict)
  tokenized_datasets = my_dataset_dict.map(tokenization_label.tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})
  return tokenized_datasets

def main():
  label_list =['B-LOC','B-MISC','B-ORG','B-PERS','I-LOC','I-MISC','I-ORG','I-PERS','O']
  tokenizer = BertTokenizerFast.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
  #model = AutoModelForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner", num_labels=9)
  model = BertClassifierModel(num_classes=9)
  
  tokenized_datasets = data('/content/drive/MyDrive/RDI/task 1/ANERcorp-CamelLabSplits/ANERcorp-CamelLabSplits/ANERCorp_CamelLab_train.txt', tokenizer)

  egypt_time_str = utils.DT()
  print(egypt_time_str)

  args = TrainingArguments(
    "/content/drive/MyDrive/RDI/weights/"+egypt_time_str+"/check",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
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

 # model.save_pretrained(model,tokenizer)
 #save_model
  model.save_pretrained("weights/"+egypt_time_str+"/ner_model")
  tokenizer.save_pretrained("/content/drive/MyDrive/RDI/weights/"+egypt_time_str+"/tokenizer")
  #shutil.copy(source, distination) 

  id2label = {
      str(i): label for i,label in enumerate(label_list)
  }
  label2id = {
      label: str(i) for i,label in enumerate(label_list)
  }

  config = json.load(open("/content/drive/MyDrive/RDI/weights/"+egypt_time_str+"/ner_model/config.json"))
  config["id2label"] = id2label
  config["label2id"] = label2id

  json.dump(config, open("/content/drive/MyDrive/RDI/weights/"+egypt_time_str+"/ner_model/config.json","w"))

  model_fine_tuned = AutoModelForTokenClassification.from_pretrained("/content/drive/MyDrive/RDI/weights/"+egypt_time_str+"/ner_model") 

if __name__ == '__main__':
  main()
