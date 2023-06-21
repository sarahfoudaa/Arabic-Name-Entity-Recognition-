#receives train and val sets
#receives instant of the model from file model.py
#receive paths to where the model and the tokenizer should be saved

import accelerate
from transformers import TrainingArguments, Trainer

from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

from datasets import Dataset, DatasetDict
import datasets

import accelerate
from transformers import TrainingArguments, Trainer

import json

#import tokenizer from file
args = TrainingArguments(
    "/content/gdrive/MyDrive/RDI/task 1/for production",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer,max_length = 512)

metric = datasets.load_metric("seqeval")

#compute metric function

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

model.save_pretrained("/content/gdrive/MyDrive/RDI/weights/ner_model")

tokenizer.save_pretrained("/content/gdrive/MyDrive/RDI/weights/tokenizer")


id2label = {
    str(i): label for i,label in enumerate(label_list)
}
label2id = {
    label: str(i) for i,label in enumerate(label_list)
}


config = json.load(open("/content/gdrive/MyDrive/RDI/weights/ner_model/config.json"))

config["id2label"] = id2label
config["label2id"] = label2id

json.dump(config, open("/content/gdrive/MyDrive/RDI/weights/ner_model/config.json","w"))

model_fine_tuned = AutoModelForTokenClassification.from_pretrained("/content/gdrive/MyDrive/RDI/weights/ner_model")
