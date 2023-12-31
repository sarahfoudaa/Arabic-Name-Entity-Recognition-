from torch import nn
from transformers import AutoTokenizer,AutoModelForTokenClassification
import utils
import json

class BertClassifierModel(nn.Module):
    def __init__(self, num_classes):
      super(BertClassifierModel, self).__init__()
      self.num_classes = num_classes
      self.model = AutoModelForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner", num_labels=9)

    def forward(self, input_ids, attention_mask):
      outputs = self.model(input_ids, attention_mask)
      return outputs

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
