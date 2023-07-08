'''

#recives test dataset processed 
#receive model path, tokenizer path

from transformers import pipeline

#model_name = tokenizer path

model_fine_tuned = AutoModelForTokenClassification.from_pretrained("/content/gdrive/MyDrive/RDI/weights/ner_model")
tokenizer = BertTokenizer.from_pretrained(model_name)

nlp = pipeline(
    "token-classification",
    model=model_fine_tuned,
    tokenizer=tokenizer, grouped_entities=True
)

example = "جلس الرئيس الراحل جورج واشنتون مع ملكه المملكه المتحده في حديقه نيورك"
ner_results = nlp(example)

print(ner_results)

'''