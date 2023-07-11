from fastapi import FastAPI, HTTPException
from joblib import load

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from fastapi_standalone_docs import StandaloneDocs
import json
#import utils

# Initialize an instance of FastAPI
app = FastAPI()
StandaloneDocs(app=app)

#nlp = utils.model_tokenizer()

model = AutoModelForTokenClassification.from_pretrained('./nerModel')
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
nlp = pipeline("ner", model=model, tokenizer=tokenizer,ignore_labels = [] )
    
@app.get("/")
def root():
    return {"message": "Welcome to my first FastAPI application on docker "}

# Define the route to the NER predictor
@app.post("/predict_NER")
def predict_NER(text_message):

    if(not(text_message)):
        raise HTTPException(status_code=400, detail = "Please Provide a valid text message")
    
    punc = '،-!"#$%&\'()*+,.-–/:;<=>?@[\ـ\]^_`{؛|}~«؟'
    sent = text_message.translate (str.maketrans('', '', punc))
    
    ner_results = nlp(sent)
    test = sent.split(' ')

    test = list(filter(lambda x: x != '', test))
    l=[]
    for j in range(len(ner_results)):
        if '##' not in ner_results[j]['word']:
          l.append(ner_results[j]['entity'])

    return {
            "text_message": test, 
            "labels": l
           }
