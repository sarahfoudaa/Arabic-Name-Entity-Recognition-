# Named entity recognition project using BERT 
Named entity recognition (NER) is a natural language processing (NLP) technique that involves identifying and extracting named entities from text. Named entities are objects or concepts in text that have specific names, such as people, organizations, locations, dates, times, and numerical values.

## Table of contents
* [Requirments](#Requirments)
* [Dataset](#Dataset)
* [Presrocessing](#Pre-processing)
* [Model](#Model)
* [Results](#Results)
* [Weights](#Weights)
* [Structure](#Structure)
* [Deployment](#Deployment)

# Requirments
* Dataset==1.6.0
* datasets==2.13.1
* transformers[torch]==4.30.2
* fastapi_standalone_docs==0.1.2

# Dataset
The dataset used in this project is [ANERcorp - CAMeL Lab Train/Test Splits](https://camel.abudhabi.nyu.edu/anercorp/) 
The sentences containing the first 5/6 of the words go to train and the rest go to test. The train split has 125,102 words and the test split has 25,008 words.

* B-LOC/I-LOC:: Beginning/Inside of a location entity.
* B-MISC/I-MISC: Beginning/Inside of a miscellaneous entity (i.e. entities that do not fit into any of the other categories).
* B-ORG/I-ORG: Beginning/Inside of an organization entity.
* B-PERS/I-PERS: Beginning/Inside of a person entity.
* O: Not part of any named entity.

Number of words
* Train --> 110119
* Test --> 22561

Number of sentences 
* Train --> 3972
* Test --> 924

Test dataset file
* Min number of words in a sentence --> 2
* Max number of words in a sentence --> 146
* Average number of words in a sentence --> 25.98
* Standard deviation number of words in a sentence --> 16.99

Train dataset file
* Min number of words in a sentence --> 2
* Max number of words in a sentence --> 530
* Average number of words in a sentence --> 30.10
* Standard deviation number of words in a sentence --> 23.24

The train file dataset is then split by a percentage of 70:30 to have the train and validation datasets

# Pre-processing
The dataset had two preprocessing steps 

1. Cleaning
* removing punctuation
* removing rows with white spaces

2. Restructuring
After removing the punctuation there were two ways to structure the dataset to train and evaluate the model
* Train/Validat dataset --> Using the white space as a separator between each sentence, put all the words of each sentence in a list corresponding to it a list of their labels all in a data frame
* Test dataset --> Using the white space as a separator between each sentence, concatenate all the words of each sentence in a string corresponding to it a list of their labels all in a data frame


# Model
[CAMeL-Lab/bert-base-arabic-camelbert-mix-ner](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-ner)
CAMeLBERT-Mix NER Model is a Named Entity Recognition (NER) model that was built by fine-tuning the [CAMeLBERT Mix](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix/) model. For the fine-tuning, we used the ANERcorp [dataset](https://camel.abudhabi.nyu.edu/anercorp/).

# Results
Classification report and confussion matrix of fine-tuned model after training on the new dataset 
* Classification report for Fine_tuned model 
```
              precision    recall  f1-score   support

       B-LOC       0.90      0.95      0.92       665
      B-MISC       0.75      0.63      0.68       235
       B-ORG       0.78      0.74      0.76       450
      B-PERS       0.88      0.86      0.87       857
       I-LOC       0.82      0.81      0.81        83
      I-MISC       0.75      0.37      0.49       163
       I-ORG       0.77      0.69      0.73       275
      I-PERS       0.90      0.88      0.89       638
           O       0.98      0.99      0.99     19093

    accuracy                           0.97     22459
   macro avg       0.84      0.77      0.79     22459
weighted avg       0.96      0.97      0.96     22459

```  

* Confussion matrix for Fine_tuned model 
```
	    B-LOC    B-MISC   B-ORG   B-PERS  I-LOC   I-MISC  I-ORG   I-PERS    O
B-LOC     632       1       7       3       0       0       3       1      18  
B-MISC      6     148      12       4       0       2       0       3      60  
B-ORG      22       5     333      19       4       0       3       0      64  
B-PERS     12      12      17     735       0       1       0      36      44  
I-LOC       5       1       2       0      67       1       0       0       7  
I-MISC      8      10       2       1       4      60      18       2      58  
I-ORG       3       0      16       3       2       1     190       9      51  
I-PERS      3       2       0      41       1       5       5     561      20  
O          11      19      38      29       4      10      27       8   18947  
```

# Weights

To download the latest run model click the [link](https://drive.google.com/drive/folders/1Sq352cLfmxkDocm0AuZQ5YYzdHjcRQnL?usp=sharing)

disclaimer: the model weights and the results are of 6 epochs

# Structure

```
RDI-NER
 ├─ ANERcorp-CamelLabSplits
 │   ├─ ANERCorp_Benajiba.txt
 │   ├─ ANERCorp_CamelLab_test.txt    
 │   ├─ ANERCorp_CamelLab_train.txt
 │   ├─ README.txt
 │   └─ ...
 ├─ docker
 │   ├─ Dockerfile
 │   └─ ...
 ├─ app.py
 ├─ dataset_proc.py   
 ├─ evaluate.py
 ├─ model.py
 ├─ infer.py
 ├─ split.py
 ├─ test.ipynb
 ├─ tokenization_label.py
 ├─ train.py
 ├─ utils.py
 ├─ README.txt
 └─ ...
```

# Deployment 
  * Step 1: Building the API
  Fastapi
app.py containing all the instructions on the server-side
the user/client sends a request to the uvicorn server which interacts with the API to trigger the prediction model.  

Run the API
```
uvicorn app:app --reload
```
For default route 
```
http://127.0.0.1:8000/
```
For other routes
```
http://127.0.0.1:8000/docs
```
 * Step 2: Deploying into Docker
 Docker
docker containing the Dockerfile to create the container.

Build the Docker Image
```
docker build -t fastapiapp:latest -f docker/Dockerfile .
```
Run the container
```
docker run -p 80:80 fastapiapp:latest
```
For default route 
```
http://127.0.0.1:8000/
```
for other routes
```
http://127.0.0.1:8000/docs
```

