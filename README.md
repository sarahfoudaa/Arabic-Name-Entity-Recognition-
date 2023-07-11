# RDI-NER

# Requirments
* Dataset==1.6.0
* datasets==2.13.1
* transformers[torch]==4.30.2
* fastapi_standalone_docs==0.1.2

# Enviroment

# Dataset

# Usage

# Model

# Results

Classification report for Fine_tuned model 
             | precision   | recall | f1-score  | support|
			 |-------------|--------|-----------|--------|
       B-LOC |      0.90   |   0.95 |     0.92  |     665|
      B-MISC |      0.75   |   0.63 |     0.68  |     235|
       B-ORG |      0.78   |   0.74 |     0.76  |     450|
      B-PERS |      0.88   |   0.86 |     0.87  |     857|
       I-LOC |      0.82   |   0.81 |     0.81  |      83|
      I-MISC |      0.75   |   0.37 |     0.49  |     163|
       I-ORG |      0.77   |   0.69 |     0.73  |     275|
      I-PERS |      0.90   |   0.88 |     0.89  |     638|
           O |      0.98   |   0.99 |     0.99  |   19093|
			 |             |        |           |        |
    accuracy |             |        |     0.97  |   22459|
   macro avg |      0.84   |   0.77 |     0.79  |   22459|
weighted avg |      0.96   |   0.97 |     0.96  |   22459|




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


Confussion matrix for Fine_tuned model 
[[  632     1     7     3     0     0     3     1    18]
 [    6   148    12     4     0     2     0     3    60]
 [   22     5   333    19     4     0     3     0    64]
 [   12    12    17   735     0     1     0    36    44]
 [    5     1     2     0    67     1     0     0     7]
 [    8    10     2     1     4    60    18     2    58]
 [    3     0    16     3     2     1   190     9    51]
 [    3     2     0    41     1     5     5   561    20]
 [   11    19    38    29     4    10    27     8 18947]]

# Weigths
To download the latest run model click the [link](https://drive.google.com/drive/folders/1Sq352cLfmxkDocm0AuZQ5YYzdHjcRQnL?usp=sharing)


disclaimer the model weights and the results are of 6 epochs
