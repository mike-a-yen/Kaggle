# Toxic Comments

## Directory Tree
```
./
├── data
│   ├── all.zip
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── test_labels.csv
│   └── train.csv
├── models
├── prepare_data.py
└── README.md
```

## Prepare Data
Tokenize and split data into training and validation 85-15. 
A word2vec model is then trained on the training set.
This step takes a while, its mostly the tokenization.
```
$ python prepare_data.py --file-path data/train.csv --save-path data/
 Read 159571 rows.
 Class Distribution:
 toxic            0.095844
 severe_toxic     0.009996
 obscene          0.052948
 threat           0.002996
 insult           0.049364
 identity_hate    0.008805
 Name: mean, dtype: float64

 Tokenizing...
```
Creates `data/<train|val>.json` with comments tokenized.
