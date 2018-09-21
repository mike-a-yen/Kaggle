import pandas as pd
import spacy
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

nlp = spacy.load('en')

def tokenize(text):
    return [t.text for t in nlp(text)]

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path")
    parser.add_argument("--save-path")
    args = parser.parse_args()
    file_path = args.file_path
    save_path = args.save_path

    raw_data = pd.read_csv(file_path)
    data_size = len(raw_data)
    print("Read",data_size,"rows.")
    desc = raw_data.describe().loc['mean']
    print('Class Distribution:')
    print(desc)
    print()
    print('Tokenizing...')
    raw_data['tokens'] = raw_data['comment_text'].apply(tokenize)
    train,val = train_test_split(raw_data,test_size = 0.15)
    train_size = len(train)
    val_size = len(val)
    train.to_json(os.path.join(save_path,'train_tokenized.json'))
    val.to_json(os.path.join(save_path,'train_tokenized.json'))
    print('Training size:',train_size)
    print('Validation size:',val_size)

    print('Training Word2Vec....')
    train_comments = ([t.lower() for t in sent] for sent in train['tokens'])
    w2v = Word2Vec(train_comments,min_df=5,max_features=20000)
    w2v.wv.save(os.path.join(save_path,'keyed_vectors.wv'))
