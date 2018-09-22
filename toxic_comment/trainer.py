import torch
import torch.nn as nn

import time
import numpy as np
from sklearn.metrics import roc_auc_score

CLASS_LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

def class_mean_auc(y_pred,y_true):
    true = np.array(y_true)
    pred = np.array(y_pred)
    n_classes = true.shape[1]
    score = 0.
    for i in range(n_classes):
        score += roc_auc_score(true[:,i],pred[:,i])
    return score/n_classes

def pad_sequence(seq,maxlen,value):
    diff = maxlen-len(seq)
    return diff*[value]+seq

class Batch(object):
    def __init__(self,sample,pad_val,cuda=True):
        columns = CLASS_LABELS
        self.sample = sample
        self.maxlen = max(map(len,self.sample['encoded'].values))
        tokens = [pad_sequence(seq,self.maxlen,pad_val) for seq in sample['encoded'].values]
        self.torch_input = torch.LongTensor(tokens).contiguous().transpose(0,1)
        self.outputs = torch.FloatTensor(self.sample[columns].values).contiguous()
        if cuda:
            self.torch_input = self.torch_input.cuda()
            self.outputs = self.outputs.cuda()
        self.size = len(sample)

class Trainer(object):
    def __init__(self,model,log_inteval=100,cuda=True):
        self.vocab_size = model.encoder.embedding.num_embeddings
        self.pad_val = self.vocab_size - 1
        self.model = model
        self.cuda = cuda
        self.loss = nn.BCELoss()
        self.batch_size = 32
        self.log_interval = log_inteval

    def predict_batch(self,batch):
        X = batch.torch_input
        H,C = self.model.encoder.init_hidden(batch.size)
        if self.cuda:
            H = H.cuda()
            C = C.cuda()
        Y_hat = self.model(X,(H,C))
        return Y_hat

    def train_batch(self,batch):
        Y = batch.outputs
        Y_hat = self.predict_batch(batch)
        cost = self.loss(Y_hat,Y)
        return cost,Y_hat

    def train_epoch(self,data,opt):
        self.model.train()
        n_batches = len(data)//self.batch_size+1
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        loss = 0
        auc= 0
        sample_count = 0
        batch_count = 0
        preds = []
        trues = []
        start_time = time.time()
        for slow in range(0,len(data)-1,self.batch_size):
            self.model.zero_grad()
            sample = data.iloc[indices[slow:slow+self.batch_size]]
            batch = Batch(sample,self.pad_val,self.cuda)
            cost,y_pred = self.train_batch(batch)
            loss += batch.size*cost.item()
            cost.backward()
            opt.step()
            for p,t in zip(y_pred.cpu().detach().numpy(),batch.outputs.cpu().numpy()):
                preds.append(p)
                trues.append(t)
            batch_count += 1
            sample_count += batch.size
            if batch_count%self.log_interval==0:
                elapsed = time.time()-start_time
                int_loss = loss/sample_count
                auc = class_mean_auc(preds,trues)
                int_auc = 100*auc
                message = "| Epoch {} | {}/{} batch| {:4.1f} sec| Interval loss: {:4.6f} | Interval AUC: {:0.2f}% |"
                message = message.format(1,batch_count,n_batches,elapsed,int_loss,int_auc)
                print(message)
                start_time = time.time()
        self.model.zero_grad()
        loss /= len(data)
        auc = class_mean_auc(preds,trues)
        return loss,auc

    def train(self,data,opt,epochs,stepper=None,val=None):
        history = {'train_loss':[],'train_auc':[],
                   'val_loss':[],'val_auc':[]}
        if val is not None:
            val_loss,val_auc = self.evaluate(val)
            val_auc *= 100
            message = "| Initial | Val loss: {:0.6f} | Val AUC: {:2.2f}% |"
            message = message.format(val_loss,val_auc)
            print(message)
        for epoch_id in range(1,epochs+1):
            print('-'*79)
            train_loss,train_auc = self.train_epoch(data,opt)
            train_auc *= 100
            if val is not None:
                val_loss,val_auc = self.evaluate(val)
                val_auc *= 100
            else:
                val_loss = None
                val_auc = None
            message = "| End Epoch {} | train loss: {:4.6f} | train AUC: {:0.2f}% | val loss {:4.6f} | val AUC {:0.2f}%"
            message = message.format(epoch_id,train_loss,train_auc,val_loss,val_auc)
            print(message)
            history['train_loss'].append(train_loss)
            history['train_auc'].append(train_auc)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            if stepper is not None:
                stepper.step()
            return history

    def evaluate(self,data):
        self.model.eval()
        loss = 0
        auc = 0
        trues = []
        preds = []
        for slow in range(0,len(data)-1,self.batch_size):
            sample = data.iloc[slow:slow+self.batch_size]
            batch = Batch(sample,self.pad_val,self.cuda)
            Y_hat = self.predict_batch(batch)
            loss += batch.size*self.loss(Y_hat,batch.outputs).item()
            for p,t in zip(Y_hat.cpu().detach().numpy(),batch.outputs.cpu().numpy()):
                trues.append(t)
                preds.append(p)
        loss /= len(data)
        auc = class_mean_auc(preds,trues)
        return loss,auc

if __name__ == "__main__":
    import argparse
    import os
    import json
    import pandas as pd
    import numpy as np
    from gensim.models import KeyedVectors
    from model import RNNEncoder, Classifier, ToxicClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path")
    parser.add_argument("--wv-path")
    parser.add_argument("--save-path")
    parser.add_argument("--log-interval",default=500,type=int)
    args = parser.parse_args()
    file_path = args.file_path
    wv_path = args.wv_path
    save_path = args.save_path
    log_interval = args.log_interval

    n_classes = len(CLASS_LABELS)
    train = pd.read_json(os.path.join(file_path,'train_tokenized.json'))
    val = pd.read_json(os.path.join(file_path,'val_tokenized.json'))
    wv = KeyedVectors.load(wv_path)

    weights = wv.syn0
    unk = weights.mean(axis=0)[np.newaxis,:]
    embeddings = np.vstack([weights,unk])
    print('Word Embeddings:',embeddings.shape)

    epochs = 16
    lr = 0.01
    step_size = 2
    gamma = 0.8

    rnn_size = 256
    rnn_layers = 2
    dropout = 0.3
    bi = True
    cuda = True

    encoder = RNNEncoder(embeddings,rnn_size,rnn_layers,dropout,bi)
    clf = Classifier((1+bi)*rnn_size,n_classes)
    model = ToxicClassifier(encoder,clf)
    if cuda:
        model.cuda()

    trainer = Trainer(model,log_interval,cuda)
    opt = torch.optim.Adam(model.parameters(),lr)
    stepper = torch.optim.lr_scheduler.StepLR(opt,step_size,gamma=gamma)
    print('Training...')
    history = trainer.train(train,opt,epochs,stepper,val)

    torch.save(trainer.model.state_dict(),
               os.path.join(save_path,'toxic_model.state'))
    model_params = {'rnn_size':rnn_size,
                    'rnn_layers':rnn_layers,
                    'dropout':dropout,
                    'bi':bi,
                    'n_classes':n_classes,
                    'history':history}
    json.dump(model_params,
              open(os.path.join(save_path,'toxic_model.params'),'w'))
