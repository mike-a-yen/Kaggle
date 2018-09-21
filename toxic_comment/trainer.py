import torch
import torch.nn as nn

CLASS_LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

def pad_sequence(seq,maxlen,value):
    diff = maxlen-len(seq)
    return diff*[value]+seq

class Batch(object):
    def __init__(self,sample,cuda=True):
        columns = CLASS_LABELS
        self.sample = sample
        self.maxlen = max(map(len,self.sample['encoded'].values))
        tokens = [pad_sequence(seq,self.maxlen,7330) for seq in sample['encoded'].values]
        self.torch_input = torch.LongTensor(tokens).contiguous().transpose(0,1)
        self.outputs = torch.FloatTensor(self.sample[columns].values).contiguous()
        if cuda:
            self.torch_input = self.torch_input.cuda()
            self.outputs = self.outputs.cuda()
        self.size = len(sample)

class Trainer(object):
    def __init__(self,model,log_inteval=100,cuda=True):
        self.model = model
        self.cuda = cuda
        self.loss = nn.BCELoss()
        self.batch_size = 16
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
        corr = Y.eq((Y_hat>0.5).to(torch.float32)).sum().item()
        return cost,corr

    def train_epoch(self,data,opt):
        self.model.train()
        n_batches = len(data)//self.batch_size+1
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        loss = 0
        acc = 0
        sample_count = 0
        batch_count = 0
        for slow in range(0,len(data)-1,self.batch_size):
            self.model.zero_grad()
            sample = data.iloc[indices[slow:slow+self.batch_size]]
            batch = Batch(sample,self.cuda)
            cost,corr = self.train_batch(batch)
            acc += corr
            loss += batch.size*cost.item()
            cost.backward()
            opt.step()
            batch_count += 1
            sample_count += batch.size
            if batch_count%self.log_interval==0:
                int_loss = loss/sample_count
                int_acc = 100*acc/(sample_count*self.model.clf.W.out_features)
                message = "| Epoch {} | {}/{} batch | Interval loss: {:4.6f} | Interval Acc: {:0.2f}% |"
                message = message.format(1,batch_count,n_batches,int_loss,int_acc)
                print(message)
        self.model.zero_grad()
        loss /= len(data)
        acc /= (len(data)*self.model.clf.W.out_features)
        return loss,acc

    def train(self,data,opt,epochs,val=None):
        for epoch_id in range(1,epochs+1):
            print('-'*79)
            train_loss,train_acc = self.train_epoch(data,opt)
            train_acc *= 100
            if val:
                val_loss,val_acc = self.evaluate(val)
                val_acc *= 100
            else:
                val_loss = None
                val_acc = None
            message = "| End Epoch {} | train loss: {} | train acc: {:0.2f}% | val loss {} | val acc {:0.2f}%"
            message = message.format(epoch_id,train_loss,train_acc,val_loss,val_acc)
            print(message)

    def evaluate(self,data):
        self.model.eval()
        loss = 0
        acc = 0
        for slow in range(0,len(data)-1,self.batch_size):
            sample = data.iloc[slow:slow+self.batch_size]
            batch = Batch(sample)
            Y_hat = self.predict_batch(batch)
            loss += batch.size*self.loss(Y_hat,batch.outputs).item()
            y_pred = (Y_hat>0.5).to(float32)
            acc += (batch.outputs.eq(y_pred)).sum().item()
        loss /= len(data)
        acc /= len(data)*self.model.clf.W.out_features
        return loss,acc

if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    import numpy as np
    from gensim.models import KeyedVectors
    from model import RNNEncoder, Classifier, ToxicClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path")
    parser.add_argument("--wv-path")
    parser.add_argument("--save-path")
    parser.add_argument("--log-interval",default=100,type=int)
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

    epochs = 4
    lr = 0.005
    rnn_size = 128
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
    print('Training...')
    trainer.train(train,opt,epochs,val)
