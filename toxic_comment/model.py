import torch
import torch.nn as nn
import torch.nn.functional as F

class ToxicClassifier(nn.Module):
    def __init__(self,encoder,clf):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.clf = clf

    def forward(self,word_ids,hidden):
        encoded,hidden = self.encoder(word_ids,hidden)
        logits = self.clf(encoded[-1])
        return torch.sigmoid(logits)

class RNNEncoder(nn.Module):
    def  __init__(self,embedding_weights,rnn_size,rnn_layers,dropout,bi):
        nn.Module.__init__(self)
        self.drop = nn.Dropout(dropout)
        vocab_size = embedding_weights.shape[0]
        embedding_size = embedding_weights.shape[1]
        self.embedding = nn.Embedding(vocab_size,embedding_size,padding_idx=vocab_size-1)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        self.rnn = nn.LSTM(embedding_size,rnn_size,rnn_layers,dropout=dropout,bidirectional=bi)

        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.bi = bi

    def forward(self,word_ids,hidden):
        word_embeddings = self.drop(self.embedding(word_ids))
        encoded,hidden = self.rnn(word_embeddings,hidden)
        encoded = self.drop(encoded)
        return encoded,hidden

    def init_hidden(self,batch_size):
        bi = 1+self.bi
        h_state = torch.zeros((bi*self.rnn_layers,batch_size,self.rnn_size))
        c_state = torch.zeros((bi*self.rnn_layers,batch_size,self.rnn_size))
        return h_state,c_state

class Classifier(nn.Module):
    def __init__(self,input_size,n_classes):
        nn.Module.__init__(self)
        self.W = nn.Linear(input_size,n_classes)

    def forward(self,X):
        logits = self.W(X)
        return logits
