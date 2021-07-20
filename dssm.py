import torch 
import torch.nn as nn
import torchtext
from transformers import BertTokenizer
import numpy as np

class DSSM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, dropout):
        super(DSSM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        print(a.shape)
        a1 = self.embed(a).sum(1)
        b1 = self.embed(b).sum(1)
        print(a1.shape)
        a2 = self.dropout(torch.tanh(self.fc1(a1)))
        a3 = self.dropout(torch.tanh(self.fc2(a2)))
        a4 = self.dropout(torch.tanh(self.fc3(a3)))

        b2 = self.dropout(torch.tanh(self.fc1(b1)))
        b3 = self.dropout(torch.tanh(self.fc2(b2)))
        b4 = self.dropout(torch.tanh(self.fc3(b3)))

        print(a4.shape)
        print(b4.shape)

        cosine = torch.cosine_similarity(a4, b4, dim=1, eps=1e-8)   # 计算两个句子的余弦相似度
        return cosine
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


if __name__ =='__main__':

    # tokenizer = torchtext.data.get_tokenizer() # 默认就是split()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') 

    VOCAB_SIZE = tokenizer.vocab_size
    MAX_LEN = 64

    sentences = ['我是谁，我在哪里', '我爱你']
    
    sentences_cls = ["[CLS] " + s.strip() for s in sentences]
    tokenized_sentences = [tokenizer.tokenize(s) for s in sentences_cls]
    tokenized_len_limit_sentences = [t[:MAX_LEN - 1] + ['SEP'] for t in tokenized_sentences]
    ids_sentences = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_len_limit_sentences]
    ids_sentences_with_padding = np.array([np.pad(i, (0, MAX_LEN - len(i)), mode='constant') for i in ids_sentences]) 
    input_vector = torch.tensor(ids_sentences_with_padding)
    
    print(input_vector[0])
    print(input_vector[0].shape)


    model = DSSM(VOCAB_SIZE, 100, 0.2)
    model._init_weights()
    print(model)

    result = model(input_vector[0].unsqueeze(0), input_vector[1].unsqueeze(0))
    print(result)


    
    
    
    
    # github下载whl文件 https://github.com/explosion/spacy-models/releases/tag/zh_core_web_sm-3.1.0
    # pip install zh_core_web_sm-3.1.0-py3-none-any.whl 
    
    # import spacy
    # from spacy.lang.zh.examples import sentences 
    # nlp = spacy.load("zh_core_web_sm")
    # doc = nlp(sentences[0])
    # print(doc)
    # print(doc.text)
    # for token in doc:
    #     print(token.text, token.pos_, token.dep_)