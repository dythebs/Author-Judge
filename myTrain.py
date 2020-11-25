# 模型训练
# 导入相关包
import copy
import os
import numpy as np
import jieba as jb
import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchtext import data, datasets
from argparse import Namespace

from torchtext.data import Field, Dataset, Iterator, Example, BucketIterator

EMBEDDING_DIM = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./dataset"
split_ratio = 0.7


# OK
class Net_origin(nn.Module):
    def __init__(self):
        super(Net_origin, self).__init__()
        self.lstm = torch.nn.LSTM(1, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 5)
    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        output, hidden = self.lstm(x.unsqueeze(2).float())
        h_n = hidden[1]
        out = self.fc2(self.fc1(h_n.view(h_n.shape[1], -1)))
        return out



# 数据处理 def processing_data(data_path, split_ratio=0.7):
sentences = []  # 片段
target = []  # 作者
labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}
files = os.listdir(data_path)  # 打开文件获取句子，存储句子及其对应作者编号
for file in files:
    if not os.path.isdir(file) and not file[0] == '.':
        f = open(data_path + "/" + file, 'r', encoding='UTF-8');
        for index, line in enumerate(f.readlines()):
            sentences.append(line)  # sentences中存储句子
            target.append(labels[file[:-4]])  # target中存储的是枚举数字01234即为句字作者
mydata = list(zip(sentences, target))  # 将文件中的数据封装到mydata中 mydata是由tuple组成的列表形式[(鲁迅句xx,0),(鲁迅句xx,0),...]
TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x), lower=True,use_vocab=True)  # sequential表示是否顺序.Field对象指定要如何处理某个字段；fields可简单理解为每一列数据和Field对象的绑定关系
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))  # 一个列表，为Dataset所需要，Example为对数据集中一条数据的抽象
dataset = Dataset(examples, fields=FIELDS)  # Dataset定义数据源信息，是用于构建词表的数据集，下面使用
TEXT.build_vocab(dataset,vectors='glove.6B.100d')  # 构建词表：给每个单词编码，也就是用数字来表示每个单词，这样才能够传入模型 dataset为用于构建词表的数据集
train, val = dataset.split(split_ratio=split_ratio)
train_iter, val_iter = BucketIterator.splits(  # BucketIterator将类似长度的样本处理成一个batch，有利于训练。这里同时对训练集和验证集进行迭代器构建
    (train, val),
    batch_sizes=(16, 16),
    device=DEVICE,  # 如果使用gpu，此处将-1更换为GPU的编号
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    repeat=False
)
Text_vocab = TEXT.vocab
vocab_size = len(Text_vocab)
# or
len_vocab = len(Text_vocab)


save_model_path = "results/model.pth"  # 保存模型路径和名称
train_val_split = 0.7




args = Namespace(
    num_vocab=len(Text_vocab),
    embedding_dim=5,
    padding_index=0,
    rnn_hidden_size=5,
    rnn_bidirection=True,
    rnn_layer=2,
    num_classes=5,
    drop_rate=0.5,
)
class TextRCNN(nn.Module):
    def __init__(self):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding(args.num_vocab, args.embedding_dim, padding_idx=args.padding_index)
        if args.rnn_bidirection:
            self.rnn = nn.LSTM(args.embedding_dim, args.rnn_hidden_size, bidirectional=True, num_layers=args.rnn_layer, batch_first=True, dropout=args.drop_rate)
            self.fc = nn.Linear(2*args.rnn_hidden_size+args.embedding_dim, args.num_classes)
        else:
            self.rnn = nn.LSTM(args.embedding_dim, args.rnn_hidden_size, bidirectional=False, num_layers=args.rnn_layer, batch_first=True, dropout=args.drop_rate)
            self.fc = nn.Linear(args.rnn_hidden_size+args.embedding_dim, args.num_classes)

    def forward(self, x_in):
        emb = self.embedding(x_in)
        out, _ = self.rnn(emb)
        out = torch.cat([emb, out], dim=-1)
        out = F.relu(out).transpose(1, 2)
        out = F.max_pool1d(out, out.size(-1)).squeeze(-1)
        out = self.fc(out)
        return out


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 5)
        self.dropout = nn.Dropout(0.5)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)
        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)
        logit = self.fc(attn_output)
        return logit


EMBEDDING_DIM = 100   #词向量维度
LEARNING_RATE = 1e-3


model = BiLSTM_Attention(len_vocab, EMBEDDING_DIM, hidden_dim=64, n_layers=2)

# model = TextRCNN()
pretrained_embedding = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embedding)

model.to(DEVICE)







optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

criteon = nn.BCEWithLogitsLoss()





# 迭代训练，每个epoch里还对验证集进行预测
for epoch in range(1):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0

    for idx, batch in enumerate(train_iter):
        text, label = batch.text, batch.category

        optimizer.zero_grad()

        out = model(text)

        loss = loss_fn(out, label)
        #loss = criteon(out, label)


        loss.backward(retain_graph=True)  # 反向传播
        optimizer.step()  # 优化器

        accracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())

        # 计算每个样本的acc和loss之和
        train_acc += accracy * len(batch)
        train_loss += loss.item() * len(batch)

        # 原有代码 print("\r epoch:{} loss:{}, train_acc:{}".format(epoch,loss.item(),accracy),end=" ")

    # 每个epoch在验证集上进行预测
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text, label = batch.text, batch.category
            out = model(text)
            loss = loss_fn(out, label.long())
            accracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
            # 计算一个batch内每个样本的acc和loss之和
            val_acc += accracy * len(batch)
            val_loss += loss.item() * len(batch)

    # 在每个epoch中还需要进行这些计算 四个量，每次epoch要输出一次，所以在这里格式化输出
    train_acc /= len(train_iter.dataset)
    train_loss /= len(train_iter.dataset)
    val_acc /= len(val_iter.dataset)
    val_loss /= len(val_iter.dataset)

    print('{{"metric": "train_acc", "value": {}}}'.format(train_acc))
    print('{{"metric": "train_loss", "value": {}}}'.format(train_loss))
    print('{{"metric": "val_acc", "value": {}}}'.format(val_acc))
    print('{{"metric": "val_loss", "value": {}}}'.format(val_loss))

# 保存模型
torch.save(model.state_dict(), 'results/temp.pth')




