
class Net_bilstm_attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):  # 参数需要补全
        super(Net_bilstm_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def attention_net(self, x, query, mask=None):  # 软性注意力机制（key=value=x）
        d_k = query.size(-1)  # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分机制  scores:[batch, seq_len, seq_len]
        p_attn = f.softmax(scores, dim=-1)  # 对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)  # 对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))  # [seq_len, batch, embedding_dim]
        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]
        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)  # 和LSTM的不同就在于这一句
        logit = self.fc(attn_output)
        return logit




class Net_lstm(nn.Module):
    def __init__(self, vocab_size):
        super(Net_lstm, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)
        self.decoder = nn.Linear(128, 2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]
        final = lstm_out[-1]
        y = self.decoder(final)
        return y



# 这个不能用
# 2020-11-24最简单一个lstm例子
class Enet(nn.Module):
    def __init__(self):
        super(Enet, self).__init__()
        self.embedding = nn.Embedding(len_vocab, 100)
        self.lstm = nn.LSTM(100, 128, 3, batch_first=True)  # ,bidirectional=True)
        self.linear = nn.Linear(128, 5)

    def forward(self, x):
        batch_size, seq_num = x.shape
        vec = self.embedding(x)
        out, (hn, cn) = self.lstm(vec)
        out = self.linear(out[:, -1, :])
        out = f.softmax(out, -1)
        return out

# model = Net_origin()
model = Enet()

# 对应最简单1124案例
model.embedding.weight.data.copy_(Text_vocab.vectors)
model.to(DEVICE)

