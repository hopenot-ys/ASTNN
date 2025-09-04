import torch.nn as nn
import torch.nn.functional as F # 导入PyTorch的函数API，提供激活函数，损失函数等
import torch
from torch.autograd import Variable


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):  # init 初始化函数
        super(BatchTreeEncoder, self).__init__()  # 调用父类的初始化函数
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        # 创建一个嵌入层，将词汇表中的每个词映射到一个embedding_dim 维的向量
        self.encode_dim = encode_dim
        # 保存编码维度
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        # 定义一个线性层，用于将词向量从embedding_dim转化到encode_dim
        self.W_l = nn.Linear(encode_dim, encode_dim)
        # 定义另一个线性层，输入和输出都是 encode_dim 维度，通常用于处理左子树信息。
        self.W_r = nn.Linear(encode_dim, encode_dim)  # 右子树
        self.activation = F.relu  # 激活函数
        self.stop = -1  # 用于树结构遍历时判断节点是否结束
        self.batch_size = batch_size  # 保存批次大小
        self.use_gpu = use_gpu  #保存
        self.node_list = []

        self.th = torch.cuda if use_gpu else torch
        # 这行代码常见于需要同时兼容 GPU 与 CPU 的 PyTorch 代码中

        self.batch_node = None  # 初始化 batch_node，后续会用于存储批次节点的张量。

        # pretrained  embedding
        if pretrained_weight is not None: # 如果提供了预训练的词向量权重
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        # self 就是“当前这个对象”，让方法能操作属于自己的数据。
        if self.use_gpu: # use_gpu为True,则使用GPU
            return tensor.cuda() # 如果使用 GPU，则将张量转移到 GPU 上（加速计算），并返回。
        return tensor # 如果不使用 GPU，则直接返回原始张量（在 CPU 上）。
    
    # 函数作用：根据是否使用 GPU，把输入的张量自动放到合适的设备（GPU 或 CPU）上，方便后续进行高效计算。

    def traverse_mul(self, node, batch_index):
        # 批量树遍历
        size = len(node) # 多少个并行节点
        if not size: # NULL
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
        # 为当前层的节点初始化一个全零张量，形状为 (size, encode_dim)
        
        index, children_index = [], []
        current_node, children = [], []  # 列表

        for i in range(size):
            if node[i][0] is not -1:
                index.append(i) # 记录有效父节点的位置
                current_node.append(node[i][0])
                temp = node[i][1:]  # 取出该父节点的所有孩子列表
                c_num = len(temp)  # 孩子个数
                for j in range(c_num):
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:  # 没有为第J个孩子创建容器
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

            # 经过以上： index(当前层所有有效父节点的位置) current_node(当前层所有有效父节点的sambol id,与index一一对应)
            # children_index(每个孩子对应的父节点在当前层的位置) children(每个孩子的节点信息,与children_index一一对应)



        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current # 返回这一批节点的编码表示

    def forward(self, x, bs):  # 模型的前向传播函数，x是输入数据，bs是batch size
        # x是批量的树结构数据
        self.batch_size = bs # 在运行时动态地设置batch size
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        # (batch_size, encode_dim) 全零张量,用于存储每个样本的节点编码
        # torch.zeros 创建一个指定形状的全零张量
        # 新版 PyTorch 中，张量本身就支持自动求导，直接用 tensor 即可，Variable 已经合并进 tensor。
        self.node_list = []  # 收集若干个（bs,encode_dim）张量，代表树不同位置的编码结果
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BatchProgramClassifier(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramClassifier, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        #class "BatchTreeEncoder"
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            seq.append(encodes[start:end])
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)
        encodes = nn.utils.rnn.pack_padded_sequence(encodes, torch.LongTensor(lens), True, False)

        # gru
        gru_out, _ = self.bigru(encodes, self.hidden)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, padding_value=-1e9)

        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]

        # linear
        y = self.hidden2label(gru_out)
        return y

