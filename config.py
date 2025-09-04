VOCAB_SIZE = None # The vocabulary size of Word2Vec, None for no limit.
MIN_COUNT = 3 # Ignores all words with total frequency lower than this.
EMBEDDING_SIZE = 128 # Embedding size of the word vectors.
RATIO = "3:1:1" # The ratio for spliting dataset into training, validation, and testing respectively.
HIDDEN_DIM = 100 # The hidden dimension of the ST-Tree encoder.
ENCODE_DIM = 128 # The hidden dimension of the BiGRU encoder. BiGRU 编码器的隐藏层维度大小为 128。
LABELS = 104 # The number of the classes for the output.
EPOCHS = 15 # The number of epochs for training. 训练轮数
BATCH_SIZE = 64 # 每个训练批次的样本数量
USE_GPU = True # 使用GPU进行训练
# congif.py文件可能包含模型参数，数据处理参数，训练参数等内容