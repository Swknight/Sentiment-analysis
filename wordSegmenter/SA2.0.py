import jieba
import jieba.analyse
from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary
import  re
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding,Dropout,Activation
from keras.layers import LSTM


def getText():
    # 获取文本输入
    contents = []
    words = []
    evalutation = []
    with codecs.open('train_content.csv', encoding='utf-8', mode='rU') as f:
        for line in f:
            line1 = re.sub("[\s+\.\!\/_,\"\'$%^*()?;；:-]+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)
            contents.append(line1)
    print('len of contents  :', len(contents))
    with codecs.open('train_word.csv', encoding='utf-8', mode='rU') as f:
        for line in f:
            words.append(line)
    print('len of words : ', len(words))
    with codecs.open('evaluation set.csv', encoding='utf-8', mode='rU') as f:
        for line in f:
            line1 = re.sub("[\s+\.\!\/_,\"\'$%^*()?;；:-]+|[+——！，;:。？、~@#￥%……&*（）]+", "",line)
            evalutation.append(line1)
    return contents, words, evalutation

def calWordCover(contents, words, evaluation,k=10):
    keysLen = 0
    coverLen = 0
    print('关键词 / 被覆盖的关键词')
    for i in range(20000):
        tags = jieba.lcut(contents[i].strip(), cut_all=False)
        keys = set([i for i in words[i].strip().split(';') if len(i) > 0])  # 去掉换行符及空的关键词
        evaluates = jieba.lcut(evaluation[i].strip(), cut_all=False)
        keysLen += len(keys)
        coverLen += len(keys & set(tags))
        if i < 5:
            print(keys, keys & set(tags))
        contents[i] = ' '.join(list(tags))
        evaluation[i] = ' '.join(list(evaluates))
        words[i] = list(keys)
    print('覆盖率 : ', coverLen / keysLen)
    return contents, words, evaluation

contents, words, evaluation = getText()
contents, words, evaluation = calWordCover(contents, words, evaluation, 15)

# 创建词语字典，并返回word2vec模型中词语的索引，词向量
def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec

sentences = [s.split() for s in evaluation+contents]
model = word2vec.Word2Vec(sentences,
                 size=100,  # 词向量维度
                 min_count=5,  # 词频阈值
                 window=5)  # 窗口大小

# model.save('word2vec.model')  # 保存模型

# 索引字典、词向量字典
index_dict, word_vectors= create_dictionaries(model)


# output = open("word2vec_pkl.pkl", 'wb')
# pickle.dump(index_dict, output)  # 索引字典
# pickle.dump(word_vectors, output)  # 词向量字典
# output.close()


#train

vocab_dim = 100  # 向量维度
maxlen = 140  # 文本保留的最大长度
batch_size = 32
n_epoch = 5
input_length = 140


def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)

    return np.array(new_sentences)

n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
embedding_weights = np.zeros((n_symbols, 100))  # 创建一个n_symbols * 100的0矩阵
for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
    embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(allsentences, labels, test_size=0.2)         # 数据，标签

# 转为数字索引形式
X_train = text_to_index_array(index_dict, X_train_l)
X_test = text_to_index_array(index_dict, X_test_l)
print (u"训练集shape： ", X_train.shape)
print (u"测试集shape： ", X_test.shape)

y_train = np.array(y_train_l)  # 转numpy数组
y_test = np.array(y_test_l)

# 将句子截取相同的长度maxlen，不够的补0
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

def train_lstm(p_n_symbols, p_embedding_weights, p_X_train, p_y_train, p_X_test, p_y_test):
    print (u'创建模型...')
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=p_n_symbols,
                        mask_zero=True,
                        weights=[p_embedding_weights],
                        input_length=input_length))

    model.add(LSTM(output_dim=50,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print (u'编译模型...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print (u"训练...")
    model.fit(p_X_train, p_y_train, batch_size=batch_size, nb_epoch=n_epoch,
              validation_data=(p_X_test, p_y_test))

    print (u"评估...")
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    print ('Test score:', score)
    print ('Test accuracy:', acc)