# https://github.com/EliasCai/sentiment 代码持续更新，欢迎大家关注，希望有所帮助，共同提升
# 个人介绍：工作从事大数据开发，熟悉机器学习和深度学习的使用
# 比赛经验：曾参加场景分类（AiChallenger）、口碑商家客流量预测（天池）、用户贷款风险预测（DataCastle）、
#           摩拜算法天战赛（biendata）等，寻找队友冲击前排，希望不吝收留！
# 版本：v1.1
# 环境：python3; tensorflow-1.0.0; keras-2.0.6
# 邮箱：elias8888#qq.com
# 使用：将data文件夹中的三个csv文件放到py文件同个文件夹下面即可运行
# Finish：
# 使用jieba进行分词，并用LSTM对第一个情感关键词进行预测，10轮epochs后验证样本的准确率为0.70
#
# Todo：
# 1、将情感关键词添加到jieba的字典里
# 2、将第2、3个关键词添加到样本，将预测的概率大于阈值的位置作为情感关键词输出
# 3、完成主题和情感正负面的分析
# 4、完善LSTM的网络
# 5、试试CNN的效果

import jieba
import jieba.analyse
import pandas as pd
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


def getText():
    # 获取文本输入
    contents = []
    words = []
    evalutation = []
    with codecs.open('train_content.csv', encoding='utf-8', mode='rU') as f:
        for line in f:
            contents.append(line)
    print('len of contents  :', len(contents))
    with codecs.open('train_word.csv', encoding='utf-8', mode='rU') as f:
        for line in f:
            words.append(line)
    print('len of words : ', len(words))
    with codecs.open('evaluation set.csv', encoding='utf-8', mode='rU') as f:
        for line in f:
            evalutation.append(line)
    return contents, words, evalutation
# contents, words, evaluation = getText()
# for i in evaluation:
#     print (i)

def calWordCover(contents, words, evaluation,k=10):
    # 计算分词结果对情感关键词的覆盖率
    # params
    # contents : 评论内容
    # words ：情感关键词
    # k : 关键词数量，暂时用不上
    # return
    # contents : 分词后重新组成的评论内容
    # words : 将关键词拆分并组成List
    keysLen = 0
    coverLen = 0
    print('关键词 / 被覆盖的关键词')
    for i in range(20000):
        # tags = set(jieba.analyse.extract_tags(contents[i], topK=k))
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


dfWords = pd.DataFrame(words)
print('平均情感关键词数量 : ', dfWords.count(axis=1).mean())  # 平均情感关键词的数量
# 剔除没有情感关键词的句子
indexWord = dfWords.index[~pd.isnull(dfWords.iloc[:, 0])]  # 第一个关键词非空的index 去掉了~
word = []
delimiter = ','
totalWord=[]
for i in indexWord:
    oneWord = list(filter(None, np.array(dfWords.iloc[[i]]).tolist()[0]))
    word.append(oneWord)
for i in word:
    totalWord.append(' '.join(i))
#print (totalWord)

#word = dfWords.iloc[indexWord,: ]  # 第一个关键词非空的列表

contents = pd.DataFrame(contents).iloc[indexWord, 0].tolist()  # 第一个关键词非空对应的评论内容
#indexEWord = dfWords.index
evaluation = pd.DataFrame(evaluation).iloc[:, 0].tolist()
tokenizer = Tokenizer(num_words=32000)
tokenizer.fit_on_texts(contents + totalWord + evaluation)
sequences = tokenizer.texts_to_sequences(contents)
sequences_words = tokenizer.texts_to_sequences(totalWord)
#print (sequences_words)
sequences_evaluation = tokenizer.texts_to_sequences(evaluation)
data_x = pad_sequences(sequences, maxlen=50)  # 平均长度是20，最长是200，设置为50
data_e = pad_sequences(sequences_evaluation, maxlen=50)


# print (data_x)
def getDataY(data_x):
    # 获取情感关键词在评论内容中的位置
    data_y = []
    for i in range(data_x.shape[0]):
        a = []
        for j in sequences_words[i]:
            try:
                a.append(list(data_x[i]).index(j))#只查看第一个关键词的位置?
            except:
                a.append(-1)  # 如果情感的关键词不在分词里面
        data_y.append(a)
    return data_y

data_y = getDataY(data_x)
data_yy = []
for i in data_y:
    a = np.array(i)
    onehot_y = to_categorical(a[a >= 0], num_classes=50)  # 将位置的信息转化为OneHot
    if len(onehot_y)==0:
        onehot_y = np.zeros(50).tolist()
       # print (onehot_y)
    df1 = pd.DataFrame(onehot_y)
    onehot_y = np.array(df1.apply(lambda  x: x.sum())).tolist()
    if len(onehot_y)==1:                    # 结构冗余，不管，以后再说
        onehot_y = np.zeros(50).tolist()
    data_yy.append(onehot_y)

data_yy = np.array(data_yy)
data_y = np.array(data_y)
judge = []
for row in data_y:
    if sum(row)==0:
        judge.append(False)
    else:
        judge.append(True)
judge = np.array(judge)

train_x, test_x, train_y, test_y = train_test_split(data_x[judge], data_yy[judge])


def trainModel(train_x, test_x, train_y, test_y):
    # 训练模型
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(50, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_x, train_y,
              batch_size=32,
              epochs=20,
              validation_data=(test_x, test_y))
    return model


model = trainModel(train_x, test_x, train_y, test_y)


def getPredWord(model, word_index, x, y):
    # 获取评论内容的预测关键词
    pos_pred = model.predict_classes(x)
    for i in pos_pred:
        print (i)
    # pos_y = np.argmax(y, axis=1)  # 最大值的位置，也就是还原到data_y的形式
    # x[[i for i in range(pos_pred.shape[0])], pos_pred] # 通过预测的位置获取关键词
    # index2word = pd.DataFrame.from_dict(word_index, 'index').reset_index().set_index(0)  # 将word2index转化为index2word
    #
    # print('模型预测的关键词')
    # print(index2word.loc[x[[i for i in range(pos_pred.shape[0])], pos_pred]].head(5))
    # print('实际样本的关键词')
    # print(index2word.loc[x[[i for i in range(pos_y.shape[0])], pos_y]].head(5))


getPredWord(model, tokenizer.word_index, test_x, test_y)