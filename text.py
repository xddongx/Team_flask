import pandas as pd
import matplotlib.pyplot as plt
import re                              # 정규표현식을 지원한다
from konlpy.tag import Okt             # 한국어 처리 패키지
from tensorflow.keras.preprocessing.text import Tokenizer  # 토큰화(나눠준다고생각하자)
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences # 샘플의 길이 동일하게(패딩), 길이가 다른 경우 0을 넣어서 맞춰준다
import os
from keras import backend as K
import tensorflow as tf

K.clear_session()
tf.reset_default_graph()

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 초기화할 GPU number

def model_text(feel=None):

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)

    data =pd.read_excel("./static/csv/wordtrainfinal.xlsx")
    # 행 무작위로 순서바꾸자.
    data=data.sample(frac=1)  # >> 모든행을 임의의 순서로 반환한다.
    data[data.label == 1]

    test_data=data.iloc[:200,:]

    train_data = data.iloc[200:,:]

    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    # >> 한글과 공백을 제외하고 모두 제거한다는 뜻

    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','을',
                 '를','으로','자','에','와','한','하다','로','이다']

    # >> 토큰화를 위한 형태소 분석기는 KoNLPy 의 Okt 를 사용한다
    # >> KoNLPy : 띄어쓰기,알고리즘 정규화를 이용해서 맞춤법 틀린 문장 어느정도 고쳐주면서 형태소 분석과 품사를 태깅

    okt = Okt() # KoNLPy 에서 제공하는 형태소 분석기이다.(영어는 띄어쓰기 기준으로 토큰화하지만 한국어는 주로 형태소로 나눈다)

    X_train = []
    for sentence in train_data['document']:
        temp_X = []
        temp_X = okt.morphs(sentence, stem=True, norm=True) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_train.append(temp_X)

    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    X_test=[]
    for sentence in test_data['document']:
        temp_X = []
        temp_X = okt.morphs(sentence, stem=True, norm=True)
        temp_X = [word for word in temp_X if not word in stopwords]
        X_test.append(temp_X)


    max_words = 35000
    tokenizer = Tokenizer(num_words=max_words) # 상위 35000개 단어만 보존

    tokenizer.fit_on_texts(X_train) # 단어 인덱스를 구축

    X_train = tokenizer.texts_to_sequences(X_train) # 문자열을 정수 인덱스의 리스트로 변환
    X_test = tokenizer.texts_to_sequences(X_test)
    #print(X_train) # 단어 대신 단어에 대한 인덱스 부여


    print('글자 최대 길이 :',max(len(l) for l in X_train))

    # 모델이 처리할 수 있도록 X_train, X_test 의 모든 샘플의 길이를 동일하게 하자


    max_len =8 # 길이를 10으로 정했다
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    # print(X_train) # 패딩 된것을 확인할수있다( 없는 부분은 0으로 채움)

    y_train = np.array(train_data['label'])
    # print(y_train)
    y_test = np.array(test_data['label'])

    from tensorflow.keras.layers import Embedding, Dense, LSTM
    from tensorflow.keras.models import Sequential


    model3 =Sequential()
    model3.add(Embedding(max_words,30))
    model3.add(LSTM(40))
    model3.add(Dense(1))
    model3.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

    model3.fit(X_train, y_train, epochs=13, batch_size=30, verbose=1,validation_split=0.2)

    print(model3.evaluate(X_test, y_test))

    TRAINED_CLASSIFIER_PATH = "face/model/dual_encoder_lstm_classifier.h5"  # 모델이름

    model3.save(TRAINED_CLASSIFIER_PATH)

    #1단계 : 기분을 대입



    # 2단계 : 대입한 기분 전처리( 토큰화, 벡터화 등)
    def Preprocess(feel):
        X = okt.morphs(feel, stem=True, norm=True)
        X = [word for word in X if not word in stopwords]

        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=max_len)
        return X  # X는 예측할 데이터 전처리한 결과

    #3단계 : 모델에 대입
    print('Rnrmnralaksdflka')
    predict = model3.predict(Preprocess(feel))   # 전처리한 결과 모델에 넣었다
    # print(predict)  # 예측결과
    print('두번째')
    # https://codepractice.tistory.com/71

    def Delete(predict):
        text_lists = []
        predict = predict.reshape(-1).astype('int')
        predict = np.around(predict)
        from collections import Counter  # 최빈값구하려고 부르는 매소드
        list = []
        for i in range(len(predict)):
            if predict[i] != 2 and predict[i] != 5:  # 2나 5는 제거
                a = list.append(predict[i])

        if Counter(list).most_common(1)[0][0] == 0:
            df = '당신의 기분은 화가난것으로 추정됩니다'
            emotion = '화난'
            text_lists.append(df)
            text_lists.append(emotion)
        if Counter(list).most_common(1)[0][0] == 1:
            df = '당신의 기분은 행복한것으로 추정됩니다'
            emotion = '행복'
            text_lists.append(df)
            text_lists.append(emotion)
        if Counter(list).most_common(1)[0][0] == 3:
            df = '당신의 기분은 슬픈것으로 추정됩니다'
            emotion = '슬픈'
            text_lists.append(df)
            text_lists.append(emotion)
        if Counter(list).most_common(1)[0][0] == 4:
            df = '당신의 기분은 불안한것으로 추정됩니다'
            emotion = '불안한'
            text_lists.append(df)
            text_lists.append(emotion)
        print(text_lists)
        return text_lists

    return Delete(predict)



# feel = '행복하다'
# #
# print(model_text(feel))