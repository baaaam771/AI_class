from konlpy.tag import Okt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from konlpy.tag import Twitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

!wget "https://drive.google.com/uc?export=download&id=1ByJN0Vh4ctIwNdnO2jevEcBgrbRHuNyM" - 0 "ratings_train.txt"
!wget "https://drive.google.com/uc?export=download&id=1fNm-8pQJsuDbFaVIMow1DI-7lsnNLRFB" - 0 "ratings_test.txt"

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

train_data.head()
# 별점에 따른 라벨링인것 같다 정답 라벨이 쫌 다르다


twt = Twitter()

x_train = [twt.morphs(sentence) for sentence in train_data['document']]
x_test = [twt.morphs(sentence) for sentence in test_data['document']]

x_train[0]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

threshold = 3
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :', vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

drop_train_idx = [index for index,
                  sentence in enumerate(x_train) if len(sentence) < 1]

x_train = np.delete(x_train, drop_train_idx, axis=0)
y_train = np.delete(y_train, drop_train_idx, axis=0)

len(x_train), len(y_train)


print('리뷰 최대 길이', max(list(map(lambda x: len(x), x_train))))
print('리뷰 평균 길이', sum(map(len, x_train)) / len(x_train))

plt.hist([len(s) for s in x_train], bins=50)
plt.xlabel('length of data')
plt.ylabel('number of data')


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' %
          (max_len, (cnt / len(nested_list))*100))


below_threshold_len(30, x_train)

# pad sequences

x_train = pad_sequences(x_train, maxlen=30)
x_test = pad_sequences(x_test, maxlen=30)


# Embedding (vocab_size, ???, 30), LSTM(??), Dense(?)

model = Sequential([
    Embedding(vocab_size, 128, input_length=30),
    LSTM(16, kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()

# compile
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# fit
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=128, validation_split=0.2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 위의 값 시각화

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# evalute 모델 test
model.evaluate(x_test, y_test)


def sentiment_predict(sentence):

    new_sentence = Okt().morphs(phrase=sentence, stem=True)

    encoded = tokenizer.texts_to_sequences([new_sentence])
    padded = pad_sequences(encoded, maxlen=30)

    score = float(model.predict(padded))

    if (score > 0.5):
        print('{:.2f}% 확률로 긍정'.format(score * 100))
    else:
        print('{:.2f}% 확률로 부정'.format((1-score) * 100))


sentiment_predict('이 영화 완전 꿀잼')

sentiment_predict('쓰레기같은 최악의 영화')
