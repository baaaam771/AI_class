from tensorflow.keras.preprocessing.text import Tokenizer
from konlpy.tag import Twitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 단어들이 나오면 통계를 내는 과정을 진행
# 단어의 갯수
total_count = len(tokenizer.word_index)
# 등장 빈도 수가 너무 적은 단어 갯수
rare_count = 0
# 훈련 데이터에서 전체 단어 빈도 수 합
total_seq = 0
# rare 빈도 수 합(적게 나오는 것은 제외하려고)
rare_seq = 0

#  단어, 빈도수
for k, v in tokenizer.word_counts.items():
    total_seq += v
# 3번 이하면 포함하지 않기
    if v < 3:
        rare_count += 1
        rare_seq += v
