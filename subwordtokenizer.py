# -*- coding: utf-8 -*-
"""subwordTokenizer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OfNoWUSYAwk5NVPW-k7eVAMS43aX2UVr
"""

import tensorflow_datasets as tfds
import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

train_df =pd.read_csv('IMDb_Reviews.csv')

train_df.head()

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(train_df['review'], target_vocab_size=3000)

print(tokenizer.subwords[:100])

sample =train_df['review'][0]

tokenized =tokenizer.encode(sample)
# 정수 인코딩
tokenized

# 문자 디코딩
original =tokenizer.decode(tokenized)
original

for ts in tokenized:
  print('{}---->{}'.format(ts,tokenizer.decode([ts])))

sample2="My family and I normally do not watch local movies for the simple reasonxyz that they are poorly made"

tokenized =tokenizer.encode(sample2)
# 정수 인코딩
tokenized

# 문자 디코딩
original =tokenizer.decode(tokenized)
original

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")

train_data =pd.read_table('ratings_train.txt')

train_data.head()

train_data.isnull().sum()

train_data =train_data.dropna(how= 'any')

train_data.isnull().sum()

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(train_data['document'], target_vocab_size=3000)

print(tokenizer.subwords[:100])

k_sample =train_data['document'][21]

k_sample

tokenized =tokenizer.encode(k_sample)
# 정수 인코딩
tokenized

# 문자 디코딩
original =tokenizer.decode(tokenized)
original

m_sample="보면서 웃지 않는 건 불가능하다 ㅋㅋㅋ"

tokenized =tokenizer.encode(m_sample)
# 정수 인코딩
tokenized

original =tokenizer.decode(tokenized)
original

for ts in tokenized:
  print('{}---->{}'.format(ts,tokenizer.decode([ts])))

!pip install sentencepiece

import sentencepiece as spm

with open('imdb_review.txt', 'w', encoding='utf-8') as f:
  f.write('\n'.join(train_df['review']))

spm.SentencePieceTrainer.Train(
    '--input=imdb_review.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=1000'
)

import csv
vocab_list =pd.read_csv('imdb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)

vocab_list.sample(10)

sp =spm.SentencePieceProcessor()
vocab_file= 'imdb.model'
sp.load(vocab_file)

lines = [
  "I didn't at all think of it this way.",
  "I have waited a long time for someone to film"
]

for l in lines:
  print(l)
  print(sp.encode_as_pieces(l))
  print(sp.encode_as_ids(l))
  print()

sp.GetPieceSize()

sp.IdToPiece(430)

sp.PieceToId('_off')

sp.PieceToId('off')

sp.PieceToId('character')

sample =  'I have waited a long time for someone to film'

sp.encode(sample ,out_type=str)

sp.encode(sample, out_type=int)

with open('naver_review.txt', 'w', encoding='utf8') as f:
  f.write('\n'.join(train_data['document']))

spm.SentencePieceTrainer.Train(
     '--input=naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999'
)

vocab_list =pd.read_csv('naver.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)

vocab_list[:10]

sp =spm.SentencePieceProcessor()
vocab_file ='naver.model'
sp.load(vocab_file)

lines =[
        "뭐 이딴 것도 영화냐.",
        "진짜 최고의 영화입니다!! ㅋㅋㅋ"
]
for l in lines:
  print(l)
  print(sp.encode_as_pieces(l))
  print(sp.encode_as_ids(l))
  print()

