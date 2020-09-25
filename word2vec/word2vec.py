from ckonlpy.tag import Twitter, Postprocessor
from gensim.models import Word2Vec as w2v
import pandas as pd
import numpy as np
import torch

# 데이터 불러오기
with open('./data/KCC940_Korean_sentences_UTF8.txt', 'r') as text_file:
    sentence_list = text_file.readlines()
print('문장개수: ', len(sentence_list), '문장'), print('파일크기: ', 954231, 'KB')

sentence_df = pd.DataFrame({'sentence': sentence_list})
sentence_df.head()

# 데이터프레임 각 행마다 앞 뒤 공백 제거
sentence_df['sentence'] = sentence_df['sentence'].str.strip()

# 정규 표현식 사용
sentence_df['sentence'] = sentence_df['sentence'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
sentence_df.head()

# 결측치가 아닌 공백을 결측치로 치환
sentence_df['sentence'].replace('', np.nan, inplace = True)
sentence_df.head()

# ckonlpy Postprocessor value 정의, Twitter() 정의
passtags = {'Noun'}
twitter = Twitter()

# 불용어 정의
stopwords = {
    '의','가','이','은','들','는','좀','잘','걍','과', '로', '을'
    '도','를','으로','자','에','와','한','하다', '어요'
}


postprocessor = Postprocessor(
    base_tagger = twitter, # base tagger
    stopwords = stopwords, # 해당 단어 필터
    #passwords = passwords, # 해당 단어만 선택
    passtags = passtags, # 해당 품사만 선택
    #replace = replace, # 해당 단어 set 치환
    #ngrams = ngrams # 해당 복합 단어 set을 한 단어로 결합
)


# 수정된 데이터프레임 리스트 변환
input_sentence_list = sentence_df['sentence'][:10000].to_list()
input_sentence_list


# result = [postprocessor.pos(sentence) for sentence in input_sentence_list]

# 토큰화 진행
tokenized_nouns_list = []
i = 0
for sentence in input_sentence_list:
    tokenized_word = postprocessor.pos(sentence)
    tokenized_sentence_list = []
    for word in tokenized_word:
        tokenized_sentence_list.append(word[0])
    tokenized_nouns_list.append(tokenized_sentence_list)
    i += 1
    print("{}/{}".format(i, len(input_sentence_list)))
tokenized_nouns_list

'''
Word2Vec 학습
'''
model = w2v(tokenized_nouns_list, size=100, window=4, min_count=3, workers=4, sg=0)
model.save('./data/model_new.bin')

'''
벡터리스트 생성
'''
vector_list = [model.wv[v] for v in model.wv.vocab.keys()]
model.wv.vocab.keys()

'''
수정
'''
# 단어사전 value 변경
idx = 0
for i in list(model.wv.vocab):
    model.wv.vocab[i] = vector_list[idx]
    idx += 1
vocab = list(model.wv.vocab)
len(vocab)

'''
각 단어에 대해 매핑
'''
model.wv.vocab['OOV'] = np.random.randn(100)
encoded = []
for word in vocab:
    temp = []
    for w in word:
        try:
            temp.append(model.wv.vocab[w])
        except KeyError:
            temp.append(model.wv.vocab['OOV'])
    encoded.append(temp)

model.wv.vocab
