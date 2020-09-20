from gensim.models import Word2Vec as w2v
from konlpy.tag import Okt
import gensim
import pandas as pd
import numpy as np

'''
데이터 불러오기 및 결측치 확인
'''
fixed_data_df = pd.read_csv('./data/original_data.csv')
fixed_data_df.isnull().sum()
fixed_data_df.shape

'''
공백 제거
'''
fixed_data_df['document'] = fixed_data_df['document'].str.strip()

'''
정규 표현식 사용
'''
fixed_data_df['document'] = fixed_data_df['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
fixed_data_df['document'].replace('', np.nan, inplace = True) # 결측치가 아닌 공백을 결측치로 치환

'''
결측치 제거
'''
# fixed_data_df.loc[fixed_data_Df.document.isnull()][:5] # 결측치 출력
fixed_data_df = fixed_data_df.dropna(axis = 0)
fixed_data_df.isnull().sum()
fixed_data_df.shape

'''
데이터 저장, 198,884개 문장
'''
fixed_data_df.to_csv('./data/fixed_data.csv')

'''
단어 토큰화
'''
okt = Okt()
original_data_df = pd.read_csv('./data/fixed_data.csv')
test_data_df = original_data_df

# 시간 오래 걸림
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다'] # 불용어
tokenized_data = []
i = 0
for text in test_data_df['document']:    
    tokenized_text = okt.morphs(text, stem = True)
    tokenized_text = [text for word in tokenized_text if not word in stopwords]
    tokenized_data.append(tokenized_text)
    i += 1
    print('{}/{}'.format(i, len(test_data_df)))

'''
w2v 모델 학습, 모델 저장
'''
model_new = w2v(tokenized_data, size = 100, window = 4, min_count = 0, workers = 4, iter = 100, sg = 1)
model_new.save('./data/model_new.bin')

'''
모델 불러오기
'''
model = w2v.load('./data/model_new.bin')

'''
단어 벡터화
'''
vector_np = model.wv
vocabs = vector_np.vocab.keys()
vector_list = [vector_np[v] for v in vocabs]

# pre-trained 모델 단어 수
print(len(list(model.wv.vocab.keys())))

# 단어 출력
print(list(model.wv.vocab.keys())[0])

# 단어 벡터 출력
len(vector_list)
