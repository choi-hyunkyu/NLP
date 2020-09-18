# gensim 설치 ko.bin 다운로드

from gensim.models import Word2Vec as w2v
from konlpy.tag import Okt
import gensim
import pandas as pd

'''
데이터 불러오기 및 결측치 제거
'''
fixed_data_df = pd.read_csv('./data/original_data.csv')
fixed_data_df.isnull().sum()
fixed_data_df.shape

fixed_data_df = fixed_data_df.dropna(axis = 0)
fixed_data_df.isnull().sum()
fixed_data_df.shape
fixed_data_df.to_csv('./data/fixed_data.csv')

'''
단어 토큰화
'''
okt = Okt()
original_data_df = pd.read_csv('./data/fixed_data.csv')
test_data_df = original_data_df

tokenized_data = []
i = 0
for text in test_data_df['document']:    
    tokenized_text = okt.morphs(text)
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