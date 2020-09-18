'''
정수 인코딩
zero padding
https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
'''

'''
데이터 분리
'''
from konlpy.tag import Okt

okt = Okt()

text_data = ['안녕?', '나는 현규라고 해', '만나서 반가워', '너 진짜 싫다', '아 기분나빠.']
label_data = [1, 1, 1, 0, 0,]

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

'''
토큰화
'''
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenized_data = []
i = 0
for text in text_data:    
    tokenized_text = okt.morphs(text, stem = True)
    tokenized_text = [word for word in tokenized_text if not word in stopwords] # 불용어 제거
    tokenized_data.append(tokenized_text)
    i += 1
    print('{}/{}'.format(i, len(text_data)))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_data)
tokenizer.word_index

vocab_size = len(tokenizer.word_index)
vocab_size

tokenizer = Tokenizer(vocab_size, oov_token= 'OOV')
tokenizer.fit_on_texts(tokenized_data)
tokenized_sequence = tokenizer.texts_to_sequences(tokenized_data)
tokenized_sequence

'''
패딩
'''
import matplotlib.pyplot as plt

print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_sequence))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_sequence))/len(tokenized_sequence))
plt.hist([len(s) for s in tokenized_sequence], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 5
below_threshold_len(max_len, tokenized_sequence)

tokenized_sequence = pad_sequences(tokenized_sequence, maxlen = max_len)
tokenized_sequence
