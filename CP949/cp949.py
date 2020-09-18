# cp949 추가된 8,822자 출력
import pandas as pd
import numpy as np

cp_df = pd.read_csv('./encoding/CP949.csv')
cp_list = np.array(cp_df['CP949_text']).tolist()

encoded_word_list = []
i = 0
for word in cp_list:
    encoded_word = word.encode('cp949')
    encoded_word_list.append(encoded_word)
    i += 1
    print("Number: {}/{} | Encoded Word: {} | Appended Word: {}".format(
        i, len(cp_list),
        encoded_word,
        encoded_word.decode('cp949'))
         )