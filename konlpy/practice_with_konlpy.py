from konlpy.tag import Komoran, Hannanum, Okt, Kkma
import pandas as pd

# 토크나이저 정의
komoran = Komoran()
hannanum = Hannanum()
okt = Okt()
kkma = Kkma()

# 문장 생성
text = "4차산업혁명에서 인공지능기술은 매우 중요한 부분을 차지하고 있으며, 인공지능과 딥러닝 기술을 개발하기위해서는 컴퓨터 언어로 파이썬뿐만 아니라 C언어 또는 자바 코딩을 잘 해야 한다."

# komoran tokenizing
komoran_pos_result = komoran.pos(text)
komoran_morphs_result = komoran.morphs(text)
komoran_nouns_result = komoran.nouns(text)

# hannanum tokenizing
hannanum_pos_result = hannanum.pos(text)
hannanum_morphs_result = hannanum.morphs(text)
hannanum_nouns_result = hannanum.nouns(text)

# okt tokenizing
okt_pos_result = okt.pos(text)
okt_morphs_result = okt.morphs(text)
okt_nouns_result = okt.nouns(text)

# kkma tokenizing
kkma_pos_result = kkma.pos(text)
kkma_morphs_result = kkma.morphs(text)
kkma_nouns_result = kkma.nouns(text)

# komoran 데이터프레임 저장
komoran_result = pd.DataFrame(
    {
        "komoran_pos_result": [komoran_pos_result],
        "komoran_morphs_result": [komoran_morphs_result],
        "komoran_nouns_result": [komoran_nouns_result],
        }
)
komoran_result.to_csv('./data/komoran_result.csv')
print(komoran_result.head())

# hannanum 데이터프레임 저장
hannanum_result = pd.DataFrame(
    {
        "hannanum_pos_result": [hannanum_pos_result],
        "hannanum_morphs_result": [hannanum_morphs_result],
        "hannanum_nouns_result": [hannanum_nouns_result],
        }
)
hannanum_result.to_csv('./data/hannanum_result.csv')
print(hannanum_result.head())

# okt 데이터프레임 저장
okt_result = pd.DataFrame(
    {
        "okt_pos_result": [okt_pos_result],
        "okt_morphs_result": [okt_morphs_result],
        "okt_nouns_result": [okt_nouns_result],
        }
)
okt_result.to_csv('./data/okt_result.csv')
print(okt_result.head())

# kkma 데이터프레임 저장
kkma_result = pd.DataFrame(
    {
        "kkma_pos_result": [kkma_pos_result],
        "kkma_morphs_result": [kkma_morphs_result],
        "kkma_nouns_result": [kkma_nouns_result]
        }
)
kkma_result.to_csv('./data/kkma_result.csv')
print(kkma_result.head())

# 데이터프레임 일괄 저장
result = pd.DataFrame(
    {
        "komoran_pos_result": [komoran_pos_result],
        "komoran_morphs_result": [komoran_morphs_result],
        "komoran_nouns_result": [komoran_nouns_result],
        
        "hannanum_pos_result": [hannanum_pos_result],
        "hannanum_morphs_result": [hannanum_morphs_result],
        "hannanum_nouns_result": [hannanum_nouns_result],
        
        "okt_pos_result": [okt_pos_result],
        "okt_morphs_result": [okt_morphs_result],
        "okt_nouns_result": [okt_nouns_result],
        
        "kkma_pos_result": [kkma_pos_result],
        "kkma_morphs_result": [kkma_morphs_result],
        "kkma_nouns_result": [kkma_nouns_result]
        }
)
result.to_csv('./data/result.csv')
