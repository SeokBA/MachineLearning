from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))

from nltk.tag import pos_tag
x=word_tokenize(text)
pos_tag(x)

from konlpy.tag import Okt
okt=Okt()
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

from konlpy.tag import Kkma
kkma=Kkma()
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))