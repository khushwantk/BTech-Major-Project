from hanlp_restful import HanLPClient


key="MTc0MkBiYnMuaGFubHAuY29tOnBrd280N1VOQUIxYmVVNGU="
sentence= "Hello this is a test sentence"

HanLP = HanLPClient('https://www.hanlp.com/api', auth=key, language='mul')

print("Sentence : ", sentence)

print("\n")
HanLP(sentence, tasks=('con'))

print("Tokenisation : ")
HanLP(sentence, tasks='tok')
print("POS Tagging abd Dependency Parse : ")
HanLP(sentence, tasks='ud')['pos']
print("Constituency Parse: ")
HanLP(sentence, tasks='con')
