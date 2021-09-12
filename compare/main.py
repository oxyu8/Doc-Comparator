# import MeCab
import numpy
import janome
import sklearn
import plotly

from janome.tokenizer import Tokenizer
import gensim

# m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

# print(
#     m.parse(
#         "遺伝子組み換え食品の多くは、植物が害虫に食べられないように、害虫の嫌いな成分を作る遺伝子を導入するものなので、作物を人間が食べても健康に影響はありません。しかし、その作物を重荷食害していた害虫がその作物を食べられなくなることによって、生態系に影響があるのではないかと心配されています。"
#     ))

snippets = [
    '遺伝子組換え技術とは、ある生物が持つ遺伝子（DNA）の一部を、他の生物の細胞に導入して、その遺伝子を発現（遺伝子の情報をもとにしてタンパク質が合成されること）させる技術のことです。 遺伝子とは何か、遺伝子組換えとは何か、詳しくは「 遺伝子組換えとは（PDF：1,347KB） 」をご覧ください。',
    '遺伝子組換え表示制度に関する食品表示基準の一部を改正する内閣府令の公布について [PDF:180KB] 食品表示基準の一部を改正する内閣府令新旧対照条文 [PDF:100KB] 「食品表示基準について(平成27年3月30日消食表第139号)」',
    "遺伝子組み換え技術と品種改良は違うの？ 従来の品種改良は、生物学的に似通った植物の間で、人工的に交配させ、新しい品種を作り出すものでした。 複数の植物の遺伝情報を混ぜることで、その中で突然変異などが発生することもありましたが、種の壁を越えることはありませんでした。",
    '厚生労働省は、平成１３年４月から遺伝子組換え食品の安全性審査を食品衛生法上の義務としています。',
    '遺伝子組み換え作物とは 遺伝子組み換え作物は、文字通り「遺伝子組み換え技術」を利用して改良された作物です。 ここでは、遺伝子組み換え作物の概要や、私たちが食べてきた作物の品種改良の歴史と遺伝子組み換え作物が誕生した背景、そして、遺伝子組み換え作物と従来の品種との違い ...',
    '遺伝 子組み換え技術は酵母や麹菌の 育種 、甘味料の製造、また食品添加物にも利用されています。 牛乳を凝固させる 酵素 「 キモシン 」はナチュラルチーズの製造に欠かせませんが、天然の キモシン は仔牛の第４胃の中で分泌されるもので、大量に抽出することができず、貴重で高価なものでしたが、 キモシン を作る 遺伝 子を微生物に組み込むことで大量に生産することが可能になりました。 2．医薬品',
    '遺伝子組み換えとは、作物などに対し、他の生物の細胞から抽出した遺伝子を組み換え、新たな性質を持たせる手法を言う。'
]

tokenizer = Tokenizer()

tokens = tokenizer.tokenize(snippets[0])
# for token in tokens:
# print(token)

# 各文書から名詞のみを抽出する
docs = []  # snippets中の各テキストを名詞の配列に変換したものをdocsに格納する．

for snippet in snippets:
    tokens = tokenizer.tokenize(snippet)
    doc = []
    for token in tokens:
        if token.part_of_speech.split(',')[0] == "名詞":
            doc.append(token.base_form)
    docs.append(doc)

# print(docs[0])

# 文書コーパスを与えて、id->単語の辞書を得る
dictionary = gensim.corpora.Dictionary(docs)

# 単語-IDの辞書
word_id_dictionary = dictionary.token2id
print(word_id_dictionary)

id_docs = [dictionary.doc2bow(doc) for doc in docs]
print(id_docs)

model = gensim.models.TfidfModel(
    id_docs, normalize=False)  # normalize=Trueとすると、文書長でtfを正規化
tfidf_docs = model[id_docs]

print("doc0 = ", tfidf_docs[0])

print([(dictionary[f[0]], f[1]) for f in tfidf_docs[0]])

tf_vectors = gensim.matutils.corpus2dense(id_docs, len(dictionary)).T
print(tf_vectors[0])

tfidf_vectors = gensim.matutils.corpus2dense(tfidf_docs, len(dictionary)).T
print("doc0 = ", tfidf_vectors[0])

# コサイン類似度
from scipy.spatial.distance import cosine


def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)  # scipyのcosine関数は類似度ではなく距離を出力するので，類似度に変換する


print("sim(doc0, doc1) = ",
      cosine_similarity(tfidf_vectors[0], tfidf_vectors[1]))
print("sim(doc0, doc2) = ",
      cosine_similarity(tfidf_vectors[0], tfidf_vectors[2]))
print("sim(doc1, doc2) = ",
      cosine_similarity(tfidf_vectors[1], tfidf_vectors[2]))
