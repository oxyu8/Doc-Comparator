# coding: UTF-8
from flask import Flask, request
import json

from scipy.spatial.distance import cosine
import numpy
import janome
import sklearn
import plotly

from janome.tokenizer import Tokenizer
import gensim

app = Flask(__name__)

ANSWERS = [
    '他の回答をみて、[遺伝子組み換え作物（GMO）]について基本的な誤解があるようなので補足します。遺伝子組み換えが問題になっているのは、遺伝子が組み替えられている、というそのこと自体ではなくて、史上最悪の環境破壊企業と言われる[モンサント]がラウンドアップなどの農薬を売るために、除草剤耐性をもつよう品種改良し、その農薬と改良された種子のセットで特許をもち、独占的な支配を構築してきているからです。実名でドキュメンタリー映画にもなっています。[映画『モンサントの不自然な食べもの』公式サイト]しかもタチが悪いのは、この改変された種子が飛散し、アメリカから離れたメキシコのトウモロコシ畑を汚染するなど、品種の多様性が世界的に低下するのみならず、モンサントは自社の品種を無断で使っているとして、被害者であるはずのメキシコ農家を訴えるなど、傍若無人のかぎりをつくしています。アンチGMOというのは、力を持ちすぎた環境破壊企業と[穀物メジャー]の横暴に対する反発が根底にあり、**ボイコット、不買運動のような政治的なキャンペーン**なのです。農薬を繰り返し使用した土壌は微生物などの生態系も乱れ、現代の作物の栄養価は低下してきています。たとえば、ふつうに食事をしているだけでもオメガ6とオメガ3のバランスが崩れるようになり、炎症体質でアトピーやアレルギーになる子供が増えました。アメリカのインテリ層ではこのような知識は常識ですし、USDA（アメリカ農務省）が認定するオーガニックの基準はそれなりにしっかりしていて、たとえば「オーガニック作物を栽培する土地では、収穫前3年以上禁止物質を使用しない」、食肉なら「食肉用の動物は、3世代前から(鶏肉の場合は、生後2日目以降から)、オーガニックな管理のもとで育てられなければならない」などの細かいルールがあります。[USDA オーガニック]ひるがえって、日本の有機栽培やオーガニックの世界は無法地帯で信用できません。なぜそうなっているかというと、日本が食料輸入大国であり廃棄大国であること、狭い国土の土壌の有機物が飽和して発酵ではなく腐敗しつつあること、など、闇は深いです。これについて書き出すと上記の何倍もの長文になってしまうので別の機会に。なお、遺伝子組み替えした食品を食べるかどうかについては、これだけの支配力をもっている企業が相手なので、ほとんど避けようがありません。少なくとも外食をする人は必ず何らかの形で食べていると言えます',
    '我々は遺伝子の配列はたくさん読み取っていても、それが生体内で、多様な環境で、色んなライフサイクルで、どのように働いているかは、まだまだ暗中模索な段階です。　特定の遺伝子が作るタンパク質が、特定の効果を発揮したからといって、それを薬剤として添加するのでなく、遺伝子を組み込んで生体内で作らせるというのは、無謀すぎる。遺伝子を組み込むのに使ったベクターや制限酵素が、遺伝子の別の場所を壊したり、飛び出して別の個体に水平伝播する可能性を否定できない。。遺伝子組み換え作物を露地栽培する事で、他で栽培されている遺伝子組み換えしていない同種の作物と交配して、不測の伝播をする可能性がある。。遺伝子組み換えした個体の検証試験といっても、期間が短く、その種の一生のライフサイクルや「一生」を超えた長期の世代交代や進化への影響は検証されていない。。現時点の「遺伝子組み換え」や「ゲノム編集」といっても、薬剤として確認された殺虫・忌避成分や薬剤耐性成分を、外から散布するのではなく、遺伝子に組み込んで生体内で作らせるという「手抜き」であって、何ら革新的な事ではなく、その組み込み成分に対する耐性を害虫や雑草が獲得したら、全く意味のない「余計な毒」を内に抱える事になり、果てしなく新たな毒を組み込む恐ろしい事になる。薬剤として散布した「毒」なら、洗えば除去できるが、遺伝子組み換えで生体に作らせたのでは、洗浄できず、飼料にしても家畜の生産物にまで毒は残る。',
    '何億年という人類の歴史で人間は色々な食べ物に適応することを学びました。GMO遺伝子組み換え作物は地球上には初登場の食物ですから、人体には初めてです。他の食物は何百年、何千年、何億年とかかって慣れているのですが、地球上初の食べ物。体はどうして良いか分かりません。アレルギー反応となり、色々な症状が起こり人間を悩ませます。',
    'アメリカでは、遺伝子組み換え作物の普及と甲状腺ガンの増加の因果関係が指摘されています。(下図)もちろん、その直接のメカニズムが明らかでない以上、遺伝子組み換えのせいだと断言できませんが、食物に関しては「安全優先」が原則です。害虫の忌避成分の遺伝子組み込みにしても、除草剤耐性を持たせる遺伝子組み込みにしても、耐性のある害虫や雑草が出てきて、より強い成分を組み込んだり、より強い除草剤をまいたり、ぜんぜん「画期的」な成果ではなく、リスクだけを負った形です。',
    '本当に、が現代科学の範囲であれば。安全ってなに？という疑問もありますけど。牛肉たべて牛になった人はまだ居ないですし。作物から油とって、それで揚げ物する。遺伝子なんか残ってませんし。気象がおかしくなり、イナゴが大発生し、収穫を維持するためには遺伝子組み換えくらいやらないと間に合わない、という状況もありそうです。なにも食べるものがない、よりはあった方が生存できる可能性は大きいですね。'
]


def get_similarity(snippets, parsed_json):
    print('ANSWER', len(ANSWERS))  # 5
    tokenizer = Tokenizer()

    # 各文書から名詞のみを抽出する
    docs = []  # snippets中の各テキストを名詞の配列に変換したものをdocsに格納する．
    snippets_answers = ANSWERS + snippets

    # index0~4がquoraの回答
    print("------------")
    print('snippets: ', len(snippets))  # 10 is expected
    print('answers: ', len(ANSWERS))  # 5 is expected
    print('snippets + answers: ', len(snippets_answers))  # 15 is expected
    print("------------")

    for snippet in snippets_answers:
        tokens = tokenizer.tokenize(snippet)
        doc = []
        for token in tokens:
            if token.part_of_speech.split(',')[0] == "名詞":
                doc.append(token.base_form)
        docs.append(doc)

    print('docs')
    print(len(docs))

    # 文書コーパスを与えて、id->単語の辞書を得る
    dictionary = {}
    dictionary = gensim.corpora.Dictionary(docs)
    print('fafsadfdsa')
    print(len(dictionary))

    # 単語-IDの辞書
    word_id_dictionary = {}
    word_id_dictionary = dictionary.token2id

    id_docs = []
    id_docs = [dictionary.doc2bow(doc) for doc in docs]

    model = gensim.models.TfidfModel(
        id_docs, normalize=True)  # normalize=Trueとすると、文書長でtfを正規化
    tfidf_docs = []
    tfidf_docs = model[id_docs]

    tfidf_vectors = []
    tfidf_vectors = gensim.matutils.corpus2dense(tfidf_docs, len(dictionary)).T
    answer_tfidf_vectors = []
    answer_tfidf_vectors = tfidf_vectors[:5]
    snippet_tfidf_vectors = []
    snippet_tfidf_vectors = tfidf_vectors[5:]
    print('length')
    print(len(snippet_tfidf_vectors))
    print(snippet_tfidf_vectors)

    # コサイン類似度
    def cosine_similarity(vec1, vec2):
        # scipyのcosine関数は類似度ではなく距離を出力するので，類似度に変換する
        return 1 - cosine(vec1, vec2)

    for index, s in enumerate(snippet_tfidf_vectors):
        print("--------------")
        bing_result = parsed_json[index]
        for idx, a in enumerate(answer_tfidf_vectors):
            bing_result[str(idx)] = cosine_similarity(a, s)

    return parsed_json
    # 各回答とスニペットの類似度を計算する


@app.route('/', methods=['POST'])
def hello():
    data = request.form['data']
    parsed_json = json.loads(data)
    print(len(parsed_json))
    snippets = []
    for snippet in parsed_json:
        snippets.append(snippet['snippet'])
    result = get_similarity(snippets, parsed_json)
    enc = json.dumps(result)
    return enc


# おまじない
if __name__ == "__main__":
    app.run(debug=True)
