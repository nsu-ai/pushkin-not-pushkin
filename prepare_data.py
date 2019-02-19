import pickle
import pandas as pd
import numpy as np

from annoy import AnnoyIndex
from tqdm import tqdm
from gensim.models import FastText
from nltk.tokenize import word_tokenize


def get_sentences():
    titles = pd.read_csv("titles.csv", index_col=0, names=["title"])["title"].values
    print("# of all sentences: {}".format(len(titles)))

    sentences = []
    for title in titles:
        if 4 <= len(title.split(" ")) <= 6:
            sentences.append(title)

    with open('./data/sentences.pickle', 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return sentences

"""
def build_embeddings(sentences, batch_size=64):
    elmo = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
                      trainable=True)

    tf_embeddings = []
    begin_indexes = list(range(0, len(sentences), batch_size))
    end_indexes = list(range(batch_size, len(sentences), batch_size))

    for i, j in tqdm(zip(begin_indexes, end_indexes)):
        tf_embeddings.extend(elmo(sentences[i:j], signature="default", as_dict=True)["elmo"])

    embeddings = []
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for embedding in tqdm(tf_embeddings):
            embeddings.append(embedding.eval())

    with open("embeddings.pkl", "")

    return embeddings
"""


def load_russian_fasttext_rusvectores():
    return FastText.load("araneum_none_fasttextcbow_300_5_2018.model").wv


def build_fasttext_embeddings(sentences):
    vectorizer = load_russian_fasttext_rusvectores()

    embeddings = []
    for sentence in tqdm(sentences):
        vector = vectorize_sentence(sentence, vectorizer)
        embeddings.append(vector / np.linalg.norm(vector))

    return embeddings


def build_indexer(embeddings):
    t = AnnoyIndex(300)
    for i in tqdm(range(len(embeddings))):
        t.add_item(i, embeddings[i])

    t.build(10)
    t.save('test.ann')


def vectorize_sentence(sentence, vectorizer):
    embeddings = []
    for word in word_tokenize(sentence):
        try:
            embeddings.append(vectorizer[word])
        except Exception as e:
            continue

    return np.mean(embeddings, axis=0)


def retrieve(sentence, vectorizer, indexer, sentences):
    vectorized_sentence = vectorize_sentence(sentence, vectorizer)

    matches = indexer.get_nns_by_vector(vectorized_sentence, 10)

    return np.array(sentences)[matches]


if __name__ == "__main__":

    sentences = get_sentences()

    #print("starting to build embeddings")
    #embeddings = build_fasttext_embeddings(sentences)

    #print("starting to build an indexer")
    #build_indexer(embeddings)

    #indexer = AnnoyIndex(300)
    #indexer.load("test.ann")

    #vectorizer = load_russian_fasttext_rusvectores()

    #print(retrieve("Сбербанк искусственный интеллект", vectorizer, indexer, sentences))
