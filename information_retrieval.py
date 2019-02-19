import pickle
import random
import re
import numpy as np

from gensim.models import FastText
from annoy import AnnoyIndex
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as hub


class InformationRetriever(object):
    def __init__(self):
        self.sentences = self._load_sentences()
        self.vectorizer = self._load_vectorizer()
        self.indexer = self._load_indexer()
        self.english_letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()')
        self.seeds = ['россия щедрая душа', 'всё будет хорошо', 'мы идём от победы к победе']
        self.re_for_splitting = re.compile(r'\W+')

    def train(self):
        pass

    def split_text(self, src):
        return list(filter(lambda it: len(it) > 0, self.re_for_splitting.split(src)))

    def retrieve(self, sentence, n=10):
        vectorized_sentence = self._vectorize_sentence(sentence)
        matches = self.indexer.get_nns_by_vector(vectorized_sentence, n * 10)
        sentences = list(filter(lambda it: len(set(it) & self.english_letters) == 0,
                                map(lambda it2: it2.strip(), self.sentences[matches])))
        if len(sentences) < 2:
            vectorized_sentence = self._vectorize_sentence(random.choice(self.seeds))
            matches = self.indexer.get_nns_by_vector(vectorized_sentence, n * 10)
            sentences = list(filter(lambda it: len(set(it) & self.english_letters) == 0,
                                    map(lambda it2: it2.strip(), self.sentences[matches])))
        sentences_ = list(filter(lambda it: len(self.split_text(it)) == 4, sentences))
        if len(sentences_) < 2:
            sentences_ = list(filter(lambda it: len(self.split_text(it)) == 4, sentences)) + \
                         list(filter(lambda it: len(self.split_text(it)) == 5, sentences))
        if len(sentences_) < 2:
            vectorized_sentence = self._vectorize_sentence(random.choice(self.seeds))
            matches = self.indexer.get_nns_by_vector(vectorized_sentence, n * 10)
            sentences = list(filter(lambda it: len(set(it) & self.english_letters) == 0,
                                    map(lambda it2: it2.strip(), self.sentences[matches])))
            sentences_ = list(filter(lambda it: len(self.split_text(it)) == 4, sentences))
            if len(sentences_) < 2:
                sentences_ = list(filter(lambda it: len(self.split_text(it)) == 4, sentences)) + \
                             list(filter(lambda it: len(self.split_text(it)) == 5, sentences))
        if len(sentences_) > n:
            sentences_ = sentences_[:n]
        return list(map(lambda it2: it2[:-1].strip() if it2.endswith('-') else it2,
                        map(lambda it: it[1:].strip() if it[0].startswith('-') else it, sentences_)))

    def _vectorize_sentence(self, sentence):
        word_embeddings = []

        word_embeddings = self.vectorizer(
            [sentence],
            signature="default",
            as_dict=True)["elmo"]

        print(word_embeddings)

        return word_embeddings

    def _load_sentences(self):
        with open("./data/sentences.pickle", "rb") as f:
            sentences = pickle.load(f)

        return np.array(sentences)

    def _load_vectorizer(self):
        #return FastText.load("./data/araneum_none_fasttextcbow_300_5_2018.model").wv
        #return ELMoEmbedder('http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz')
        elmo = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
                          trainable=True)
        return elmo

    def _load_indexer(self):
        indexer = AnnoyIndex(300)
        indexer.load("./data/test.ann")

        return indexer


if __name__ == "__main__":
    print(InformationRetriever().retrieve("Греф"))
