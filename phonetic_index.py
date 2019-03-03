import pickle
import os
import functools
import warnings

import gensim
import numpy as np
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from russian_g2p.Accentor import Accentor
from russian_g2p.Transcription import Transcription

class PhoneticIndex():
    def __init__(self, path_to_w2v='modelphonemes.model', path_to_annoy='annoy_index.ann', path_to_dict='data.pkl'):
        self.your_transcriptor = Transcription()
        self.your_accentor = Accentor()
        if os.path.isfile(path_to_w2v):
            self.model = gensim.models.Word2Vec.load(path_to_w2v)
        else:
            raise IOError("File {} does not exist!".format(path_to_w2v))
        if os.path.isfile(path_to_dict):
            with open(path_to_dict, 'rb') as f:
                self.dict_of_acc = pickle.load(f)
        else:
            raise IOError("File {} does not exist!".format(path_to_dict))
        self.accents = list(self.dict_of_acc.keys())
        f = len(self.accents[0])
        self.t = AnnoyIndex(f, metric='hamming')
        if os.path.isfile(path_to_annoy):
             self.t.load(path_to_annoy)
        else:
            raise IOError("File {} does not exist!".format(path_to_annoy))

    def transform(self, sentence, acc_number=10, sent_number=1):
        assert acc_number >= sent_number, "number of variants for nearest neighbors should be bigger than number of nearest sentences"
        phonemes = self.get_phonemes(sentence)
        accents = self.get_accents(sentence)
        closest_vectors = self.get_closest_vecs(accents, number=acc_number)
        closest_sentences = self.get_embeddings(closest_vectors, phonemes, number=sent_number)
        return closest_sentences

    def get_phonemes(self, sentence):
        # выдает транскрипцию
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            new_sentence = self.transcriptor(sentence)
        text = []
        for string in new_sentence[0]:
            text += sum(string[1], [])
        if len(text) != 0:
            try:
                # строит эмбеддинги пакетно
                phoneme_sent = self.model[text]
            except:
                # если символа нет в словаре эмбеддингов, строит поэлементно, заменяя неизвестный на вектор из 100 нулей
                phoneme_sent = []
                for word in text:
                    try:
                        phoneme_word = self.model[word]
                    except:
                        #print("unknown word", word)
                        phoneme_word = np.zeros(100)
                    phoneme_sent.append(phoneme_word)
                phoneme_sent = np.array(phoneme_sent)
            if len(phoneme_sent) < 100:
                # приведение к единому размеру 100
                difference = 100 - len(phoneme_sent)
                part = np.zeros((difference, 100))
                phoneme_sent = np.concatenate((part, phoneme_sent))
            assert len(phoneme_sent) == 100, "len of vector is inappropriate: {}".format(sentence)
        else:
            phoneme_sent = np.zeros((100, 100))
        return phoneme_sent

    def get_accents(self, sentence):
        # выдает вектор из 0 и 1 - ударений в предложении
        vector = []
        sentence = sentence.translate(sentence.maketrans('', '', '!&?\./(){}[]"$%^*+=@№<>|–—_€£±•`≠…§~«»₽,:;')).lower()
        for word in sentence.split():
            # ставит ударение в слове, если слово неизвестное, возвращается без ударения
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    accents = self.accentor(word)
            except:
                #print("unknown accent word: ", word)
                accents = [[word]]
            s = accents[0][0]
            vowels = "эоуиаеёюыяЭОУАЕИЁЫЮЯ"
            for letter, next_letter in zip(s, s[1:] + " "):
                # преобразование слов в бинарные вектора, где ударная гласная - 1, безударная 0
                if letter in vowels:
                    if next_letter == "+":
                        vector.append(1)
                    else:
                        vector.append(0)
        if len(vector) < 29:
            # приведение векторов к стандартному размеру - 29
            difference = 29 - len(vector)
            part = [0 for n in range(difference)]
            vector = part + vector
        assert len(vector) == 29, "len of vector is inappropriate: {}".format(sentence)
        return tuple(vector)

    def get_closest_vecs(self, vector, number=10):
        # возвращает список ближайших векторов в количестве number
        closest = [self.t.get_item_vector(x) for x in self.t.get_nns_by_vector(vector, number)]
        closest_int = [[int(x) for x in vector] for vector in closest]
        return closest_int

    def get_embeddings(self, vectors, source_embedding, number=1):
        # возвращает список ближайших предложений в количестве number
        possible_sentences = []
        for vector in vectors:
            possible_sentences += self.dict_of_acc[tuple(vector)]
        possible_embs = []
        embs_sentences = {}
        for sentence in possible_sentences:
            emb_sentence = self.get_phonemes(sentence)
            full_emb = np.concatenate(tuple(emb_sentence))
            possible_embs.append(full_emb)
            full_emb = tuple(full_emb)
            if full_emb not in embs_sentences:
                embs_sentences[full_emb] = list()
                embs_sentences[full_emb].append(sentence)
            else:
                embs_sentences[full_emb].append(sentence)
        assert len(possible_embs) >= number, "Number of nearest neighbors should be less than number of possible neighbors"
        source_embedding = np.concatenate(tuple(source_embedding))
        final_sentences = []
        neigh = NearestNeighbors(number)
        neigh.fit(possible_embs)
        nearest_neighbors = neigh.kneighbors([source_embedding], return_distance=False).tolist()
        for element in nearest_neighbors[0]:
            for sentence in embs_sentences[tuple(possible_embs[element])]:
                final_sentences.append(sentence.replace('\xa0', ' '))
        return final_sentences

    @functools.lru_cache(maxsize=None)
    def accentor(self, word):
        return self.your_accentor.do_accents([[word]])

    @functools.lru_cache(maxsize=None)
    def transcriptor(self, sentence):
        return self.your_transcriptor.transcribe([sentence])

example = PhoneticIndex().transform('Я пошел в лес выпить воды, выпил, но что-то пошло не так', acc_number=10, sent_number=1)
print(example)