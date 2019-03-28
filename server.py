import codecs
import json
import os
import random
import re

from flask import Flask, request, jsonify
from information_retrieval import InformationRetriever
from phonetic_index import PhoneticIndex

#from poetry_generation import generate_poems
from poetry_ranking import PoemsRanker
#from rnnmorph.predictor import RNNMorphPredictor
#from rhyme import find_rhyme, load_phonetic_dict

import sys
#sys.path.append('rupo')
#from rupo.api import Engine


app = Flask(__name__)

retriever = InformationRetriever()
ranker = PoemsRanker()
phonetic_index = PhoneticIndex()

#rnn_morph = RNNMorphPredictor(language="ru")
#russian_lexemes_name = os.path.join(os.path.dirname(__file__), 'data', 'russian_lexemes.json')
#with codecs.open(russian_lexemes_name, mode='r', encoding='utf-8', errors='ignore') as fp:
#    russian_lexemes_data = json.load(fp)

#russian_phonetic_dictionary = load_phonetic_dict(os.path.join(os.path.dirname(__file__), 'data', 'voxforge_ru.dic'))
#engine = Engine(language="ru")
#path_to_stress_models = os.path.join(os.path.dirname(__file__), 'rupo', 'rupo', 'data', 'stress_models',
#                                     'stress_ru_LSTM64_dropout0.2_acc99_wer8.h5')

#path_to_zaliznyak = os.path.join(os.path.dirname(__file__), 'rupo', 'rupo', 'data', 'dict', 'zaliznyak.txt')
#engine.load(path_to_stress_models, path_to_zaliznyak)
re_for_splitting = re.compile(r'\W+')


@app.route('/ready')
def ready():
    return 'OK'


@app.route('/generate/<poet_id>', methods=['POST'])
def generate(poet_id):
    seed = request.get_json()["seed"]

    sentences = list(set(map(lambda it: it.replace('\xa0', ' '), retriever.retrieve(seed, n=4))))
    print('Number of sentences is {0}'.format(len(sentences)))

    poem_1 = [sentences[0], sentences[2]]
    poem_2 = [sentences[1], sentences[3]]

    poem_11 = [retriever.retrieve(poem_1[0], 1)[0], retriever.retrieve(poem_1[1], 1)[0]]
    poem_22 = [retriever.retrieve(poem_2[0], 1)[0], retriever.retrieve(poem_2[1], 1)[0]]

    poems = list()
    poems.append('\n'.join([poem_1[0], poem_1[1], poem_11[0], poem_11[1]]))
    poems.append('\n'.join([poem_2[0], poem_22[0], poem_2[1], poem_22[1]]))

    poem = ranker.select_best_poem(poems, seed)
    return jsonify({'poem': '\n'.join(map(lambda it: it[0].upper() + it[1:], poem.split('\n')))})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)
