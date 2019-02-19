import codecs
import json
import os
import random
import re

from flask import Flask, request, jsonify
from information_retrieval import InformationRetriever
from poetry_generation import generate_poems
from poetry_ranking import PoemsRanker
from rnnmorph.predictor import RNNMorphPredictor
from rhyme import find_rhyme, load_phonetic_dict

import sys
sys.path.append('rupo')
from rupo.api import Engine


app = Flask(__name__)

retriever = InformationRetriever()
ranker = PoemsRanker()
rnn_morph = RNNMorphPredictor(language="ru")
russian_lexemes_name = os.path.join(os.path.dirname(__file__), 'data', 'russian_lexemes.json')
with codecs.open(russian_lexemes_name, mode='r', encoding='utf-8', errors='ignore') as fp:
    russian_lexemes_data = json.load(fp)
russian_phonetic_dictionary = load_phonetic_dict(os.path.join(os.path.dirname(__file__), 'data', 'voxforge_ru.dic'))
engine = Engine(language="ru")
path_to_stress_models = os.path.join(os.path.dirname(__file__), 'rupo', 'rupo', 'data', 'stress_models',
                                     'stress_ru_LSTM64_dropout0.2_acc99_wer8.h5')
path_to_zaliznyak = os.path.join(os.path.dirname(__file__), 'rupo', 'rupo', 'data', 'dict', 'zaliznyak.txt')
engine.load(path_to_stress_models, path_to_zaliznyak)
re_for_splitting = re.compile(r'\W+')


@app.route('/ready')
def ready():
    return 'OK'


@app.route('/generate/<poet_id>', methods=['POST'])
def generate(poet_id):
    seed = request.get_json()["seed"]

    sentences = list(set(map(lambda it: it.replace('\xa0', ' '), retriever.retrieve(seed, n=6))))
    print('Number of sentences is {0}'.format(len(sentences)))

    rhymes = []
    sentences_ = []
    for cur in sentences:
        metre = engine.classify_metre(' '.join(re_for_splitting.split(cur)))
        print('Sentence: ' + cur + ', metre is {0}'.format(metre))
        all_variants = find_rhyme(' '.join(re_for_splitting.split(cur)), rnn_morph=rnn_morph, russian_lexemes=russian_lexemes_data,
                                  phonetic_dict=russian_phonetic_dictionary)
        filtered_variants = list(filter(lambda it: engine.classify_metre(it) == metre, all_variants))
        print('Number of all rhyme variants is {0}.'.format(len(all_variants)))
        print('Number of all filtered variants is {0}.'.format(len(filtered_variants)))
        print('')
        if len(filtered_variants) > 0:
            sentences_.append(cur)
            rhymes.append(filtered_variants)
        else:
            sentences_.append(cur)
            rhymes.append([random.choice(all_variants)])

    poems = generate_poems(sentences_, rhymes)
    if len(poems) > 5000:
        poems = poems[:5000]

    print('Number of poems is {0}.'.format(len(poems)))
    poem = ranker.select_best_poem(poems, seed)
    return jsonify({'poem': '\n'.join(map(lambda it: it[0].upper() + it[1:], poem.split('\n')))})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    print(1)