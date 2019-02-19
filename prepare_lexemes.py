from argparse import ArgumentParser
import codecs
import json
import os
import sys
from typing import Union

import pymorphy2
from russian_tagsets import converters

sys.path.append('rupo')
from rupo.api import Engine


def check_word(src: str) -> bool:
    russian_letters = set('АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя')
    if len(src) < 3:
        return False
    if not(set(src) <= russian_letters):
        return False
    all_vowels = set('АаЕеЁёИиОоУуЫыЭэЮюЯя')
    n_vowels = 0
    for cur in src:
        if cur in all_vowels:
            n_vowels += 1
    n_consonants = len(src) - n_vowels
    if (n_vowels == 0) or (n_consonants == 0):
        return False
    if len(src) >= 4:
        if (n_vowels < 2) or (n_consonants < 2):
            return False
    return True


def get_morphodata(src_morpho_data: str) -> Union[str, None]:
    morphodata = list(filter(lambda it: len(it) > 0, src_morpho_data.split()))
    if len(morphodata) < 2:
        return None
    pos = morphodata[0]
    if pos not in {'NOUN', 'VERB', 'ADJ'}:
        return None
    if pos == 'NOUN':
        tag = '|'.join(filter(lambda it: it.split('=')[0] in {'Case', 'Gender', 'Number'}, morphodata[1].split('|')))
    elif pos == 'VERB':
        tag = '|'.join(filter(lambda it: it.split('=')[0] in {'Gender', 'Number', 'Mood', 'VerbForm'},
                              morphodata[1].split('|')))
    else:
        tag = '|'.join(filter(lambda it: it.split('=')[0] in {'Case', 'Gender', 'Number'},
                              morphodata[1].split('|')))
    if len(tag) == 0:
        return None
    return pos + ' ' + tag


def unknown_word(parsing: list):
    ok = False
    for cur in parsing:
        if any(map(lambda it: str(it[0]).lower().find('unk') >= 0, cur.methods_stack)):
            ok = True
            break
    return ok


def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='src_name', type=str, required=True,
                        help='A text file with source dictionary.')
    parser.add_argument('-d', '--dst', dest='dst_name', type=str, required=True,
                        help='A JSON file with destination dictionary.')
    args = parser.parse_args()

    with codecs.open(args.src_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        all_words = sorted(list(set(
            map(
                lambda it5: it5[1],
                filter(
                    lambda it4: check_word(it4[1]) and (int(it4[0]) >= 10),
                    map(
                        lambda it3: it3.lower().strip().split(),
                        filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), fp.readlines()))
                    )
                )
            )
        )))
    print('Number of selected words is {0}.'.format(len(all_words)))
    words_dict = dict()
    morph = pymorphy2.MorphAnalyzer()
    to_ud20 = converters.converter('opencorpora-int', 'ud20')
    engine = Engine(language="ru")
    engine.load(
        os.path.join(os.path.dirname(__file__), 'rupo', 'rupo', 'data', 'stress_models',
                     'stress_ru_LSTM64_dropout0.2_acc99_wer8.h5'),
        os.path.join(os.path.dirname(__file__), 'rupo', 'rupo', 'data', 'dict', 'zaliznyak.txt')
    )
    syllables_of_words = dict()
    counter = 0
    unknown_counter = 0
    for cur_word in all_words:
        if cur_word in syllables_of_words:
            n_syllables = syllables_of_words[cur_word]
        else:
            n_syllables = len(engine.get_word_syllables(cur_word))
            syllables_of_words[cur_word] = n_syllables
        if n_syllables == 0:
            continue
        parsing = morph.parse(cur_word)
        if unknown_word(parsing):
            unknown_counter += 1
        else:
            for it in parsing:
                morphodata = get_morphodata(to_ud20(str(it.tag)))
                if morphodata is None:
                    continue
                if morphodata in words_dict:
                    if n_syllables in words_dict[morphodata]:
                        words_dict[morphodata][n_syllables].add(cur_word)
                    else:
                        words_dict[morphodata][n_syllables] = {cur_word}
                else:
                    words_dict[morphodata] = {n_syllables: {cur_word}}
        counter += 1
        if counter % 10000 == 0:
            print('{0} words have been processed...'.format(counter))
    print('There are {0} unknown words.'.format(unknown_counter))
    for morphodata in words_dict:
        for n_syllables in words_dict[morphodata]:
            words_dict[morphodata][n_syllables] = sorted(list(words_dict[morphodata][n_syllables]))
    with codecs.open(args.dst_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(words_dict, fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
