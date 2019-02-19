import codecs
import difflib
import random
from typing import List, Dict

from nltk import word_tokenize
from rnnmorph.predictor import RNNMorphPredictor

from prepare_lexemes import get_morphodata


MAX_VARIANTS_OF_WORD = 3


def calc_number_of_syllables(src: str) -> int:
    vowels = set('АаЕеЁёИиОоУуЫыЭэЮюЯя')
    n = 0
    for cur in src:
        if cur in vowels:
            n += 1
    return n


def calc_phonetic_similarity(left_word: str, right_word: str, phonetic_dict: Dict[str, tuple]) -> float:
    if (left_word in phonetic_dict) and (right_word in phonetic_dict):
        left_transcription = phonetic_dict[left_word]
        right_transcription = phonetic_dict[right_word]
        matcher = difflib.SequenceMatcher(a=''.join(left_transcription), b=''.join(right_transcription))
    else:
        matcher = difflib.SequenceMatcher(a=left_word, b=right_word)
    return matcher.ratio()


def is_rhyme(left_word: str, right_word: str, phonetic_dict: Dict[str, tuple], c: float=1.0) -> bool:
    if (left_word in phonetic_dict) and (right_word in phonetic_dict):
        left_transcription = phonetic_dict[left_word]
        right_transcription = phonetic_dict[right_word]
        n = min(len(left_transcription), len(right_transcription), 6)
        matcher = difflib.SequenceMatcher(a=''.join(left_transcription[-n:]), b=''.join(right_transcription[-n:]))
        th = 0.9 * c
    else:
        n = min(len(left_word), len(right_word), 6)
        matcher = difflib.SequenceMatcher(a=left_word[-n:], b=right_word[-n:])
        th = 0.7 * c
    return matcher.ratio() >= th


def select_new_variant(src_words: List[str], morphotags: List[str], syllables: List[str], russian_lexemes: dict,
                       phonetic_dict: Dict[str, tuple], n_pass: int, new_variant: List[str]):
    if len(src_words) == 0:
        if len(new_variant) > 0:
            yield new_variant
    else:
        if morphotags[0] is None:
            yield from select_new_variant(src_words[1:], morphotags[1:], syllables[1:], russian_lexemes,
                                          phonetic_dict, 0, new_variant + [src_words[0]])
        elif morphotags[0] in russian_lexemes:
            if n_pass > 0:
                target_syllables_number = str(int(syllables[0]) + n_pass)
            else:
                target_syllables_number = syllables[0]
            possible_words = set()
            if target_syllables_number in russian_lexemes[morphotags[0]]:
                if len(src_words) > 1:
                    possible_words = set(russian_lexemes[morphotags[0]][target_syllables_number]) - {src_words[0]}
                else:
                    possible_words = set(filter(lambda it: is_rhyme(src_words[0], it, phonetic_dict) and
                                                           (src_words[0] != it),
                                                russian_lexemes[morphotags[0]][target_syllables_number]))
                    if len(possible_words) == 0:
                        possible_words = set(filter(lambda it: is_rhyme(src_words[0], it, phonetic_dict, 0.5) and
                                                               (src_words[0] != it),
                                                    russian_lexemes[morphotags[0]][target_syllables_number]))
            if len(possible_words) > 0:
                possible_words = sorted(
                    list(possible_words),
                    key=lambda it: -calc_phonetic_similarity(src_words[0], it, phonetic_dict)
                )
                if len(possible_words) > (3 * MAX_VARIANTS_OF_WORD):
                    possible_words = possible_words[:(3 * MAX_VARIANTS_OF_WORD)]
                if len(possible_words) > MAX_VARIANTS_OF_WORD:
                    random.shuffle(possible_words)
                    possible_words = sorted(possible_words[:MAX_VARIANTS_OF_WORD])
                for cur in possible_words:
                    yield from select_new_variant(src_words[1:], morphotags[1:], syllables[1:], russian_lexemes,
                                                  phonetic_dict, 0, new_variant + [cur])
            else:
                yield from select_new_variant(src_words[1:], morphotags[1:], syllables[1:], russian_lexemes,
                                              phonetic_dict, 0, new_variant + [src_words[0]])
            if morphotags[0].startswith('ADJ ') and (n_pass == 0) and (len(src_words) > 1):
                yield from select_new_variant(src_words[1:], morphotags[1:], syllables[1:], russian_lexemes,
                                              phonetic_dict, int(syllables[0]), new_variant)
        else:
            yield from select_new_variant(src_words[1:], morphotags[1:], syllables[1:], russian_lexemes,
                                          phonetic_dict, 0, new_variant + [src_words[0]])


def load_phonetic_dict(file_name: str) -> Dict[str, tuple]:
    phonetic_dict = dict()
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                parts = prep_line.split()
                if len(parts) > 1:
                    new_word = parts[0]
                    new_transcription = tuple(parts[1:])
                    idx = new_word.find('(')
                    if idx < 0:
                        phonetic_dict[new_word] = new_transcription
            cur_line = fp.readline()
    return phonetic_dict


def find_rhyme(src: str, russian_lexemes: dict, rnn_morph: RNNMorphPredictor,
               phonetic_dict: Dict[str, tuple]) -> List[str]:
    russian_letters = set('АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя')
    src_words = list(filter(
        lambda it2: set(it2) <= russian_letters,
        map(lambda it1: it1.strip().lower(), word_tokenize(src))
    ))
    if len(src_words) == 0:
        return [src]
    morphotags = [get_morphodata(cur.pos + ' ' + cur.tag) for cur in rnn_morph.predict(src_words)]
    print('morphotags', morphotags)
    syllables_of_words = [str(calc_number_of_syllables(cur_word)) for cur_word in src_words]
    print('syllables_of_words', syllables_of_words)
    variants = []
    new_variant = []
    for it in select_new_variant(src_words, morphotags, syllables_of_words, russian_lexemes, phonetic_dict, 0,
                                 new_variant):
        variants.append(' '.join(it))
        del it
    return variants
