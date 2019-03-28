import os
import pickle
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
from itertools import islice


os.environ["TFHUB_CACHE_DIR"] = './data/tfhub_cache'

EMBEDDING_DIM = 1024


class Vectorizer(object):
    def __init__(self, embedding_type='elmo'):
        self.embedding_type = embedding_type
        self.vectorizer = self._load_vectorizer(self.embedding_type)

    def vectorize_sentence(self, sentence):
        vectorizer = self._load_vectorizer('elmo')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        vector = sess.run(vectorizer([sentence], signature='default', as_dict=True)["elmo"])[0]
        print(vector.shape)

        sess.close()
        tf.reset_default_graph()

        return np.max(vector, axis=0)

    def vectorize_by_batch(self, sentences, sess):
        vector = sess.run(self.vectorizer(sentences, signature='default', as_dict=True)["elmo"])

        max_pooling_batch = np.zeros((len(vector), 1024), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            if len(sentence.split(' ')) == 0:
                continue

            max_pooling_batch[i] = np.max(vector[i][:len(sentence.split(' '))], axis=0)

        del vector

        return max_pooling_batch

    def _load_vectorizer(self, embedding_type='elmo'):
        if embedding_type == 'elmo':
            return hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
                              trainable=True)


def vectorize_corpus(arguments):
    sentences_filename = arguments.sentences_filename
    vectorized_dir = arguments.vectorized_dir
    embeddings_type = arguments.embeddings_type
    batch_size = arguments.batch_size

    with open(sentences_filename, 'r') as f:
        sentences = [sentence.strip() for sentence in f.readlines()]
    n_sentences = len(sentences)

    vectorized_sentences = np.zeros((n_sentences, EMBEDDING_DIM), dtype=np.float32)

    for i in tqdm(range(0, n_sentences, batch_size), total=n_sentences // batch_size + 1):
        vectorizer = Vectorizer(embeddings_type)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        vectorized_sentences[i:i + batch_size] = vectorizer.vectorize_by_batch(sentences[i:i + batch_size], sess)

        sess.close()
        tf.reset_default_graph()

    vectorized_filename = os.path.basename(sentences_filename).split('.')[0] + f'_{embeddings_type}_{EMBEDDING_DIM}'

    np.save(vectorized_filename, vectorized_sentences)
    #with open(os.path.join(vectorized_dir, vectorized_filename), 'wb') as f:
    #    pickle.dump(vectorized_sentences, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_corpus = subparsers.add_parser('vectorize_corpus')
    parser_corpus.add_argument('--sentences_filename', type=str, required=True)
    parser_corpus.add_argument('--vectorized_dir', type=str, required=True)
    parser_corpus.add_argument('--embeddings_type', type=str, required=True)
    parser_corpus.add_argument('--batch_size', type=int, required=True)

    args = parser.parse_args()

    if args.mode == 'vectorize_corpus':
        vectorize_corpus(args)
    else:
        raise Exception("Error!")
