import os
import pickle
import argparse
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm


class Indexer(object):
    def __init__(self, filename='./data/indexer_21_1024_147989.annoy'):
        self.indexer = self._load_indexer(filename)

    def get_n_closest(self, vector, n_samples=10):
        return np.array(self.indexer.get_nns_by_vector(vector, n_samples))

    def _load_indexer(self, indexer_file_name):
        parameters = indexer_file_name.split('_')
        n_trees = int(parameters[1])
        vector_dim = int(parameters[2])
        n_samples = int(parameters[3].split('.')[0])

        annoy_index = AnnoyIndex(vector_dim)
        annoy_index.load(indexer_file_name)

        return annoy_index

    @staticmethod
    def build_indexer(vectorized_sentences, n_trees=10, save_dir='./data'):
        embedding_dim = vectorized_sentences[0].shape[0]

        annoy_index = AnnoyIndex(embedding_dim)

        for i, embedding in tqdm(enumerate(vectorized_sentences), total=len(vectorized_sentences)):
            annoy_index.add_item(i, embedding)

        annoy_index.build(n_trees)
        annoy_index.save(os.path.join(save_dir, f'indexer_{n_trees}_{embedding_dim}_{i}.annoy'))


def index_corpus(arguments):
    vectorized_filename = arguments.vectorized_filename
    n_trees = arguments.n_trees
    save_dir = arguments.save_dir

    with open(vectorized_filename, 'rb') as f:
        vectorized_sentences = pickle.load(f)

    Indexer.build_indexer(vectorized_sentences, n_trees, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_corpus = subparsers.add_parser('index_corpus')
    parser_corpus.add_argument('--vectorized_filename', type=str, required=True)
    parser_corpus.add_argument('--n_trees', type=int, required=True)
    parser_corpus.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'index_corpus':
        index_corpus(args)
    else:
        raise Exception("Error!")
