from __future__ import print_function
import json
import logging
import sys
import argparse

import gensim


class IterCorpus:
    def __init__(self, path=None, known_words=None, iteration_hook=lambda i: i,
            cycle=5000):
        """A simple streaming corpus to use with gensim.models.Doc2Vec.

        Parameters
        -----------
        path : str
            Path to the corpus to be to streamed.
        known_words : set
            Non-stopword words.
        iteration_hook: function
            A function taking in a document and returning a transformed
            version of it.
        cycle : int
            How often to print logging messages.

        Returns
        -----------
        :IterCorpus
        """

        self.path = path
        self.known_words = known_words
        self.iteration_hook = iteration_hook
        self.logger = logging.getLogger(__name__)
        self.cycle = cycle

    def __iter__(self):
        with open(self.path, mode='r', buffering=-1) as f:
            for idx, line in enumerate(f):
                doc = json.loads(line)
                if (idx%self.cycle)==0:
                    self.logger.info('@ {}'.format(idx))
                doc['bow'] = [word for word in doc['bow'] if
                        word in self.known_words]
                yield self.iteration_hook(doc)


def iteration_hook(doc):
    """Produces a tagged document suitable for use with Doc2Vec.

    Parameters
    ------------
    doc : dictionary
        A dictionary with the keys 'bow' and 'idx'.

    Returns
    -----------
    TaggedDocument : collections.NamedTuple
        A NamedTuple with attributes 'words' and 'tags'.
    """

    return gensim.models.doc2vec.TaggedDocument(words=doc['bow'], tags=['idx'])


def _make_parser():
    parser = argparse.ArgumentParser(
            description='Run doc2vec on a corpus that is too large to hold \
                    in memory.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id2word',
            required=True,
            help='Location of a saved id2word in gensims binary format.',
            metavar='gensim_dictionary_binary'.upper())
    parser.add_argument('--data',
            required=True,
            help='Path of data to be streamed.',
            metavar='input_data'.upper())
    parser.add_argument('--output',
            default='doc2vec.bin',
            help='Location to save trained model.',
            metavar='output_location'.upper())
    parser.add_argument('--mmap',
            default='mapfile.bin',
            help='Path to save memory mapped arrays. They should be in the\
                    same folder as OUTPUT_LOCATION',
                    metavar='memory_map'.upper())
    return parser


if __name__=='__main__':
    parser = _make_parser()
    args = parser.parse_args()
    logging = logging.basicConfig(level=logging.DEBUG)
    logger = logging.getlogger(__name__)
    id2word = gensim.corpora.Dictionary.load(args.id2word)
    

    #modify word removal settings here
    #TODO make this a command line option
    id2word.filter_extremes(no_below=2, no_above=0.9)
    known_words = set(id2word.values())

    itercorpus = IterCorpus(path=args.data, known_words=known_words,
            iteration_hook=iteration_hook)

    #arguments are set to some sensible defaults for our use case
    doc2vec = gensim.models.Doc2Vec(documents=itercorpus,
            min_count=0, workers=8, size=500, window=4, dbow_words=0,
            negative=5, sample=1e-5, docvecs_mapfile=args.mmap)

    doc2vec.save(args.output, separately=None)
