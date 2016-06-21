from __future__ import print_function 
import json
import itertools
import argparse
import logging
import sys

import spacy
import gensim
import numpy as np

from preprocess_utils import preprocess_utils as pt

class RawStream:
    def __init__(self, fd, iteration_hook=None, **kwargs):
        self._file_desc = fd
        self.io_count = 0
        self.iteration_hook = iteration_hook
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)


    def __iter__(self):
        for line in self._file_desc:
            doc = json.loads(line)
            self.io_count += 1
            if self.iteration_hook is not None:
                try:
                    yield self.iteration_hook(doc, **self.kwargs)
                except Exception as error:
                    self.fault_handler(error, doc)
                    continue
            else:
                yield doc

    def __repr__(self):
        return '({cnt},{fd},{hook_given})'.format(
                cnt=self.io_count,
                fd=self._file_desc,
                hook_given=self.iteration_hook is None)


    def fault_handler(self, error, doc):
        with open('err_dump.txt', 'w') as err_dump:
            err_dump.write(json.dumps(doc))
        self.logger.error('thrown at {cnt}:{error}'.format(
            cnt=self.io_count,
            error=error))

def make_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Run a spacy pipeline for parsing a json corpus.')
    parser.add_argument('--data',
            help='Location of data.',
            required=True)
    parser.add_argument('--run_limited', type=int,
            help='Number of iterations to run for. If not specified, runs \
            until `StopIteration` is raised.')
    parser.add_argument('--hook', type=eval,
            help='A short hook to modify each document before being passed \
                    to spacy.',
            default='lambda i: i')
    return parser


if __name__ == '__main__':
    logging.basicConfig(
            format='%(levelname)s %(asctime)s %(funcName)s %(message)s',
            level=logging.DEBUG)
    parser = make_parser()
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    nlp = spacy.en.English()
    with open(args.data, mode='r',buffering=-1) as fd:
        stream = RawStream(fd,iteration_hook=args.hook)
        logger.info(stream)
        for doc in nlp.pipe(stream, n_threads=4):
            list(pt.approp_doc(doc))
            if stream.io_count == args.run_limit:
                logger.info('run_limit reached {}'.format(args.run_limit))
                break
