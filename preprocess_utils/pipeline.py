from __future__ import print_function 
import json
import itertools
import argparse
import logging
import sys
import collections
import contextlib

import spacy
import gensim

from preprocess_utils import approp_doc, walk_dependencies

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
        return 'RawStream:({cnt},{fd},{hook_given})'.format(
                cnt=self.io_count,
                fd=self._file_desc,
                hook_given=self.iteration_hook is None)


    def fault_handler(self, error, doc):
        with open('err_dump.txt', 'a') as err_dump:
            err_dump.write(unicode(doc)+'\n')
        self.logger.error('thrown at {cnt}:{error}'.format(
            cnt=self.io_count,
            error=error))

class StreamBuffer:
    def __init__(self, buf_size=1000, flush_loc=None):
        self.container = collections.deque()
        self.buf_size = buf_size
        self.flush_loc = flush_loc
    def flush(self):
        with open(self.flush_loc, 'a',buffering=-1) as fd:
            for item in self.container:
                fd.write(item+'\n')
        self.container.clear()
    def append(self, item):
        if len(self.container)==self.buf_size:
            self.flush()
        self.container.append(item)
    def __repr__(self):
        return 'StreamBuffer:({buf_size},{container_size},{flush_loc})'.format(
                buf_size=self.buf_size,
                container_size=len(self.container),
                flush_loc = self.flush_loc)

    def close(self):
        self.flush()

def make_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Run a spacy pipeline for parsing a json corpus.')
    parser.add_argument('--data',
            help='Location of data.',
            required=True)
    parser.add_argument('--run_limit', type=int,
            help='Number of iterations to run for. If not specified, runs \
            until `StopIteration` is raised.')
    parser.add_argument('--hook', type=eval,
            help='A short hook to modify each document before being passed \
                    to spacy.',
            default='lambda i: i')
    parser.add_argument('--output',
            type=str,
            help='Location to dump output.',
            default='./dump')
    parser.add_argument('--debug',
            type='store_true')
    return parser


if __name__ == '__main__':
    logging.basicConfig(
            format='%(levelname)s %(asctime)s %(funcName)s %(message)s',
            level=logging.DEBUG)
    parser = make_parser()
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    nlp = spacy.en.English()

    with open(args.data, mode='r',buffering=-1) as fd, \
    contextlib.closing(StreamBuffer(flush_loc=args.output)) as streambuffer:
        stream = RawStream(fd,iteration_hook=args.hook)
        logger.info(stream)
        logger.info(streambuffer)


        for doc in nlp.pipe(stream, n_threads=4):
            try:
                streambuffer.append(
                        json.dumps(
                            {'bow':list(approp_doc(doc)),
                                'idx':stream.io_count}
                            ))
            except Exception as error:
                stream.fault_handler(error, doc)

            if args.debug:
                print(stream.io_count, '\r')

            if stream.io_count == args.run_limit:
                logger.info('run_limit reached {}'.format(args.run_limit))
                break

            elif stream.io_count % 1000 == 0:
                logger.info(stream)
                logger.info(streambuffer)
