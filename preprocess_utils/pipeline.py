from __future__ import print_function 
import json
import itertools
import argparse
import logging
import sys
import collections
import contextlib
import codecs
import functools

import spacy
import gensim

from preprocess_utils import approp_doc, walk_dependencies

class RawStream:
    """Stream data from an iterable file like object, with logging.
    
    Parameters
    ------------
    fd : file object like
        A file like object with a context manager.
    
    iteration_hook : function
        A function which carries out optional modifications to the elements
        return from the stream.

    Examples
    ------------
    >>> import json
    >>> with open('example.json', 'a') as f:
    >>>     f.write(json.dumps({'a':1, 'b':2})+'\n')
    >>> with open('example.json', 'r') as f:
    >>>     stream = RawStream(fd=f, iteration_hook= lambda x: x['b'])
    >>>     for item in stream:
    >>>         #do something
    """

    def __init__(self, fd=None, iteration_hook=None, **kwargs):
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
        with codecs.open('err_dump.txt', 'a',encoding='utf-8') as err_dump:
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
        with codecs.open(self.flush_loc, 'a',encoding='utf-8',
                buffering=-1) as fd:
            for item in self.container:
                fd.write(item+'\n')
        self.container.clear()
    def __iter__(self):
        for item in self.container:
            yield item

    def __getitem__(self, key):
        return self.container[key]

    def append(self, item):
        if len(self.container)==self.buf_size:
            self.flush()
        self.container.append(item)
    def __repr__(self):
        return 'StreamBuffer:({buf_size},{container_size},{flush_loc})'.format(
                buf_size=self.buf_size,
                container_size=len(self.container),
                flush_loc = self.flush_loc)

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.flush()

def make_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Run a spacy pipeline for parsing a json corpus.')
    parser.add_argument('--data',
            help='Location of data.',
            required=True)
    parser.add_argument('--run-limit', type=int,
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
            action='store_true')
    parser.add_argument('--bufsize',
            type=int,
            default=10000,
            help='Internal size of the SpaCy and StreamBuffers.')
    parser.add_argument('--id2word-save',
            help='Location to store the id2word.',
            default='./vocab.bin')
    return parser

class Pipeline:
    def __init__(self,flush_loc=None, run_limit=None, debug=False,
            stream=None, pipe=None, preprocess_hook=None, **kwargs):
        self.streambuffer = StreamBuffer(flush_loc=flush_loc)
        self.debug = debug
        self.run_limit = run_limit
        self.stream = stream
        self.pipe = pipe
        self.preprocess_hook = preprocess_hook
        self.kwargs = kwargs

    def __call__(self):
        with self.streambuffer as streambuffer:
            for idx, doc in enumerate(self.pipe(self.stream,n_threads=4)):
                streambuffer.append(
                        self.preprocess_hook(idx=idx, doc=doc, **kwargs))




if __name__ == '__main__':
    logging.basicConfig(
            format='%(levelname)s:%(name)s:%(funcName)s:%(message)s',
            level=logging.DEBUG,
            filename='pipeline.log',
            filemode='w')
    parser = make_parser()
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    nlp = spacy.en.English()

    with open(args.data, mode='r',buffering=-1) as fd, \
            StreamBuffer(buf_size=args.bufsize ,flush_loc=args.output) as \
            streambuffer:
        stream = RawStream(fd,iteration_hook=args.hook)
        logger.info(stream)
        logger.info(streambuffer)
        id2word = gensim.corpora.Dictionary(prune_at=None)


        for idx,doc in enumerate(nlp.pipe(stream, n_threads=8,
            batch_size=args.bufsize)):
            try:
                bow_doc_extended = list(approp_doc(
                        itertools.chain.from_iterable(walk_dependencies(doc))))
                _ = id2word.doc2bow(bow_doc_extended, allow_update=True)
                streambuffer.append(
                        json.dumps(
                            {'bow':bow_doc_extended,
                                'idx':idx}
                            ))
            except Exception as error:
                #may not be the best place to handle errors that occur
                #here
                stream.fault_handler(error, doc)

            if args.debug:
                print(idx, '\r')

            if idx == args.run_limit:
                logger.info('run_limit reached {}'.format(idx))
                break

            elif idx % 5000 == 0:
                logger.info('{idx}:{stream}|{streambuffer}'.format(
                    idx=idx, stream=stream, streambuffer=streambuffer))

        id2word.save(args.id2word_save)

