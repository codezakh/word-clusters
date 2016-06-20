from __future__ import print_function, division
import json
import itertools
import functools

import spacy

class RawStream:
    def __init__(self, fd):
        self._file_desc = fd
        self.io_count = 0


    def __iter__(self):
        for line in self._file_desc:
            doc = json.loads(line)
            self.io += 1
            yield doc



if __name__ == '__main__':
    nlp = spacy.en.English()
    with open('.', 'r') as fd:
        stream = RawStream(fd)
        for doc in nlp.pipe(stream, n_threads=4):
            pass
