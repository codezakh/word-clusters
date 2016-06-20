from __future__ import print_function, division
import json
import itertools
import functools

import spacy

class RawStream:
    def __init__(self, fd, iteration_hook=None, **kwargs):
        self._file_desc = fd
        self.io_count = 0
        self.iteration_hook = iteration_hook


    def __iter__(self):
        for line in self._file_desc:
            doc = json.loads(line)
            self.io_count += 1
            if self.iteration_hook is not None:
                yield self.iteration_hook(doc, **kwargs)
            else:
                yield doc



if __name__ == '__main__':
    nlp = spacy.en.English()
    with open('.', 'r') as fd:
        stream = RawStream(fd)
        for doc in nlp.pipe(stream, n_threads=4):
            pass
