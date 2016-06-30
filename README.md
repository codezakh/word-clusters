A simple pipeline for streaming large corpora from disk that uses SpaCy
and gensim to train a doc2vec model.

Input Format
--------------
The input can be any sort of object which can be represented as a generator
or iterator in Python. Each element of the input stream should be deserializable
by `json.loads()`. A custom hook can be passed in order to manipulate the resulting
Python object before yielding the output to SpaCy.


CLI Interface
--------------
A CLI interface is included along with the modules, however it is opinionated and
was designed for a specific use and may not be fit for general purpose use. If
extensive customization is not needed, then the CLI tools may be sufficent.


Memory Usage
--------------
The library and CLI tools are designed to be constant memory for arbitrarily
large corpora - however, the `gensim.corpora.Dictionary` object which holds
the vocabulary of the pipeline is not memory mapped. This is easy to modify by
supplying the optional parameter to a path that gensim will use as a memory mapped
file. 


Hardcoded Constants
--------------------
Several constants have been hardcoded in instead of being configurable - and
this should change in the future. For now, the important ones to note are:

1. Number of cores `spacy.en.English().pipe` uses.
2. Number of workers `gensim.models.Doc2Vec` uses.
3. Location of the output stream buffers io. This is hardcoded to a be a file.
