#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.parsing import PorterStemmer
from gensim.models import Word2Vec

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

"""
https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/

"""

global_stemmer = PorterStemmer()


# <editor-fold desc="Stemmer">
class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """

    # This reverse lookup will remember the original forms of the stemmed
    # words
    word_lookup = {}

    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """

        # Stem the word
        stemmed = global_stemmer.stem(word)

        # Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)

        return stemmed

    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """

        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word
# </editor-fold>


def get_wiki_text(showtext=True):
    from wikipedia import page
    from wikipedia import search
    title = "Machine learning"
    wikipage = page(title)

    if showtext:
        print("Title:", wikipage.title)
        print("Summary:", wikipage.summary)
        print("Content:", wikipage.content)

    # incase you don’t have the exact title and want to do a search
    # titles = search('machine learning')
    # wikipage = page(titles[0])

    return wikipage


def process_text(text):
    # A ‘term’ could be individual words like ‘machine’, or phrases(n-grams)
    # like ‘machine learning’, or a combination of both.
    # remove all special characters and short lines from the article, to
    # eliminate noise.
    # Stopwords: http://www.ranks.nl/stopwords
    # (seems to be cool: http://www.ranks.nl/home)

    # stopwords: https://pydigger.com/pypi/many-stop-words
    # same repo in pypi: https://pypi.python.org/pypi/many-stop-words
    # this repo used ranks.nl

    import many_stop_words as stop_words



    pass


def learn_word2vec(sentences, size=50, window=4, min_count=2):
    # ref: https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/

    # parameters
    # 1. size:  If you have read the document and have an idea of how many
    # ‘topics’ it has, you can use that number. A good heuristic that is
    # frequently used is the square-root of the length of the vocabulary,
    # after pre-processing.
    # 2. min_count: Terms that occur less than min_count number of times are
    # ignored in the calculations.
    # 3. window – Only terms hat occur within a window-neighbourhood of a term,
    # in a sentence, are associated with it during training.
    # 4. sg: If equal to 1, the skip-gram technique is used. Else, the CBoW
    # method is employed.

    # initialize the model and use it
    model = Word2Vec(sentences, min_count=min_count, size=size, window=window)

    # access all the terms in its vocabulary
    vocab = list(model.vocab.keys())
    print(vocab[:10])

    # get the vectorial representation of a particular term
    print(model['learn'])

    # figure out the terms most similar to a particular one
    print(model.most_similar(StemmingHelper.stem('classification')))
    # model.similarity(StemmingHelper.stem('classification'), 'supervis')
    # model.similarity('unsupervis', 'supervis')


def main(argv):
    import time
    start_time = time.time()

    # Obtain the text
    text = get_wiki_text()
    # print(text)
    print(type(text))

    # process text
    sentences = process_text()

    # learn_word2vec(sentences)

    print("\n\n--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
