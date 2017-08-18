#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.parsing import PorterStemmer

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

"""
https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/

"""

global_stemmer = PorterStemmer()


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
    pass


def learn_word2vec():
    # ref: https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/

    # Obtain the text
    text = get_wiki_text()


def main(argv):
    import time
    start_time = time.time()
    # analyze_healthmap_data()
    # analyze_healthmap_csv_data()

    learn_word2vec()

    print("\n\n--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
