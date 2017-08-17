import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from nltk.stem.snowball import *
from nielsenstemmer import *
import re
from random import shuffle

stemmer = SnowballStemmer('english')


def partition_data(source_docs_loc):
    full_df = pd.read_pickle(source_docs_loc)
    pattern = '\\b[A-Za-z]+\\b'
    arabic_texts = list()
    english_texts = list()
    for index, row in full_df.iterrows():
        unprocessed_text_data = row["text_data"]
        if not unprocessed_text_data is None:
            arabic_text_data = stem(unprocessed_text_data, transliteration=False) #nielsenstemmer for Arabic
            if len(arabic_text_data) > 50:
                row["text_data"] = arabic_text_data
                new_data = list(row.values.flatten())
                arabic_texts.append(new_data)
            english_text_data = ' '.join([stemmer.stem(w) for w in re.findall(pattern, unprocessed_text_data)])
            if len(english_text_data) > 50:
                row["text_data"] = english_text_data
                english_texts.append(row.values.flatten())
    arabic_df = pd.DataFrame(arabic_texts, columns=list(full_df.columns.values))
    english_df = pd.DataFrame(english_texts, columns=list(full_df.columns.values))
    return arabic_df, english_df
    
def build_labels(df, fields):
    # assumes that we're doing classification among GSR docs
    return None

def main():
    gsr_source_f = str(sys.argv[1])
    arabic_doc_df, english_doc_df = partition_data(gsr_source_f)    
    arabic_doc_df.to_pickle("processed_arabic_neg_samples.pkl")
    english_doc_df.to_pickle("processed_english_neg_samples.pkl")
    
    # arabic_labeled_data = build_binary_labels(arabic_doc_df, negative_samples)
    # english_labeled_data = build_binary_labels(english_doc_df, negative_samples)
    

if __name__ == '__main__':
    main()
