#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import elasticsearch as es
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import json
import re
from nielsenstemmer import stem

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

"""
elastic search template:
    first level key: query, size, aggs

"""
def get_connection_to_es(host, port):
    client = es.Elasticsearch([{'host': host, 'port': port}])
    return client

# <editor-fold desc="util methods"> 
def get_term_list(file_name):
    term_list = list()
    with open(file_name, 'rb') as fo:
        file_data = fo.readlines()
        for line in file_data:
            try:
                encoded_line = line.decode('utf-8')
                if len(encoded_line.strip()) > 0:
                    term_list.append(encoded_line.strip())
            except UnicodeDecodeError as e:
                print(e)
                continue
    return term_list


def get_dict(file_name):
    with open(file_name, 'r', encoding='utf-8-sig') as fo:
        lookup_dict = json.load(fo)
    arabic_term_list = list()
    english_term_list = list()
    for key in lookup_dict:
        en_seg = re.sub('[^A-Za-z]+', '', key)
        ar_seg = text = stem(key, transliteration=False)
        if len(en_seg) > 0:
            english_term_list.append(key)
        if len(ar_seg) > 0:
            arabic_term_list.append(key)
    return lookup_dict, english_term_list, arabic_term_list

# </editor-fold> 

# <editor-fold desc="time series"> 
def ts_daily_doc_count(host, port, index, outdir):
    es_client = get_connection_to_es(host, port)
    # get es connection
    if not es_client.indices.exists(index):
        print("Bad index name. Exiting.")
        sys.exit(1)

    body = {
        "query": {
            "match_all": {
            }
        },
        "size": 0,
        "aggs": {
            "daily_event_count": {
                "date_histogram": {
                    "field": "document_date_utc_iso",
                    "interval": "day"
                }
            }

        }
    }

    ret = es_client.search(index=index, body=body)
    print("total num of entries =", ret["hits"]["total"])
    # print(ret["aggregations"])

    df = pd.DataFrame(ret["aggregations"]["daily_event_count"]["buckets"])
    # print(df.head())
    df.rename(columns={"key_as_string": "date", "doc_count": "count"},
              inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    ts = df['count']
    print("Min date =", ts.index.min())
    print("Max date =", ts.index.max())
    ts = ts.reindex(index=pd.date_range(ts.index.min(),
        ts.index.max()), fill_value=0)
    ts.index.name = 'date'

    return ts

# </editor-fold> 

def get_a_single_doc_instance(host, port, index, outdir):
    es_client = get_connection_to_es(host, port)


def daily_docs(host, port, index, outdir):
    es_client = get_connection_to_es(host, port)
    # get es connection
    if not es_client.indices.exists(index):
        print("Bad index name. Exiting.")
        sys.exit(1)

    body = {
        "query": {
            "match_all": {
            }
        },
        "size": 2
    }
    ret = es_client.search(index=index, body=body)
    entries = ret["hits"]["hits"]
    print("total num of entries =", ret["hits"]["total"])

    for entry in entries:
        print(entry.keys())


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='isicvl03',
        help='Elastic search hostname')
    parser.add_argument('--port', default=8150,
        help='Elastic search port')
    parser.add_argument('--index', default='arabiainform',
        help='Elastic search index')
    parser.add_argument('--doc-type', default='doc',
        help='Elastic search index')
    parser.add_argument('--outdir', default='.',
        help='Elastic search index')
    options = parser.parse_args()
    print(options)

    # get es connection
    es_client = get_connection_to_es(options.host, options.port)
    if not es_client.indices.exists(options.index):
        print("Bad index name. Exiting.")
        sys.exit(1)

    # experiments
    outdie = options.outdir
    ts = ts_daily_doc_count(host, port, index, outdir)
    filepath = os.path.join(outdir, "ts_all_doc_count.csv") 
    print("Saving:", filepath)
    ts.to_csv(filepath, header=True)
    filepath = os.path.join(outdir, "ts_all_doc_count.png") 
    print("Saving:", filepath)
    ts.plot().get_figure().savefig(filepath)
    # ts.plot().clf()


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv)) 
