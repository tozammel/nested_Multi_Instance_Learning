#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from datacuration import io
from datacuration import dataprocessing as dp
from datacuration import processjson as pj
from datacuration import nielsenstemmer as nstemmer

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

"""
Health map data format:
-status
-data
--0
---ymdhs (int string)
---cr  (time stamp)
---rdt (time stamp)
---url
---dom (arabnews.com)
---title ()
---description (raw text: english, arabic, image)
---authors (a list)
----0:<str>
---tags (a list)
---custom_m (list)
---place_list (list)
"""


def load_healthmap_json_data(gssdir):
    gss_filepaths = io.load_filepaths_w_ext(gssdir, 'json', save_df=False)
    # TODO: process small number of files
    gss_filepaths = gss_filepaths[:10]
    print("Number of json files =", len(gss_filepaths))
    data = list()
    places = list()
    for idx, filepath in enumerate(gss_filepaths):
        print("Loading:", filepath, "(", idx, ")")
        gss_data = dp.load_json(filepath, flat=True)
        # pj.traverse_json_keys(gss_data)
        # print(len(gss_data['data']))
        for article in gss_data['data']:
            # pj.traverse_json_keys(article)
            # print(article)
            if 'cr' in article:
                creation_date = pd.to_datetime(article['cr']).date()
                if 'rdt' in article:
                    rdt_date = pd.to_datetime(article['rdt']).date()
                else:
                    rdt_date = None
                # data.append([creation_date, rdt_date, article['dom'], article['url'],
                #              article['title'], article['description']])
                data.append([creation_date, article['dom'], article['url'],
                             article['title'], article['description']])

    # columns = ['cr_date', 'rdt_date, ''dom', 'url', 'title', 'description']
    columns = ['cr_date', 'dom', 'url', 'title', 'description']
    print('#articles =', len(data))
    # print(data)
    df = pd.DataFrame(data, columns=columns)

    if save_df:
        outpath = os.path.join("data", "healthmap_gss")
        os.makedirs(outpath, exist_ok=True)
        outpath = os.path.join(outpath, "all_gss.csv")
        print("Saving:", outpath)
        df.to_csv(outpath, index=False)
        # print(df.head())
    return df


def analyze_raw_healthmap_data():
    datapath = "/Users/tozammel/safe/data/healthMap_GSS/data"
    load_healthmap_json_data(datapath)


def analyze_healthmap_csv_data():
    datapath = "data/healthmap_gss/all_gss.csv"
    gss = pd.read_csv(datapath, header=0)
    # print(gss.head())
    print("Shape =", gss.shape)

    start_date = gss['cr_date'].min()
    end_date = gss['cr_date'].max()

    print("From ", start_date, " to ", end_date)

    ts = gss.groupby('cr_date').size()
    ts.name = 'doc_count'
    ts.index.name = 'date'
    outfile = os.path.join("data", "healthmap_gss", "all_gss_ts.csv")
    os.makedirs(os.path.split(outfile)[0], exist_ok=True)
    print("Saving:", outfile)
    ts.to_csv(outfile, header=True)

    outfile = os.path.join("data", "healthmap_gss", "all_gss_ts.png")
    print("Saving:", outfile)
    ts.plot().get_figure().savefig(outfile)

    ts2 = gss.groupby('rdt_date').size()
    ts2.name = 'doc_count'
    ts2.index.name = 'date'
    outfile = os.path.join("data", "healthmap_gss", "all_gss_ts_rdt.csv")
    os.makedirs(os.path.split(outfile)[0], exist_ok=True)
    print("Saving:", outfile)
    ts2.to_csv(outfile, header=True)

    outfile = os.path.join("data", "healthmap_gss", "all_gss_ts_rdt.png")
    print("Saving:", outfile)
    ts2.plot().get_figure().savefig(outfile)


def analyze_text_csv_data():
    datapath = "data/healthmap_gss/all_gss.csv"
    gss = pd.read_csv(datapath, header=0)
    gss = gss[0:2]

    for row in gss.iterrows():
        print(type(row))
        print(row)


def main(argv):
    import time
    start_time = time.time()
    # analyze_raw_healthmap_data()
    # analyze_healthmap_csv_data()
    analyze_text_csv_data()
    print("\n\n--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
