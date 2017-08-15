#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from datacuration import io
from datacuration import dataprocessing as dp
from datacuration import processjson as pj

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


def load_healthmap_data(gssdir):
    gss_filepaths = io.load_filepaths_w_ext(gssdir, 'json')
    print("Number of json files =", len(gss_filepaths))
    data = list()
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
                data.append([creation_date, article['dom'], article['url'],
                             article['description']])

    print('#articles =', len(data))
    # print(data)
    df = pd.DataFrame(data)
    print(df.head())


def analyze_healthmap_data():
    datapath = "/Users/tozammel/safe/data/healthMap_GSS/data"
    load_healthmap_data(datapath)



def main(argv):
    import time
    start_time = time.time()
    analyze_healthmap_data()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
