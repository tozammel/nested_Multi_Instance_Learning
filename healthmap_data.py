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

    for filepath in gss_filepaths[0:10]:
        print("Loading:", filepath)
        gss_data = dp.load_json(filepath, flat=True)
        # pj.traverse_json_keys(gss_data)
        # print(len(gss_data['data']))
        for article in gss_data['data'][0:1]:
            pj.traverse_json_keys(article)
            # gsr_data.extend(gss_data)


def analyze_healthmap_data():
    datapath = "/Users/tozammel/safe/data/healthMap_GSS/data"
    load_healthmap_data(datapath)


def main(argv):
    analyze_healthmap_data()


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
