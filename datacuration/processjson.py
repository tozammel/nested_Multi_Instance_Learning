#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def load_json(fpath, flat=False):
    with open(fpath, 'rb') as fh:
        decoded = json.loads(fh.read().decode('utf-8-sig'))
        return decoded if flat else decoded['values']


def traverse_json_keys(ajson_obj, level=0):
    for k, v in ajson_obj.items():
        print('-' * level, k, sep="")
        if isinstance(v, dict):
            traverse_json_keys(v, level+1)
