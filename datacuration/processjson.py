#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def load_json(fpath, flat=False):
    with open(fpath, 'rb') as fh:
        decoded = json.loads(fh.read().decode('utf-8-sig'))
        return decoded if flat else decoded['values']
