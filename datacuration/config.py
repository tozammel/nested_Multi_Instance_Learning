#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
# import commentjson as cjson

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def load_config(filepath):
    with open(filepath) as fh:
        return json.load(fh)


def parse_config_file(filepath):
    config = load_config(filepath)
    print(config.keys())
    for key in config.keys():
        print(config[key])
        print(type(config[key]))

