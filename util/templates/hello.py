#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--data-file',
        default='',
        help='Files containing counts')
    options = parser.parse_args()
    print(options)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
