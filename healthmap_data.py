#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd

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


def main(argv):
    datapath = "/Users/tozammel/safe/data/healthMap_GSS/data"


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
