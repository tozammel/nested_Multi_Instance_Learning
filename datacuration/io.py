#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def get_files_in_dir(dirpath):
    filepaths = []
    for dirName, subdirList, fileList in os.walk(dirpath):
        # print('Found directory: %s' % dirName)
        # print(subdirList)
        for fname in fileList:
            filepaths.append(os.path.join(dirpath, fname))
    return filepaths


def load_filepaths(dirpath):
    for dirName, subdirList, fileList in os.walk(dirpath):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            print('\t%s' % fname)


# get every .csv file under PATH you can use PATH + '/**/*.csv'
def load_filepaths_w_ext(dirpath, ext):
    return [file for file in
            glob.glob(dirpath + '/**/*.' + ext, recursive=True)]


def main(args):
    dirpath = "../data/arabiainform/feature-data"
    # filelist = load_filepaths(dirpath)
    filelist = load_filepaths_w_ext(dirpath, "csv")
    print(filelist)


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
