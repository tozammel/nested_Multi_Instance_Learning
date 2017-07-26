#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import json
import pandas as pd
from collections import defaultdict

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def show_an_instance(instance):
    # print("Type =", type(instance))
    # print("Features =\n", instance.keys())
    # 'eventType', 'id', 'protest' (T/F: class var), 'link',
    # 'location' (list), 'words' (list),
    # 'date', 'doc2vec' (list), 'population'
    # print(type(instance['doc2vec']))

    print("Features:")
    print('protest =', instance['protest'])
    print('eventType =', instance['eventType'])
    print('id =', instance['id'])
    print('date =', instance['date'])
    print('link =', instance['link'])
    print('population =', instance['population'])
    print('location =', instance['location'])
    print(len(instance['words']))
    print(len(instance['doc2vec']))
    print("")


def explore_sample_data(filepath):
    from collections import defaultdict

    with open(filepath) as fp:
        lines = fp.read().splitlines()
        print("#lines =", len(lines))

        countries = defaultdict(int)
        class_var = defaultdict(int)
        dates = list()

        for line in lines:
            line = line.strip()
            # print(line)
            instance = json.loads(line)
            country = instance['location'][0]
            countries[country] += 1
            # show_an_instance(instance)
            y = instance['protest']
            class_var[y] += 1
            dates.append(pd.to_datetime(instance['time']).date())

        print(countries)
        print(sum(countries.values()))
        print(class_var)
        print(sum(class_var.values()))


def read_lines_as_json(filepath):
    json_list = list()
    with open(filepath) as fp:
        lines = fp.read().splitlines()
        for line in lines:
            line = line.strip()
            # print(line)
            instance = json.loads(line)
            json_list.append(instance)
    return json_list


def create_daily_bags():
    print("Creating daily bags...")

    filepath = "sample-data/news_doc2vec_ar.json"
    json_list = read_lines_as_json(filepath)
    daily_bags = defaultdict(list)

    for json_obj in json_list:
        # print(json_obj['date'])
        date = pd.to_datetime(json_obj['date']).date()
        daily_bags[date].append(json_obj)

    print(daily_bags.keys())
    daily_bags_size = {key: len([key]) for key in daily_bags.keys()}
    ts_bags = pd.Series(daily_bags_size)
    print("Start date =", ts_bags.index.min())
    print("End date =", ts_bags.index.max())
    print("Min num docs per day =", ts_bags.min())
    print("Max num docs per day =", ts_bags.max())
    print(ts_bags.head())
    print(ts_bags.tail())


def analyze_sample_traindata():
    # sample training instance
    filepath = "sample-data/nMIL_lt4_ar/top6cities_realtime_TrainData_2weekshistory.json"
    json_list = read_lines_as_json(filepath)
    print("#entries =", len(json_list))

    pos = 0
    neg = 0
    dates = list()
    for json_obj in json_list:
        # print(json_obj.keys())
        # print(json_obj['time'])
        dates.append(json_obj['time'])
        if json_obj['protest']:
            pos += 1
            print("\tpos, #keys =", json_obj.keys())
            # print("\tevent type =", json_obj['eventType'])
            # print("\tpopulation =", json_obj['population'])
        else:
            neg += 1
            print("neg, #keys =", json_obj.keys())

    print("#pos =", pos)
    print("#neg =", neg)

    df = pd.DataFrame(dates, columns=['date'])
    ts = df.groupby('date').size()
    print("Start date =", ts.index.min(), "End date =", ts.index.max())
    print("Min val =", ts.min(), "Max val =", ts.max())


def analyze_sample_news_doc2vec():
    # doc2vec representation
    filepath = "sample-data/news_doc2vec_ar.json"
    json_list = read_lines_as_json(filepath)

    pos = 0
    neg = 0
    for json_obj in json_list[0:10]:
        print(json_obj.keys())
        if json_obj['protest']:
            pos += 1
            print("\tpos, #keys =", json_obj.keys())
            print("\tevent type =", json_obj['eventType'])
            print("\tpopulation =", json_obj['population'])
        else:
            neg += 1
            print("neg, #keys =", json_obj.keys())

    print("#pos =", pos)
    print("#neg =", neg)


def main(argv):
    analyze_sample_traindata()
    # analyze_sample_news_doc2vec()
    # create_daily_bags()


def main2(argv):
    filepath = "sample-data/news_doc2vec_ar.json"
    """
    #rows = 10,384
    countries = {'Argentina': 10384}
    process_happened = {False: 9465, True: 919}
    """

    from collections import defaultdict
    countries = defaultdict(int)
    class_var = defaultdict(int)
    dates = list()

    with open(filepath) as fp:
        lines = fp.read().splitlines()
        print("#lines =", len(lines))

        for line in lines:
            line = line.strip()
            # print(line)
            instance = json.loads(line)
            country = instance['location'][0]
            countries[country] += 1
            # show_an_instance(instance)
            y = instance['protest']
            class_var[y] += 1
            dates.append(pd.to_datetime(instance['date']).date())

    print(countries)
    print(sum(countries.values()))
    print(class_var)
    print(sum(class_var.values()))

    print(type(dates[0]))
    df = pd.DataFrame(dates, columns=["date"])
    ts = df.groupby("date").size()

    print(ts.index.min())
    print(ts.index.max())
    print(ts.min())
    print(ts.max())


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
