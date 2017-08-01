#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


###############################################################################
# Safe
##############################################################################


def load_counts_by_actors(filepath=None):
    if filepath is None:
        filepath = "data/actor_activity_by_day_th_300.csv"
    actors_df = pd.read_csv(filepath, header=0, index_col=0)
    actors_df.index = pd.to_datetime(actors_df.index)
    return actors_df


def get_isis():
    actors_df = load_counts_by_actors()
    col = "Islamic State / ISIS / ISIL / IS / Daesh"
    ts = actors_df[col]
    return ts


###############################################################################
# Ransomware
##############################################################################


class Ransomware:
    data = None

    def __init__(self, datafile=None):
        pass


def load_time_series(data_source):
    ts = None
    es_query = None
    if data_source == 'ransomware_locky':
        print("Fetching data from elastic search")
        print("\tData source:", options.data_source)
        ts, es_query = dp.get_ransom_data_elastic_search(
            windowsize=options.window_size, malwaretype="Locky")
    elif data_source == 'ransomware_cerber':
        print("Fetching data from elastic search")
        print("\tData source:", options.data_source)
        ts, es_query = dp.get_ransom_data_elastic_search(
            windowsize=options.window_size, malwaretype="Cerber")
    else:
        print("Error: Valid options are not given")
    return ts, es_query


def ransom_gt(filepath='./data/usenix/ground_truth.json'):
    registration_dates = []
    with open(filepath) as fh:
        lines = fh.read().splitlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            ajson = json.loads(line, encoding='utf-8')
            # print(ajson.keys())
            # print("----")
            # print(ajson['whois']['registered'])
            if ajson['whois']['registered'] != "N/A":
                date = pd.to_datetime(ajson['whois']['registered']).date()
                # print(date)
                registration_dates.append(date)

    # print(registration_dates[100])
    df = pd.DataFrame(registration_dates, columns=['date'])
    # df['date'] = pd.to_datetime(df['date']).date()
    ts = df.groupby('date').size()
    date_range = pd.date_range(ts.index.min(), ts.index.max())
    ts = ts.reindex(date_range, fill_value=0)
    return ts


def get_ransom_dataframe(datafile='ransomware/ransomware.csv',
                         malware_type=None, start_date=None,
                         end_date=None):
    df = pd.read_csv(datafile, header=0)
    df['Firstseen'] = pd.to_datetime(df['Firstseen'],
                                     format="%Y-%m-%d", utc=True)
    df['Date'] = df['Firstseen'].dt.date
    df.index = df['Date']

    # last three date entries: 2012-02-27, 2015-03-02, 2015-06-18
    df = df[:-3]
    return df


def get_ransom_series(datafile='ransomware/ransomware.csv',
                      malware_type=None, start_date=None,
                      end_date=None):
    df = pd.read_csv(datafile, header=0, encoding='latin1')
    df['Firstseen'] = pd.to_datetime(df['Firstseen'],
                                     format="%Y-%m-%d", utc=True)
    df['Date'] = df['Firstseen'].dt.date
    df.index = df['Date']

    # last three date entries: 2012-02-27, 2015-03-02, 2015-06-18
    df = df[:-3]

    if malware_type is None:
        df_malware = df
    else:
        # get only the 'locky' malwares
        df_malware = df[df['Malware'] == malware_type]

    ts_malware = df_malware.groupby('Date').size()
    date_range = pd.date_range(df.index.min(), df.index.max())
    ts_malware = ts_malware.reindex(date_range, fill_value=0)

    # subsetting
    if start_date is None:
        start_date = ts_malware.index.min()
    else:
        start_date = pd.to_datetime(start_date).date()

    if end_date is None:
        end_date = ts_malware.index.max()
    else:
        end_date = pd.to_datetime(end_date).date()

    ts_malware = ts_malware[start_date:end_date]
    return ts_malware


def get_ransom_locky(datafile='ransomware/ransomware_locky.csv'):
    return get_ransom_series(malware_type='Locky', start_day='2016-02-16')


def get_ransom_cerber(datafile='ransomware/ransomware_cerber.csv'):
    # return get_ransom_data(malware_type='Cerber', start_day='2016-06-21')
    return get_ransom_series(malware_type='Cerber', start_day='2016-06-16')


def load_data(filepath):
    data = pd.read_csv(filepath, header=0)
    # data['Firstseen'] = pd.to_datetime(data['Firstseen'], format="%Y-%m-%d %H:%M%S")
    data['Firstseen'] = pd.to_datetime(data['Firstseen'], format="%Y-%m-%d",
                                       utc=True)
    return data


def get_threat_types(data):
    return list(set(data['Threat']))


def get_malware_types(data):
    return list(set(data['Malware']))


def get_status_types(data):
    return list(set(data['Status']))


def get_list_of_countries(data):
    return list(set(data['Country']))


def get_counts_by_threat(data):
    return data.groupby('Threat').size()


def get_counts_by_malware(data):
    return data.groupby('Malware').size()


def get_counts_by_status(data):
    return data.groupby('Status').size()


# ------------------------ Chrome CVE --------------------------------


###############################################################################
# CVE
##############################################################################


def get_chrome_cve_data(datafile='google_chrome_CVE-time-sort.txt'):
    print("Data file = ", datafile)
    df = pd.read_csv(datafile, header=-1)
    df.columns = ['datetime']
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    ts = df.groupby('date').size()
    ts.sort_index(inplace=True)  # you don't need this operation, gby handle it

    print(ts.shape)
    print(ts.index[-1] - ts.index[0])
    print("min = ", min(ts.index), "\t max =", max(ts.index))
    # print(ts.head())
    idx = pd.date_range(min(ts.index), max(ts.index))
    print(idx.shape, idx[0], idx[-1])
    # ts.index = pd.DatetimeIndex(idx)
    ts = ts.reindex(idx, fill_value=0)
    print(ts.shape, "min val =", min(ts), "max val =", max(ts),
          "min indx =", min(ts.index), "max indx =", max(ts.index))

    return ts


###############################################################################
# Hackmageddon
##############################################################################

def get_hackmageddon_data(datafile='hackmageddon-cleaned.jl'):
    print("Data file = ", datafile)

    sel_column_names = ["date", "attack"]
    rows = []
    with open(datafile) as f:
        lines = f.read().splitlines()
        # print len(lines)
        for line in lines:
            line = line.strip()
            ajson = json.loads(line, encoding='utf-8')
            # print ajson.keys()
            # print ajson['date']
            # print ajson['attack']
            if ajson['date'] != "":
                frmtdate = pd.datetime.strptime(ajson['date'], "%d/%m/%Y")
                rows.append((frmtdate, ajson['attack']))
    # print rows
    df = pd.DataFrame(rows, columns=sel_column_names)
    # print(df.head())
    # print(df.dtypes)
    # indx_df = df.set_index(['date'])
    # print(indx_df.head())
    ts = df.groupby(['date']).size()
    ts.sort_index(inplace=True)

    ts.drop(ts.index[[0]], inplace=True)
    print(ts.shape)
    print(ts.index[-1] - ts.index[0])
    print("min = ", min(ts.index), "\t max =", max(ts.index))
    # print(ts.head())
    idx = pd.date_range(min(ts.index), max(ts.index))
    print(idx.shape, idx[0], idx[-1])
    # ts.index = pd.DatetimeIndex(idx)
    ts = ts.reindex(idx, fill_value=0)
    print(ts.shape, "min val =", min(ts), "max val =", max(ts),
          "min indx =", min(ts.index), "max indx =", max(ts.index))

    return ts


def get_hack_ddos_data(datafile='hackmageddon-cleaned.jl'):
    print("Data file = ", datafile)

    sel_column_names = ["date", "attack"]
    rows = []
    with open(datafile) as f:
        lines = f.read().splitlines()
        # print len(lines)
        for line in lines:
            line = line.strip()
            ajson = json.loads(line, encoding='utf-8')
            # print ajson.keys()
            # print ajson['date']
            # print ajson['attack']
            if ajson['date'] != "":
                frmtdate = pd.datetime.strptime(ajson['date'], "%d/%m/%Y")
                rows.append((frmtdate, ajson['attack']))
    # print rows
    df = pd.DataFrame(rows, columns=sel_column_names)
    print(df.shape)
    df = df[df['attack'] == 'DDoS']

    # input("press a key")

    # print(df.head())
    # print(df.dtypes)
    # indx_df = df.set_index(['date'])
    # print(indx_df.head())
    ts = df.groupby(['date']).size()
    ts.sort_index(inplace=True)
    print(ts.head())
    # input("press a key")

    print("min = ", min(ts.index), "\t max =", max(ts.index))
    print(ts.shape)
    print(ts.index[-1] - ts.index[0])
    print("min = ", min(ts.index), "\t max =", max(ts.index))
    # print(ts.head())
    idx = pd.date_range(min(ts.index), max(ts.index))
    print(idx.shape, idx[0], idx[-1])
    # ts.index = pd.DatetimeIndex(idx)
    ts = ts.reindex(idx, fill_value=0)
    print(ts.shape, "min val =", min(ts), "max val =", max(ts),
          "min indx =", min(ts.index), "max indx =", max(ts.index))

    return ts


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--data-file',
        default='hackmageddon-cleaned.jl',
        help='Files containing counts')
    options = parser.parse_args()
    # get_hackmageddon_data(options.data_file)

    # ground truth for usenix paper
    # ts = ransom_gt()
    # print(ts.head())


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
