#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import pandas as pd

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


# ========================================================================
# utility
# ========================================================================


def get_files_in_dir(dirpath):
    filepaths = []
    for dirName, subdirList, fileList in os.walk(dirpath):
        # print('Found directory: %s' % dirName)
        # print(subdirList)
        for fname in fileList:
            filepaths.append(os.path.join(dirpath, fname))
    return filepaths


# ========================================================================
# Data Load
# ========================================================================


def load_json(fpath, flat=False):
    with open(fpath, 'rb') as fh:
        decoded = json.loads(fh.read().decode('utf-8-sig'))
        return decoded if flat else decoded['values']


def save_time_series(colname='Country', colval='Syria',
                     filepath="data/mansa/gsr_syria.csv"):
    """
    ['Actor', 'Approximate_Location', 'Casualties', 'City', 'Country',
       'Earliest_Reported_Date', 'Encoding_Comment', 'Event_Date', 'Event_ID',
       'Event_Subtype', 'Event_Type', 'First_Reported_Link', 'GSS_Link',
       'Latitude', 'Longitude', 'News_Source', 'Other_Links', 'Revision_Date',
       'State', 'Target', 'Target_Name', 'Target_Status'],
    :return:
    """
    # get time series with unique ids
    print("Loading json files")
    mansa_gsr_df = load_mansa_gsr_w_unique_eid()
    print(mansa_gsr_df.shape)
    print(mansa_gsr_df.columns)

    # os.makedirs("data/mansa", exist_ok=True)
    # filepath = "data/mansa/mansa.csv"
    # print("Saving:", filepath)
    # mansa_df.to_csv(filepath, index=False)
    mansa_gsr_df_subset = mansa_gsr_df[mansa_gsr_df[colname] == colval]

    ts = mansa_gsr_df_subset.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)
    ts.name = 'count'
    ts.index.name = 'date'
    print("Saving:", filepath)
    ts.to_csv(filepath, header=True)


def load_ts_from_csv(filepath, header_line=0):  # if there is no header put -1
    # ts_df = pd.read_csv(filepath, header=header_line)
    # ts_df.columns = ['date', 'count']
    # ts_df['date'] = pd.to_datetime(ts_df['date'])
    # ts_df.set_index(['date'], inplace=True)

    ts = pd.read_csv(filepath, header=header_line, parse_dates=True,
                     index_col=0, squeeze=True)
    # ts = ts_df['count']
    ts.name = 'count'
    ts.index.name = 'date'
    idx = pd.date_range(ts.index.min(), ts.index.max())
    ts = ts.reindex(idx, fill_value=0)
    return ts


def load_ts(filepath, start_date=None, end_date=None):
    print("Loading:", filepath)
    ts = pd.read_csv(filepath, header=0, index_col=0,
                     parse_dates=True, squeeze=True)

    print(ts.describe())
    print("Min date =", ts.index.min())
    print("Max date =", ts.index.max())
    # ts_cerber_dr = ts_cerber_dr.loc["2016-08-30":"2017-02-28"]

    if start_date is not None or end_date is not None:
        ts = ts[start_date:end_date]
    return ts


# ========================================================================
# Special Methods
# ========================================================================

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
    ts_malware.name = "count"
    ts_malware.index.name = "date"
    return ts_malware


def load_mansa_gsr(gsrdir=None):
    if gsrdir is None:
        gsr_data_dir = "/Users/tozammel/safe/mercury/data/gsr"
        mansa_gsr_data_dir = os.path.join(gsr_data_dir, "mansa_gsr")
        gsrdir = mansa_gsr_data_dir

    gsr_data = []
    mansa_gsr_filepaths = get_files_in_dir(gsrdir)
    # print(mansa_gsr_filepaths)
    for filepath in mansa_gsr_filepaths:
        temp_gsr_data = load_json(filepath, flat=True)
        gsr_data.extend(temp_gsr_data)

    mansa_gsr_df = pd.DataFrame(gsr_data)
    # change the dtypes of the some columns
    mansa_gsr_df['Event_Date'] = pd.to_datetime(mansa_gsr_df['Event_Date'])
    mansa_gsr_df['Earliest_Reported_Date'] = pd.to_datetime(
        mansa_gsr_df['Earliest_Reported_Date'])
    mansa_gsr_df['Revision_Date'] = pd.to_datetime(
        mansa_gsr_df['Revision_Date'])
    # NOTE: the column Casualties has str + int data. Not sure how to handle it
    # mansa_gsr_df['Casualities'] = mansa_gsr_df['Casualities'].astype(int)
    return mansa_gsr_df


def load_mansa_gsr_w_unique_eid():
    mansa_gsr_df = load_mansa_gsr()
    print("#Raw entries =", mansa_gsr_df.shape[0])
    mansa_gsr_df_by_unique_event_id = mansa_gsr_df.groupby(
        'Event_ID', group_keys=False, as_index=False).apply(
        lambda x: x.ix[x.Event_Date.idxmax()])
    print("#Unique entries =", mansa_gsr_df_by_unique_event_id.shape[0])
    return mansa_gsr_df_by_unique_event_id


def get_top_k_actors_by_counts(k=10):
    mansa_gsr_df_by_unique_event_id = load_mansa_gsr_w_unique_eid()
    ts = mansa_gsr_df_by_unique_event_id.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)

    mansa_gsr_df_event_by_actor = \
        mansa_gsr_df_by_unique_event_id.groupby('Actor').size()
    return mansa_gsr_df_event_by_actor.nlargest(k)


def get_actors_with_count_th(count_th):
    mansa_gsr_df_by_unique_event_id = load_mansa_gsr_w_unique_eid()
    ts = mansa_gsr_df_by_unique_event_id.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)

    mansa_gsr_df_event_by_actor = \
        mansa_gsr_df_by_unique_event_id.groupby('Actor').size()
    sel_actors_df = mansa_gsr_df_event_by_actor[
        mansa_gsr_df_event_by_actor >= count_th]
    return sel_actors_df


def load_counts_by_actors(filepath=None):
    if filepath is None:
        filepath = "data/actor_activity_by_day_th_300.csv"
    actors_df = pd.read_csv(filepath, header=0, index_col=0)
    actors_df.index = pd.to_datetime(actors_df.index)
    return actors_df


def load_counts_by_cities(filepath=None):
    if filepath is None:
        filepath = "data/city_activity_by_day_th_100.csv"
    cities_df = pd.read_csv(filepath, header=0, index_col=0)
    cities_df.index = pd.to_datetime(cities_df.index)
    return


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--data-file',
        default='',
        help='Files containing counts')
    options = parser.parse_args()

    # test functions
    mansa_gsr_df = load_mansa_gsr()
    print(mansa_gsr_df.dtypes)
    # print(mansa_gsr_df['Casualties'])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
