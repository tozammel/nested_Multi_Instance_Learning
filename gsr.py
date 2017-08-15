#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from datacuration import dataprocessing as dp
from datacuration import io
from datacuration import process_time_series as processts
from datacuration import processjson as pj
from plotting import tsplot
from plotting import custom_plot
from plotting import plot_config as pc

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

"""

Columns:
-------
[] Actor                             object  * 
> how many?  632
> Top 5 actor?
Russian Military                             4727
Iraqi Military                               7791
Islamic State / ISIS / ISIL / IS / Daesh    18139
Syrian Arab Military                        42563
Unspecified                                 49360


Actor_Status                      object
Approximate_Location              object  *
Casualties                        object
City                              object  *
[] Country                           object  *
> how many?  10
> top 5?
Turkey               2
Palestine            5
Bahrain             79
Jordan             153
Yemen              519
Saudi Arabia       550
Lebanon           1948
Egypt             4870
Iraq             33135
Syria           121485

Earliest_Reported_Date    datetime64[ns]
Encoding_Comment                  object
Event_Date                datetime64[ns]
Event_ID                          object
Event_Subtype                     object
Event_Type                        object
First_Reported_Link               object
GSS_Link                          object
Latitude                         float64
Longitude                        float64
News_Source                       object
Other_Links                       object
Revision_Date             datetime64[ns]
State                             object  *
Target                            object
Target_Name                       object  *
Target_Status                     object


Shape = (162746, 23)

Min event date = 2015-05-01 00:00:00
Max event date = 2017-03-31 00:00:00

Event types = ['Military Action' 'Non-State Actor']

Event_Type         #events
Military Action    96181
Non-State Actor    66565

"""


######################################################################
# Process Raw Data
######################################################################

def load_mansa_gsr(gsrdir=None, remove_duplicates=True):
    if gsrdir is None:
        gsrdir = "/Users/tozammel/safe/isi-code/mercury/data/gsr/mansa_gsr"

    gsr_data = []
    # mansa_gsr_filepaths = io.get_files_in_dir(gsrdir)
    mansa_gsr_filepaths = io.load_filepaths_w_ext(gsrdir, 'json')
    # print(mansa_gsr_filepaths)

    for filepath in mansa_gsr_filepaths:
        temp_gsr_data = dp.load_json(filepath, flat=True)
        gsr_data.extend(temp_gsr_data)

    mansa_gsr_df = pd.DataFrame(gsr_data)
    # print(mansa_gsr_df.dtypes)

    # change the dtypes of the some columns
    mansa_gsr_df['Event_Date'] = pd.to_datetime(mansa_gsr_df['Event_Date'])
    mansa_gsr_df['Earliest_Reported_Date'] = pd.to_datetime(
        mansa_gsr_df['Earliest_Reported_Date'])
    mansa_gsr_df['Revision_Date'] = pd.to_datetime(
        mansa_gsr_df['Revision_Date'])
    # NOTE: the column Casualties has str + int data. Not sure how to handle it
    # mansa_gsr_df['Casualities'] = mansa_gsr_df['Casualities'].astype(int)

    if remove_duplicates:
        mansa_gsr_df = mansa_gsr_df.sort_values(['Event_ID', 'Revision_Date']
                                                ).groupby('Event_ID').tail(1)
    return mansa_gsr_df


def ts_given_event_type(df=None, grs_dir=None, event_type="Military Action",
                        outdir="data/time_series", save_ts=False):
    if df is None:
        df = load_mansa_gsr(gsrdir=grs_dir)
    df_sub = df[df['Event_Type'] == event_type]
    ts = df_sub.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)
    ts.name = 'count'
    ts.index.name = 'date'
    print(event_type)
    print(ts.shape)

    if save_ts:
        filename = "gsr_" + event_type.lower().replace(" ", "_") + ".csv"
        filepath = os.path.join(outdir, filename)
        print("Saving ", filepath)
        ts.to_csv(filepath, header=True)
    return df_sub, ts


def ts_given_county(df=None, grs_dir=None, country="Syria",
                    outdir="data/time_series", save_ts=False):
    if df is None:
        df = load_mansa_gsr()
    df_sub = df[df['Country'] == country]
    ts = df_sub.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)
    ts.name = 'count'
    ts.index.name = 'date'
    print(country)
    print(ts.shape)

    if save_ts:
        filename = "gsr_" + country.lower().replace(" ", "_") + ".csv"
        filepath = os.path.join(outdir, filename)
        print("Saving ", filepath)
        ts.to_csv(filepath, header=True)

        filename = "gsr_" + country.lower().replace(" ", "_") + ".png"
        filepath = os.path.join(outdir, filename)
        print("Saving ", filepath)
        # plt = ts.plot()
        # plt.get_figure().savefig(filepath, format="png", bbox_inches="tight")
        # plt.clf()
        # plt, ax = pc.get_minimal_plot(width=10, height=6.2)
        # plt, ax = pc.get_minimal_plot(width=12, height=7.42)
        plt, ax = pc.get_minimal_plot(width=15, height=9.25)
        plt.plot(ts)
        plt.savefig(filepath, format="png", bbox_inches="tight")
        plt.xlabel("Day", fontsize=20)
        plt.ylabel("#Events", fontsize=20)

    return df_sub, ts


######################################################################
# Process Processed Data
######################################################################
def load_a_ts(ts_name="gsr_syria"):
    dirpath = "data/time_series"
    filepath = os.path.join(dirpath, ts_name + ".csv")
    return processts.load_a_time_series(filepath)


def load_tslist():
    dirpath = "data/time_series"
    filelist = io.load_filepaths_w_ext(dirpath, "csv")
    return processts.load_time_series(filelist)


######################################################################
# Testing methods
######################################################################
def analyze_mansa_gsr(event_start_date="2016-10-02",
                      event_end_date="2017-07-31"):
    df = load_mansa_gsr()
    print("Mansa GSR:")
    print("Shape =", df.shape)
    print(df.dtypes)

    print("Min event date =", df['Event_Date'].min())
    print("Max event date =", df['Event_Date'].max())

    df = df[(df['Event_Date'] >= pd.to_datetime(event_start_date)) & (
        df['Event_Date'] <= pd.to_datetime(event_end_date))]
    print("Shape =", df.shape)

    print()
    # print("Event types =", df['Event_Type'].unique())
    print("\n===================Event Type======================\n")
    evnt_type = df.groupby('Event_Type').size()
    print(evnt_type, "\n", '-' * 15, "\nTotal =", evnt_type.sum())

    print("\n===================Event Sub Type======================\n")
    evnt_subtype = df.groupby('Event_Subtype').size().sort_values(
        ascending=False)
    print(evnt_subtype, "\n", '-' * 15, "\nTotal =", evnt_subtype.sum())

    print("\n===================Country======================\n")
    gby_country = df.groupby('Country').size().sort_values(ascending=False)
    print(gby_country, "\n", '-' * 15, "\nTotal =", gby_country.sum())

    print("\n===================Country & Sate======================\n")
    gby_country_state = df.groupby(['Country', 'State']).size()
    print(gby_country_state, "\n", '-' * 15, "\nTotal =",
          gby_country_state.sum())


def identify_absence_of_events(event_start_date="2016-10-02",
                               event_end_date="2017-07-31"):
    df = load_mansa_gsr()
    df = df[(df['Event_Date'] >= pd.to_datetime(event_start_date)) & (
        df['Event_Date'] <= pd.to_datetime(event_end_date))]

    # country and event subtype
    #
    df = df[(df['Country'] == 'Iraq') &
            (df['Event_Subtype'] == 'Bombing')]

    # actor
    #
    df = df[(df['Country'] == 'Iraq') &
            (df['Event_Subtype'] == 'Bombing')]

    print(df.shape)
    ts = df.groupby('Event_Date').size()
    ts = ts.reindex(pd.date_range(event_start_date, "2017-06-30"), fill_value=0)
    ts.name = 'count'
    ts.index.name = 'date'
    # print(ts.head(10))

    ts_zero = ts[ts == 0]
    print("No events")
    print(ts_zero)


def analyze_duplicates(gsrdir=None):
    if gsrdir is None:
        gsrdir = "/Users/tozammel/safe/isi-code/mercury/data/gsr/mansa_gsr"

    gsr_data = []
    # mansa_gsr_filepaths = io.get_files_in_dir(gsrdir)
    mansa_gsr_filepaths = io.load_filepaths_w_ext(gsrdir, 'json')

    for filepath in mansa_gsr_filepaths:
        temp_gsr_data = pj.load_json(filepath, flat=True)
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
    # print(mansa_gsr_df.head())

    # print(mansa_gsr_df.shape)
    # print(mansa_gsr_df['Event_ID'].unique().size)
    # print(mansa_gsr_df['Event_Date'].min())
    # print(mansa_gsr_df['Event_Date'].max())

    grp_eid = mansa_gsr_df.sort_values(['Event_ID', 'Revision_Date']
                                       ).groupby('Event_ID').tail(1)

    # checking
    # grp_eid1 = mansa_gsr_df.groupby('Event_ID').size().sort_values(
    #     ascending=False)
    # print(grp_eid1.head(5))
    # print(mansa_gsr_df[mansa_gsr_df['Event_ID'] == 'MN66270'])
    # print()
    # print(grp_eid[grp_eid['Event_ID'] == 'MN66270'])


    # print(grp_eid.head(4))

    # print(mansa_gsr_df.dtypes)
    # mansa_gsr_df

    return mansa_gsr_df


def analyze_event_types(k=10):
    mansa_gsr_df = load_mansa_gsr()
    ts = mansa_gsr_df.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)

    mansa_gsr_grp_event_by_actor_country = \
        mansa_gsr_df.groupby(['Actor', 'Event_Type'])

    mansa_gsr_df_event_by_actor_country = \
        mansa_gsr_df.groupby(['Actor', 'Event_Type']).size()
    ret = mansa_gsr_df_event_by_actor_country.nlargest(k)
    print(ret)

    # event_type = 'Military Action'
    event_type = 'Non-State Actor'
    ts_given_event_type(event_type=event_type, save_ts=True)


def analyze_countries_old(country_list=None):
    if country_list is None:
        country_list = ['Syria', 'Iraq', 'Egypt', 'Lebanon', 'Saudi Arabia',
                        'Yemen', 'Jordan']
    [ts_given_county(country=country, save_ts=True) for country in country_list]


def analyze_event_types_with_countries():
    country_list = ['Syria', 'Iraq', 'Egypt', 'Lebanon']
    event_types = ['Military Action', 'Non-State Actor']
    outdir = "data/time_series/country_event_type"
    os.makedirs(outdir, exist_ok=True)

    for country in country_list:
        df_c, ts = ts_given_county(country=country, save_ts=False)
        print("country =", country)
        print(df_c.shape)
        for et in event_types:
            df_et, ts = ts_given_event_type(df=df_c, event_type=et,
                                            save_ts=False)
            print("\t", et)
            print("\t", df_et.shape)
            filename = "gsr_" + country.lower().replace(" ", "_") + "_" + \
                       et.lower().replace(" ", "_") + ".csv"
            filepath = os.path.join(outdir, filename)
            print("Saving ", filepath)
            ts.to_csv(filepath, header=True)


def analyze_countries(save_ts=True, k=7):
    outdir = "data/time_series/_summary"
    os.makedirs(outdir, exist_ok=True)

    mansa_gsr_df = load_mansa_gsr()
    mansa_gsr_df_event_by_country = \
        mansa_gsr_df.groupby('Country').size()
    ret = mansa_gsr_df_event_by_country.nlargest(k)
    print(ret)

    filename = "gsr_country_top_" + str(k) + ".csv"
    filepath = os.path.join(outdir, filename)
    print("Saving:", filepath)
    ret.to_csv(filepath)

    outdir = "data/time_series/_country"
    os.makedirs(outdir, exist_ok=True)

    ts = mansa_gsr_df.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)

    mansa_gsr_grp_event_by_country = \
        mansa_gsr_df.groupby('Country')

    for country, country_df in mansa_gsr_grp_event_by_country:
        if country in ret.index:
            print("")
            print(country)
            country_str = country.replace("/", "_")
            country_str = country_str.replace(" ", "_")
            country_str = country_str.replace(";", "_")
            print(country_str)
            ts = country_df.groupby('Event_Date').size()
            ts = ts.reindex(idx, fill_value=0)
            print(ts.sum())
            ts.name = 'count'
            ts.index.name = 'date'
            # print(country)
            # print(ts.shape)

            if save_ts:
                filename = "gsr_" + country_str.lower().replace(" ",
                                                                "_") + ".csv"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                ts.to_csv(filepath, header=True)

                filename = "gsr_" + country_str.lower().replace(" ",
                                                                "_") + ".png"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                # plt, ax = pc.get_minimal_plot(width=10, height=6.2)
                # plt, ax = pc.get_minimal_plot(width=12, height=7.42)
                plt, ax = pc.get_minimal_plot(width=15, height=9.25)
                plt.plot(ts)
                plt.savefig(filepath, format="png", bbox_inches="tight")
                plt.xlabel("Day", fontsize=20)
                plt.ylabel("#Events", fontsize=20)


def analyze_actors(save_ts=True, k=10):
    outdir = "data/time_series/_summary"
    os.makedirs(outdir, exist_ok=True)

    mansa_gsr_df = load_mansa_gsr()
    mansa_gsr_df_event_by_actor = \
        mansa_gsr_df.groupby('Actor').size()
    ret = mansa_gsr_df_event_by_actor.nlargest(k)
    print(ret)

    filename = "gsr_actor_top_" + str(k) + ".csv"
    filepath = os.path.join(outdir, filename)
    print("Saving:", filepath)
    ret.to_csv(filepath)

    outdir = "data/time_series/_actor"
    os.makedirs(outdir, exist_ok=True)
    ts = mansa_gsr_df.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)

    mansa_gsr_grp_event_by_actor = \
        mansa_gsr_df.groupby('Actor')

    for actor, actor_df in mansa_gsr_grp_event_by_actor:
        if actor in ret.index:
            print("")
            print(actor)
            actor_str = actor.replace("/", "_")
            actor_str = actor_str.replace(" ", "_")
            actor_str = actor_str.replace(";", "_")
            print(actor_str)
            ts = actor_df.groupby('Event_Date').size()
            ts = ts.reindex(idx, fill_value=0)
            print(ts.sum())
            ts.name = 'count'
            ts.index.name = 'date'
            # print(country)
            # print(ts.shape)

            if save_ts:
                filename = "gsr_" + actor_str.lower().replace(" ", "_") + ".csv"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                ts.to_csv(filepath, header=True)

                filename = "gsr_" + actor_str.lower().replace(" ", "_") + ".png"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                # plt = ts.plot()
                # plt.get_figure().savefig(filepath, format="png", bbox_inches="tight")
                # plt.clf()
                # plt, ax = pc.get_minimal_plot(width=10, height=6.2)
                # plt, ax = pc.get_minimal_plot(width=12, height=7.42)
                plt, ax = pc.get_minimal_plot(width=15, height=9.25)
                plt.plot(ts)
                plt.savefig(filepath, format="png", bbox_inches="tight")
                plt.xlabel("Day", fontsize=20)
                plt.ylabel("#Events", fontsize=20)


                # ret = dp.get_top_k_actors_by_counts()
                # print(ret)


def analyze_actors_with_countries(save_ts=True, k=10):
    outdir = "data/time_series/_summary"
    os.makedirs(outdir, exist_ok=True)

    mansa_gsr_df = load_mansa_gsr()
    mansa_gsr_df_event_by_actor_country = \
        mansa_gsr_df.groupby(['Actor', 'Country']).size()
    ret = mansa_gsr_df_event_by_actor_country.nlargest(k)
    print(ret)

    filename = "gsr_actor_country_top_" + str(k) + ".csv"
    filepath = os.path.join(outdir, filename)
    print("Saving:", filepath)
    ret.to_csv(filepath)

    ts = mansa_gsr_df.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)

    mansa_gsr_grp_event_by_actor_country = \
        mansa_gsr_df.groupby(['Actor', 'Country'])

    outdir = "data/time_series/_actor_country"
    os.makedirs(outdir, exist_ok=True)

    for indx, actor_df in mansa_gsr_grp_event_by_actor_country:
        if indx in ret.index:
            # print("")
            print(indx)
            actor_str = indx[0].replace(" / ", "_")
            actor_str = actor_str.replace(" ", "_")
            # actor_str = actor_str.replace(";", "_")
            country_str = indx[1].replace(" ", "_")
            actor_str += "__" + country_str

            print(actor_str)
            ts = actor_df.groupby('Event_Date').size()
            ts = ts.reindex(idx, fill_value=0)
            print(ts.sum())
            ts.name = 'count'
            ts.index.name = 'date'

            if save_ts:
                filename = "gsr_" + actor_str.lower().replace(" ", "_") + ".csv"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                ts.to_csv(filepath, header=True)

                filename = "gsr_" + actor_str.lower().replace(" ", "_") + ".png"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                plt, ax = pc.get_minimal_plot(width=15, height=9.25)
                plt.plot(ts)
                plt.savefig(filepath, format="png", bbox_inches="tight")
                plt.xlabel("Day", fontsize=20)
                plt.ylabel("#Events", fontsize=20)


def analyze_actors_with_city(save_ts=False, k=10):
    outdir = "data/time_series/_summary"
    os.makedirs(outdir, exist_ok=True)

    mansa_gsr_df = load_mansa_gsr()

    mansa_gsr_df_event_by_actor_country = \
        mansa_gsr_df.groupby(['Actor', 'City']).size()
    ret = mansa_gsr_df_event_by_actor_country.nlargest(k)
    print(ret)
    filename = "gsr_actor_city_top_" + str(k) + ".csv"
    filepath = os.path.join(outdir, filename)
    print("Saving:", filepath)
    ret.to_csv(filepath)

    ts = mansa_gsr_df.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)

    mansa_gsr_grp_event_by_actor_country = \
        mansa_gsr_df.groupby(['Actor', 'City'])

    outdir = "data/time_series/_actor_city"
    os.makedirs(outdir, exist_ok=True)

    for indx, actor_df in mansa_gsr_grp_event_by_actor_country:
        if indx in ret.index:
            # print("")
            print(indx)
            actor_str = indx[0].replace(" / ", "_")
            actor_str = actor_str.replace(" ", "_")
            # actor_str = actor_str.replace(";", "_")
            country_str = indx[1].replace(" ", "_")
            actor_str += "__" + country_str

            print(actor_str)
            ts = actor_df.groupby('Event_Date').size()
            ts = ts.reindex(idx, fill_value=0)
            print(ts.sum())
            ts.name = 'count'
            ts.index.name = 'date'

            if save_ts:
                filename = "gsr_" + actor_str.lower().replace(" ", "_") + ".csv"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                ts.to_csv(filepath, header=True)

                filename = "gsr_" + actor_str.lower().replace(" ", "_") + ".png"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                plt, ax = pc.get_minimal_plot(width=15, height=9.25)
                plt.plot(ts)
                plt.savefig(filepath, format="png", bbox_inches="tight")
                plt.xlabel("Day", fontsize=20)
                plt.ylabel("#Events", fontsize=20)


def analyze_actors_with_event_type(save_ts=False, k=10):
    outdir = "data/time_series/_summary"
    os.makedirs(outdir, exist_ok=True)

    mansa_gsr_df = load_mansa_gsr()

    mansa_gsr_df_event_by_actor_country = \
        mansa_gsr_df.groupby(['Actor', 'Event_Type']).size()
    ret = mansa_gsr_df_event_by_actor_country.nlargest(k)
    print(ret)
    filename = "gsr_actor_event_type_top_" + str(k) + ".csv"
    filepath = os.path.join(outdir, filename)
    print("Saving:", filepath)
    ret.to_csv(filepath)

    outdir = "data/time_series/_actor_event_type"
    os.makedirs(outdir, exist_ok=True)
    ts = mansa_gsr_df.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)

    mansa_gsr_grp_event_by_actor_country = \
        mansa_gsr_df.groupby(['Actor', 'Event_Type'])

    for indx, actor_df in mansa_gsr_grp_event_by_actor_country:
        if indx in ret.index:
            # print("")
            print(indx)
            actor_str = indx[0].replace(" / ", "_")
            actor_str = actor_str.replace(" ", "_")
            # actor_str = actor_str.replace(";", "_")
            country_str = indx[1].replace(" ", "_")
            actor_str += "__" + country_str

            print(actor_str)
            ts = actor_df.groupby('Event_Date').size()
            ts = ts.reindex(idx, fill_value=0)
            print(ts.sum())
            ts.name = 'count'
            ts.index.name = 'date'

            if save_ts:
                filename = "gsr_" + actor_str.lower().replace(" ", "_") + ".csv"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                ts.to_csv(filepath, header=True)

                filename = "gsr_" + actor_str.lower().replace(" ", "_") + ".png"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                plt, ax = pc.get_minimal_plot(width=15, height=9.25)
                plt.plot(ts)
                plt.savefig(filepath, format="png", bbox_inches="tight")
                plt.xlabel("Day", fontsize=20)
                plt.ylabel("#Events", fontsize=20)


def analyze_city(save_ts=True, k=10):
    outdir = "data/time_series/summary"
    os.makedirs(outdir, exist_ok=True)

    mansa_gsr_df = load_mansa_gsr()
    mansa_gsr_df_event_by_city = \
        mansa_gsr_df.groupby('City').size()
    ret = mansa_gsr_df_event_by_city.nlargest(k)
    print(ret)

    filename = "gsr_city_top_" + str(k) + ".csv"
    filepath = os.path.join(outdir, filename)
    print("Saving:", filepath)
    ret.to_csv(filepath)

    ts = mansa_gsr_df.groupby('Event_Date').size()
    idx = pd.date_range(min(ts.index), max(ts.index))
    ts = ts.reindex(idx, fill_value=0)
    print("Total number of events =", ts.sum())
    print("Columns =", mansa_gsr_df.columns)

    mansa_gsr_grp_event_by_city = \
        mansa_gsr_df.groupby('City')

    outdir = "data/time_series/city"
    os.makedirs(outdir, exist_ok=True)
    for city, city_df in mansa_gsr_grp_event_by_city:
        if city in ret.index:
            print("")
            print(city)
            city_str = city.replace("/", "_")
            city_str = city_str.replace(" ", "_")
            city_str = city_str.replace(";", "_")
            print(city_str)
            ts = city_df.groupby('Event_Date').size()
            ts = ts.reindex(idx, fill_value=0)
            print(ts.sum())
            ts.name = 'count'
            ts.index.name = 'date'
            # print(country)
            # print(ts.shape)

            if save_ts:
                filename = "gsr_" + city_str.lower().replace(" ", "_") + ".csv"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                ts.to_csv(filepath, header=True)

                filename = "gsr_" + city_str.lower().replace(" ", "_") + ".png"
                filepath = os.path.join(outdir, filename)
                print("Saving ", filepath)
                # plt = ts.plot()
                # plt.get_figure().savefig(filepath, format="png", bbox_inches="tight")
                # plt.clf()
                # plt, ax = pc.get_minimal_plot(width=10, height=6.2)
                # plt, ax = pc.get_minimal_plot(width=12, height=7.42)
                plt, ax = pc.get_minimal_plot(width=15, height=9.25)
                plt.plot(ts)
                plt.savefig(filepath, format="png", bbox_inches="tight")
                plt.xlabel("Day", fontsize=20)
                plt.ylabel("#Events", fontsize=20)


def plot_timeseries():
    from datacuration import config
    from datacuration.process_time_series import load_ts
    from plotting import custom_plot
    cfg = config.load_config("data/evaluation_config/mansa_config_01.json")
    outdir = os.path.join(cfg['output_config'][0]['out_dir'], "fig")
    os.makedirs(outdir, exist_ok=True)

    for dset in cfg["data_source"]:
        ts = load_ts(filepath=dset["endopath"], start_date=None)
        filename = dset['endoname'] + ".png"
        p = custom_plot.plot_ts(ts, xlabel="Day", ylabel="Num of Events")
        p.savefig(os.path.join(outdir, filename), format='png')


def main(args):
    # analyze_duplicates()
    # dirpath = "data/time_series"
    # filelist = io.load_filepaths_w_ext(dirpath, "csv")
    # print("Number of gsr ts =", len(filelist))
    # tslist = processts.load_time_series(filelist)
    # print(tslist[0].name)

    # analyze_mansa_gsr()
    # analyze_event_types()
    #
    # analyze_countries(save_ts=True)
    # analyze_event_types_with_countries()
    #
    # analyze_actors(save_ts=True)
    # analyze_actors_with_countries(save_ts=True)
    # analyze_actors_with_city(save_ts=True)
    # analyze_actors_with_event_type(save_ts=True)

    # analyze_city()

    # plot_timeseries()
    identify_absence_of_events()


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
