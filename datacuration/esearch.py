#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from elasticsearch import Elasticsearch

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def get_connection_to_es():
    return Elasticsearch(
        ['http://ec2-52-42-169-124.us-west-2.compute.amazonaws.com/es/'],
        http_auth=('effect', 'c@use!23'), port=80)


def get_body(start_date, end_date, malwaretype, start_indx=0, payload_size=10):
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "name": malwaretype
                        }
                    },
                    {
                        "range": {
                            "observedDate": {
                                "gte": start_date.strftime("%d/%m/%Y"),
                                "lte": end_date.strftime("%d/%m/%Y"),
                                "format": "dd/MM/yyyy"
                            }
                        }
                    }
                ]
            }
        },
        "from": start_indx,
        "size": payload_size
    }
    return body


def get_ransom_data_elastic_search(windowsize=50, malwaretype="Cerber"):
    min_num_data_points = 100
    payload_size = 100
    max_timeout_count = 3
    data_search_window = 100
    if data_search_window < windowsize:
        data_search_window = data_search_window + windowsize

    es = get_connection_to_es()

    timenow = pd.datetime.now()
    end_date = timenow.date()
    start_date = end_date - pd.Timedelta(days=data_search_window - 1)
    # print("Start date =", start_date)
    # print("End date =", end_date)

    query = get_body(start_date, end_date, malwaretype)
    source_type = 'malware'
    results = es.search(index="effect/" + source_type, body=query)

    hits = results['hits']
    # print(hits.keys())
    num_entries = hits['total']
    # print("Total num of entries =", num_entries)

    # print("Number of hits =", len(hits['hits']))
    # for hit in hits['hits']:
    #     for key in hit.keys():
    #         print(key, ' -> ', hit[key])
    #         print()
    # return results['hits']

    malware_df = None
    if num_entries < min_num_data_points:
        print("Error: there are less than " + str(min_num_data_points) +
              " data points over a window of " + str(windowsize) + " days.")
        # generate a warning of low intensity
    else:
        i = 0
        payloads = []
        timeout_count = 0
        while i < num_entries:
            res = results = es.search(body=get_body(
                start_date, end_date, malwaretype, start_indx=i,
                payload_size=payload_size))
            if res['timed_out'] == False:
                payloads.append(res)
                i = i + payload_size
                timeout_count = 0
            else:
                timeout_count = timeout_count + 1
                if timeout_count == max_timeout_count:
                    sys.exit("Max time out occurred while fetching data from \
                             from Elastic Search. Program terminated")

        # print("Number of payloads =", len(payloads))
        malware_df = process_data(payloads)
        ts_malware = malware_df.groupby('observed_date').size()

        learn_end_date = ts_malware.index.max()
        learn_start_date = learn_end_date - pd.Timedelta(days=windowsize - 1)
        mask = (ts_malware.index >= learn_start_date) & (
            ts_malware.index <= learn_end_date)
        ts_malware_window = ts_malware[mask]
        date_range = pd.date_range(learn_start_date, learn_end_date)
        ts_malware_window = ts_malware_window.reindex(date_range, fill_value=0)
    return ts_malware_window, query


def process_data(payloads):
    data = []
    for payload in payloads:
        # print(payload.keys())
        # dict_keys(['took', 'timed_out', '_shards', 'hits'])
        hits = payload["hits"]["hits"]  # list object
        for ahit in hits:
            # print(ahit.keys())
            # dict_keys(['_index', '_type', '_id', '_score', '_source'])
            src = ahit["_source"]
            # print(src.keys())
            # ['a', 'url', 'observedDate', 'name', 'uri']
            # ['a', 'name', 'countryOfOrigin', 'hostedAt', 'uri', 'url',
            # 'observedDate']
            data.append(pd.to_datetime(src["observedDate"]).date())
            # data.append((src["observedDate"], src["countryOfOrigin"],
            #              src["url"], src["hostedAt"]["name"]))

    # colnames = ["observedDate", "countryOfOrigin", "url", "name"]
    colnames = ["observed_date"]
    return pd.DataFrame(data, columns=colnames)
