


def word2vec_wiki():
    # ref: https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/

    from wikipedia import page
    from wikipedia import search
    title = "Machine learning"
    wikipage = page(title)

    titles = search('machine learning')
    wikipage = page(titles[0])


def main(argv):
    import time
    start_time = time.time()
    # analyze_healthmap_data()
    # analyze_healthmap_csv_data()
    print("\n\n--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
