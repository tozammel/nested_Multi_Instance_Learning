import os

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def generate_latex_tables(df, filepath=None):
    from pylatex import Document, Tabular
    from pylatex.utils import bold

    doc = Document()
    # print(df.columns)
    # print(df.index)
    # print(df)
    header = df.index
    header_str = header.insert(0, 'Measure')
    # header_frmt ="X[l]"
    # header_frmt += " X[c]" * len(header)
    # print(header_str)
    header_frmt = "l"
    header_frmt += "c" * len(header)
    table = Tabular(header_frmt)
    table.add_hline()
    table.add_row(header_str, mapper=[bold])
    table.add_hline()

    # rows = []
    for col in df.columns:
        acol = df[col]
        minval = acol.min()
        minval_loc = acol.argmin()
        # print(minval_loc)
        row = [col]
        for idx, val in acol.iteritems():
            # if idx == minval_loc:
            if val == minval:
                # print("*", val)
                row.append(bold('%.2f' % val))
            else:
                # print(val)
                row.append('%.2f' % val)
        # rows.append(row)
        table.add_row(row)
    table.add_hline()

    # with doc.create(Center()) as centered:
    #     with centered.create(Tabu(header_frmt, spread="1in")) as data_table:
    #         # header_row1 = ["X", "Y"]
    #         header_row1 = header_str
    #         data_table.add_row(header_row1, mapper=[bold])
    #         data_table.add_hline()
    #         # row = [randint(0, 1000), randint(0, 1000)]
    #         for row in rows:
    #             data_table.add_row(row)

    doc.append(table)
    if filepath is not None:
        print("Saving:", os.path.splitext(filepath)[0])
        doc.generate_pdf(os.path.splitext(filepath)[0], clean_tex=False)
    return doc


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config-file',
                        default="data/time_series/summary/",
                        help='path to a csv file')
    options = parser.parse_args()


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
