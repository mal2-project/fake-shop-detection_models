import argparse
from helper_classes.DatabaseConnection import DatabaseConnection
from helper_classes.TableProcessing import TableProcessing


def main(table_name, create_table=False, drop_table=False, delete_record=None, populate_csv=None, populate_manual=None,
         remove_duplicates=False, get_score=None):

    dc = DatabaseConnection()
    cur, conn = dc.connect()
    tp = TableProcessing(table_name, cur, conn)
    score = None
    if create_table:
        tp.create_table()
    if drop_table:
        tp.drop_table()
    if populate_csv:
        tp.populate_table_fromcsv(populate_csv)
    if populate_manual:
        tp.populate_table_manual(populate_manual[0], populate_manual[1])
    if delete_record:
        tp.delete_site_manual(delete_record)
    if remove_duplicates:
        tp.remove_duplicates()
    if get_score:
        score = tp.get_score(get_score)
    dc.disconnect(cur, conn)
    return score

if __name__ == '__main__':
    #****************Remember not to use quotation marks for any of the values you pass here****************************
    Args = argparse.ArgumentParser(description="Training of eCommerce Sites Model")
    Args.add_argument("table_name", help="Provide table name for all following commands")
    Args.add_argument("-ct", "--create_table", action='store_true', help="Create an empty new table")
    Args.add_argument("-dt", "--drop_table", action='store_true', help="Create an empty new table")
    Args.add_argument("-dr", "--delete_record", help="Delete a record from table")
    Args.add_argument("-pcsv", "--populate_csv", help="Upload csv to an existing table")
    # C:/Users/dinicao/Documents/mal2/eCommerce/cleanup_analysis_scripts/certified_sites.csv
    Args.add_argument("-pman", "--populate_manual", nargs=2, help="Add a record to an existing table")
    Args.add_argument("-rd", "--remove_duplicates", action='store_true', help="Remove duplicates from table")
    Args.add_argument("-s", "--get_score", help="Get score of site if exists")
    args = Args.parse_args()

    score = main(**vars(args))
    print(score)
