import os
import argparse
import logging
import pandas as pd
from helper_classes.HTMLprocessing import ProcessSiteHTML, AggregateShops, ProcessDataframes


def main(certdir, fakedir):
    a = AggregateShops(do_whois=False)
    data = a.add_all_sites([certdir, fakedir])

    p = ProcessDataframes()
    X_pos, pos, vectorizers = p.make_train_test(data, full_run=False)

    feature_names = []
    for key, values in vectorizers.items():
        for f in vectorizers[key].get_feature_names():
            feature_names.append(key + '_' + f)
    df = pd.DataFrame(data=X_pos, index=pos.index, columns=feature_names)
    df['url']=df.index
    df = df.reset_index(drop=True)
    df.to_csv('features.csv', index=False)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    Logger = logging.getLogger('main.stdout')

    Args = argparse.ArgumentParser(description="Generation of eCommerce Sites Features")
    Args.add_argument("-f", "--fakedir", default="/home/mal2/fakeshop-detector-mal2/mal2_spider/results/fake_shops", help="Path to directory containing fraudulent sites")
    Args.add_argument("-c", "--certdir", default="/home/mal2/fakeshop-detector-mal2/mal2_spider/results/certified_shops", help="Path to directory containing safe sites")
    args = Args.parse_args()
    Logger.debug("FakeDir: {}, CertDir: {}".format(args.fakedir, args.certdir))

    main(args.certdir, args.fakedir)


