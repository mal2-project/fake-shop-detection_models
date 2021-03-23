import os
import argparse
import logging
import pickle
import datetime

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from helper_classes.HTMLprocessing import ProcessSiteHTML, AggregateShops, ProcessDataframes, BasicManager


class Model:

    def feature_importance(self, vectorizers, overall_feature_number):
        i = -1
        #key: column names from original pandas dataframe
        #values: an object <class 'sklearn.feature_extraction.text.TfidfVectorizer'>
        #f: individual "words" parsed by spaces after scraping
        for key, values in vectorizers.items():
            for f in range(len(vectorizers[key].get_feature_names())):
                i += 1
                if i == overall_feature_number:
                    return key, vectorizers[key].get_feature_names()[f]

    def xgboost(self, X_train_full, X_test_full, train_status_id, test_status_id, vectorizers, folder):
        
        print (">>> start","xgboost", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        dtrain = xgb.DMatrix(X_train_full, list(map(int, train_status_id)))
        dtest = xgb.DMatrix(X_test_full, list(map(int, test_status_id)))

        ProcessDataframes()
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        param = {'max_depth': 3,
                 'eta': 1,
                 'verbosity': 1,
                 'objective': 'binary:logistic'}
        #param = {}
        num_round = 100
        model = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=10)
        float_predictions = model.predict(dtest)
        print (">>> done","xgboost", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"\n")

        # save model to file
        pickle.dump(model, open("{}/xgboost.model".format(folder), "wb"))
        print('The model has been saved in the {} directory. Filename: "xgboost.model"'.format(folder))

        print('The following are the ([feature type], [feature value]) pairs that were found to be of importance in xgboost:')
        if os.path.exists("files/xgboost_feature-importance.txt"):
            os.remove("files/xgboost_feature-importance.txt")
        with open("files/xgboost_feature-importance.txt", "a") as myfile:
            for count,key in enumerate(model.get_score(importance_type='weight')):
                key, value = self.feature_importance(vectorizers, int(key[1:]))
                if count <= 10:
                    print("('%s', '%s')"%(key,value))
                if count == 0:
                    myfile.write("The following are the ([feature type], [feature value]) pairs that were found to be of importance in xgboost:\n")
                myfile.write("('%s', '%s')\n"%(key,value))

        return float_predictions

    def sklearn(self, X_train_full, X_test_full, train_status_id, algorithm, folder):
        
        print (">>> start", algorithm, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if algorithm == 'single_tree':
            model = DecisionTreeClassifier()
            model.fit(X_train_full, train_status_id)
            float_predictions = model.predict(X_test_full)
        elif algorithm == 'random_forest':
            model = RandomForestClassifier()
            model.fit(X_train_full, train_status_id)
            float_predictions = model.predict(X_test_full)
        elif algorithm == 'neural_net':
            model = MLPClassifier(alpha=1)
            model.fit(X_train_full, train_status_id)
            float_predictions = model.predict(X_test_full)

        print (">>> done", algorithm, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"\n")

        # save model to file
        pickle.dump(model, open("{0}/{1}.model".format(folder, algorithm), "wb"))
        print('The model has been saved in the {0} directory. Filename: "{1}.model"'.format(folder, algorithm))
        
        return float_predictions


def main(gooddir, baddir):
    a = AggregateShops(do_whois=False)
    data = a.add_all_sites([gooddir, baddir])

    p = ProcessDataframes()
    X_train_full, X_test_full, train_status_id, test_status_id, vectorizers = p.make_train_test(data)
    # save dictionary to file
    folder="files"
    os.makedirs(os.path.dirname(folder+'/'), exist_ok=True)
    pickle.dump(vectorizers, open("{}/vectorizers.dict".format(folder), "wb"))
    pickle.dump(X_train_full, open("{}/train_set.pkl".format(folder), "wb"))
    print('The feature dictionary has been saved in the {} directory. Filename: "vectorizers.dict"'.format(folder))

    m = Model()
    predictions = []
    models = ['single_tree', 'random_forest', 'neural_net']
    for mod in models:
        predictions.append(m.sklearn(X_train_full, X_test_full, train_status_id, mod, folder))
    predictions.append(m.xgboost(X_train_full, X_test_full, train_status_id, test_status_id, vectorizers, folder))
    models.append('xgboost')

    bm = BasicManager()
    print(bm.metrics(test_status_id, predictions, models))
    print('Metrics have been saved in the {} directory. Filename: "metrics.txt"'.format(folder))
    
    for i,model_name in enumerate(models):
        print('Generating ROC, DET, TSNE plots for ', model_name)
        # uncomment if you want roc plot
        bm.roc_plot(test_status_id, predictions[i], model_name)
        # uncomment if you want the roc probability plot
        bm.roc_prob_plot(test_status_id, predictions[i], model_name)
        # uncomment if you want det plot
        bm.det_plot(test_status_id, predictions[i], model_name)
        # uncomment if you want to run t-sne and get 2d clustered plot
        bm.tsne(X_train_full, X_test_full, train_status_id, test_status_id, model_name)

##############################################################################################

# Add whois query to DataFrame, at AIT Firewall blocks, use Research Network instead
# number of shops is higher than quota for nic.at ! delay is set to 2 seconds
# number of queries is limited

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    Logger = logging.getLogger('main.stdout')

    Args = argparse.ArgumentParser(description="Training of eCommerce Sites Model")
    Args.add_argument("-c", "--certdir", default="data/certified", help="Relative path to directory containing certified sites")
    Args.add_argument("-f", "--fakedir", default="data/fakeshops", help="Relative path to directory containing fraudulent sites")
    args = Args.parse_args()
    Logger.debug("FakeDir: {}, CertDir: {}".format(args.fakedir, args.certdir))

    main(args.certdir, args.fakedir)

