import os, glob, time
#windows specific
#mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.3.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
#os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

from bs4 import BeautifulSoup, Comment
import pandas as pd
import whois, random

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, roc_curve, det_curve
from sklearn.manifold import TSNE

from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns


class ProcessSiteHTML:
    '''Process a single html site'''

    # todo extract css, js inline

    def __init__(self, folder, index):
        ''' Read soup and call functions that process soups '''
        self.folder = folder
        site = folder + index
        self.index = site.split("/")[-2]
        self.df = pd.DataFrame(index=[self.index])
        with open(site, 'r', errors='replace') as f:
            lines = f.readlines()
        with open('temp', 'w', encoding='utf-8') as f:
            for l in lines:
                if "HTTrack" not in l:
                    f.write(l)
        with open('temp', 'r', errors='replace') as f:
            self.soup = BeautifulSoup(f.read(), "html.parser")
            # add functions that append new features here
            self.soup2html_tags()
            #self.soup2comment()
            self.soup2ngram()
            #self.soup2children()
            #self.full_text_html()
            self.css_in_path()
            self.js_in_path()
        os.remove('temp')

    def css_in_path(self):
        # appends text of every single css file
        glob_list = []

        #FIXME: css should only be contained in cssjs sub-folder (mal2-model) or main folder (mal2-scraper) - other is legacy
        #Note: scraped css files often contain a '?version=' tag at the end i.e. not recognized by extension *.css but require *.css*
        for file in ['*.css*', 'images/*.css', 'cssjs/*.css*']:
            glob_list.extend(glob.glob(self.folder + file))

        css = [open(f, errors='replace').read() for f in glob_list]
        if css == []:
            self.df['css'] = np.nan
        else:
            self.df['css'] = " ".join(css)


    def js_in_path(self):
        # appends text of every single js file
        glob_list = []

        #FIXME: js should only be contained in cssjs sub-folder (mal2-model) or main folder (mal2-scraper) - other is legacy
        for file in ['*.js', 'images/*.js', 'cssjs/*.js*']:
            glob_list.extend(glob.glob(self.folder + file))
        
        js = [open(f, errors='replace').read() for f in glob_list]
        if js == []:
            self.df['js'] = np.nan
        else:
            self.df['js'] = " ".join(js)

    def soup2html_tags(self):
        ''' Return DataFrame with list of all html tagnames, attributes and values '''
        keys = []
        values = []
        names = []

        for tag in self.soup.find_all():
            for key, value in tag.attrs.items():
                keys.append(key)
                if isinstance(value, (list)) & (len(value) > 0):
                    values.append(value[0])
                elif isinstance(value, (list)) & (len(value) == 0):
                    values.append('')
                else:
                    values.append(value)
                names.append(tag.name)

        def join_list(mylist):
            return " ".join(mylist)

        if keys == [] or values == [] or names == []:
            df = (pd.DataFrame({'key': [None], 'value': [None], 'tagname': [None]}))
        else:
            df = (pd.DataFrame({'key': keys, 'value': values, 'tagname': names}))

        df1 = (df.pivot_table(columns='tagname', values='key', aggfunc=join_list))
        df1.columns = 'key_' + df1.columns

        df1 = df1.reset_index(drop=True)
        df2 = (df.pivot_table(columns='tagname', values='value', aggfunc=join_list))
        df2.columns = 'value_' + df2.columns
        df2 = df2.reset_index(drop=True)
        df = pd.concat([df1, df2], axis=1, ignore_index=False)
        assert len(df) == 1
        df.index = [self.index]
        self.df = self.df.join(df, how='outer')

    def soup2comment(self):
        '''Append all comments in soup'''
        comments = self.soup.find_all(string=lambda text: isinstance(text, Comment))
        clean_comments = [x for x in comments if not 'Mirrored' in x]
        if clean_comments == []:
            self.df['comment'] = np.nan
        else:
            self.df['comment'] = " ".join(clean_comments)

    def soup2ngram(self):
        '''Represent edges of graph as string of names of there vertices, add all 3,4,5 edge connections'''
        trigram = []
        tetragram = []
        pentagram = []

        # try to append all multi edge connections between tags
        for tag in self.soup.find_all():
            for child in tag.children:
                try:
                    trigram.append(tag.parent.name + "_" + tag.name + "_" + child.name)
                except:
                    pass
                try:
                    tetragram.append(tag.parent.parent.name + '_' + tag.parent.name + "_" + tag.name + "_" + child.name)
                except:
                    pass
                try:
                    pentagram.append(
                        tag.parent.parent.parent.name + '_' + tag.parent.parent.name + '_' + tag.parent.name + "_" + tag.name + "_" + child.name)
                except:
                    pass

        if trigram == []:
            trigram = np.nan
        else:
            trigram = " ".join(trigram)
        if tetragram == []:
            tetragram = np.nan
        else:
            tetragram = " ".join(tetragram)
        if pentagram == []:
            pentagram = np.nan
        else:
            pentagram = " ".join(pentagram)

        self.df['trigram'] = trigram
        self.df['tetragram'] = tetragram
        self.df['pentagram'] = pentagram

    def soup2children(self):
        '''Represent edges of graph as as string of names of there vertices
        and add vertex.name + children.names'''
        relations = []
        bigrams = []
        for tag in self.soup.find_all():
            tagname = tag.name
            childnames = ("_".join([child.name for child in tag.contents if child.name is not None]))
            bigram = ([tag.name + "_" + child.name for child in tag.contents if child.name is not None])

            if childnames:
                relations.append(tagname + "_" + childnames)
            if bigram:
                bigrams.extend(bigram)

        relations = " ".join(relations)
        bigrams = " ".join(bigrams)

        self.df['relations'] = relations
        self.df['bigrams'] = bigrams

    def full_text_html(self):
        '''append full text html'''
        self.df['full_text_html'] = str(self.soup)

class AggregateShops:
    '''Aggregate all shops in a single pandas DataFrame'''

    # add external css, javascript

    def __init__(self, do_whois):
        self.do_whois = do_whois

    def add_all_sites(self, paths):
        '''walk path and append every new site'''
        df = pd.DataFrame()
        #0 for good sites, 1 for bad sites
        for status_id in range(len(paths)):
            print("adding all sites from: ",paths[status_id])
            for folder in glob.glob(paths[status_id] + "/*/"):
                try:
                    new_site = self.add_site(folder.replace('\\', '/'), status_id)
                    # New site is single row pandas DataFrame, site specific features can be added here
                    df = pd.concat([df, new_site], sort=False)
                except Exception as e:
                    print('Site was skipped. Error message:', e, '\n')
                    pass
        return df

    def add_site(self, folder, status_id):
        '''Process single site and add site specific features here'''
        try:
            glob_list = []
            for file in ['MAIN*.html', 'index.html']:
                glob_list.extend(glob.glob(folder + file))
            index_file = glob_list[0].replace('\\', '/').split('/')[-1]
        except:
            print('Site was skipped: index file named MAIN*.html or index.html not found in: {}.'.format(folder))
            return
        try:
            new_site = ProcessSiteHTML(folder, index_file).df
            if status_id is not None:
                new_site['status_id'] = status_id
            if self.do_whois:
                new_site = pd.concat([new_site, self.get_whois(folder)], axis=1)
            return new_site
        except:
            print('There is a problem with the html file in: {0}'.format(folder))
            return



    def get_whois(self, folder):
        '''take url and return DataFrame, number of queries is limited, use Research Network to bypass firewall '''
        domain = whois.query(folder.split('/')[-2])
        domain.__dict__['name_servers'] = (" ".join(list(domain.__dict__['name_servers'])))

        time.sleep(2)
        return pd.DataFrame(domain.__dict__, index=[0])

class ProcessDataframes:

    def docs2features_fit(self, docs):
        '''takes DataFrame with bag of word in cell and returns dictionary containing Tfidf and the vectorizers'''
        featuredict = {}
        vectorizers = {}
        for col in docs.columns:
            if col in ['status_id']: continue
            try:
                # remove words occurring in less than 4% of the documents
                vectorizers[col] = TfidfVectorizer(min_df=0.04)
                x = vectorizers[col].fit_transform(docs[col].values)
            except Exception as e:
                print(col, 'max_df=1.0, min_df=1 was used because:', e)
                vectorizers[col] = TfidfVectorizer()
                x = vectorizers[col].fit_transform(docs[col].values)

            featuredict[col] = x

        return featuredict, vectorizers

    def docs2features_transform(self, docs, vectorizers):
        '''Takes dataframe and vectorizers dict and transform to tf-idf matrix'''

        featuredict = {}

        for col in docs.columns:
            if col in ['status_id']: continue
            try:
                x = vectorizers[col].transform(docs[col].values)
                featuredict[col] = x
            except Exception as e:
                # due to split, some columns will be empty, ignore those
                if str(e) == "TfidfVectorizer - Vocabulary wasn't fitted.":
                    pass
                else:
                    print('{0} is in {1} but has not been processed during training in "vectorizers".'.format(e, docs.index[0]))
                    pass
        return featuredict

    def make_train_test(self, docs, full_run=True):
        docs = docs.reindex(sorted(docs.columns), axis=1)
        docs.replace(to_replace=[np.nan], value='NullValue', inplace=True)
        docs = docs.astype('str')
        # Split train and test set
        random.seed(1)
        train_idx = random.sample(list(docs.index), int(len(docs) * 7 / 10))
        train = (docs.loc[train_idx])
        train_pos = train.loc[train['status_id'] == '1']
        train_status_id = train.status_id
        train.drop('status_id', axis=1, inplace=True)
        test = (docs.loc[~docs.index.isin(train_idx)])
        test_pos = test.loc[test['status_id'] == '1']
        test_status_id = test.status_id
        test.drop('status_id', axis=1, inplace=True)
        print("Number of sites for training = %s" % len(train))
        print("Number of sites for test = %s \n" % len(test))

        featuredict_train, vectorizers = self.docs2features_fit(train)
        featuredict_test = self.docs2features_transform(test, vectorizers)
        pos = train_pos.append(test_pos)
        pos.drop('status_id', axis=1, inplace=True)
        featuredict_pos = self.docs2features_transform(pos, vectorizers)
        X_test = sp.hstack(list(featuredict_test.values())).toarray()  # NOT np.hstack
        X_train = sp.hstack(list(featuredict_train.values())).toarray() # NOT np.hstack
        X_pos = sp.hstack(list(featuredict_pos.values())).toarray()
        
        #sparsity = 1-np.count_nonzero(X_train_full, axis=0)/X_train_full.shape[0]
        #to_remove = np.where(sparsity > 0.95)[0].tolist()

        train_y = train_status_id.astype('int').to_numpy()
        test_y = test_status_id.astype('int').to_numpy()

        if full_run==True:
            return X_train, X_test, train_y, test_y, vectorizers
        else:
            return X_pos, pos, vectorizers

    def vectorize(self, df, vectorizers):
        df = df.astype('str')
        df = df.reindex(sorted(df.columns), axis=1)
        featuredict = self.docs2features_transform(df, vectorizers)
        features = list(featuredict.values())
        site_vector = sp.hstack(features).toarray()

        return site_vector


class BasicManager:
    
    def metrics(self, true_y, list_float_predictions, list_models):
        x = PrettyTable()
        x.field_names = ["model", "accuracy", "precision", "recall", "kappa", "f1_score", "tn", "fp", "fn", "tp"]
        for m in range(len(list_float_predictions)):
            predictions = [0 if p < 0.5 else 1 for p in list_float_predictions[m]]
            acc = accuracy_score(true_y, predictions)
            pre = precision_score(true_y, predictions)
            rec = recall_score(true_y, predictions)
            kappa = cohen_kappa_score(true_y, predictions)
            f1 = f1_score(true_y, predictions)
            if list(true_y) == predictions and sum(true_y) > 0:
                tn, fp, fn, tp = 0, 0, 0, len(true_y)
            elif list(true_y) == predictions and sum(true_y) == 0:
                tn, fp, fn, tp = len(true_y), 0, 0, 0
            else:
                tn, fp, fn, tp = confusion_matrix(true_y, predictions).ravel()

            x.add_row([list_models[m], acc, pre, rec, kappa, f1, tn, fp, fn, tp])

        with open('files/metrics.txt', 'w') as f:
            f.write(x.get_string())

        return x

    def roc_prob_plot(self, true_y, predictions, model_name):
        preds1 = []
        preds0 = []
        for i in range(len(true_y)):
            if true_y[i] == 0:
                preds0.append(predictions[i])
            else:
                preds1.append(predictions[i])
        f = plt.figure()
        plt.hist(preds1, bins=50, log=True, density=True, color="r", alpha=0.5, label="Fraudulent sites")
        plt.hist(preds0, bins=50, log=True, density=True, color="g", alpha=0.5, label="Certified sites")
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=0.7)
        plt.ylabel("Normalized Count")
        plt.xlabel("Prediction")
        plt.legend(loc='upper center')
        plt.savefig("files/%s_plot_roc_prob.png"%model_name)
        plt.close(f)


    def roc_plot(self, true_y, predictions, model_name):
        fpr, tpr, thresholds = roc_curve(true_y, predictions)
        f = plt.figure()
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.plot(fpr, tpr, color="r", linestyle='dotted', markersize=3)
        plt.savefig("files/%s_plot_roc.png"%model_name)
        plt.close(f)

    def det_plot(self, true_y, predictions, model_name):
        fpr, fnr, thresholds = det_curve(true_y, predictions)
        f = plt.figure()
        plt.ylabel("TPR")
        plt.xlabel("FNR")
        plt.plot(fpr, fnr, color="r", linestyle='dotted', markersize=3)
        plt.savefig("files/%s_plot_det.png"%model_name)
        plt.close(f)

    def tsne(self, x_train, x_test, colors_train, colors_test, model_name):
        #x = np.vstack((x_train.toarray(), x_test.toarray()))
        x = np.vstack((x_train, x_test))
        colors = np.hstack((colors_train, colors_test))
        num_classes = len(np.unique(colors))
        palette = np.array(sns.color_palette("hls", num_classes))
        x_embedded = TSNE().fit_transform(x)
        #c=-1
        #for x in range(len(x_embedded)):
        #    c+=1
        #    print(c, x_embedded[x], colors[x])
        f = plt.figure()
        plt.scatter(x_embedded[:,0], x_embedded[:,1], lw=0, s=20, alpha=0.7, c=palette[colors.astype(np.int)])
        plt.savefig("files/%s_plot_tsne.png"%model_name)
        plt.close(f)
