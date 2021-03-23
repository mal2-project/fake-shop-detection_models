import logging, argparse, pickle, shutil
import platform

import numpy as np
import xgboost as xgb
import lime
import lime.lime_tabular
import shap
import os
import matplotlib.pyplot as plt

from helper_classes.HTMLprocessing import ProcessSiteHTML, AggregateShops, ProcessDataframes
from multiprocessing import Process, Queue
from twisted.internet import reactor
from scrapy.utils.project import get_project_settings
from scrapy_spider import spider
import scrapy.crawler as crawler
import site_database

settings = get_project_settings()
settings["COOKIES_ENABLED"] = False
settings["LOG_ENABLED"] = False
settings["ROBOTSTXT_OBEY"] = False
runner = crawler.CrawlerRunner(settings)
logging.getLogger('scrapy').propagate = False
            
def run_scrapy_spider(spider,input,output):
    # CrawlerProcess terminates after calling process.start due to termination of reactor framework
    # twisted.internet.error.ReactorNotRestartable @see
    # https://doc.scrapy.org/en/latest/topics/practices.html#running-multiple-spiders-in-the-same-process
    # not an issue when only called once as by shell - but as in python library
    def f(q):
        try:
            runner.settings['IMAGES_STORE'] = output
            runner.settings['FILES_STORE'] = output
            deferred = runner.crawl(spider, input=input, output=output)
            deferred.addBoth(lambda _: reactor.stop())
            reactor.run()
            q.put(None)
        except Exception as e:
            q.put(e)

    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    result = q.get()
    p.join()

    if result is not None:
        raise result

def feature_importance(model, vectorizers, site_vector, method, model_type):

    def xgb_predict_proba(data_x):
        tmp_out = model.predict(xgb.DMatrix(data_x))
        # add the first column to make it like predict_proba
        out = np.zeros((data_x.shape[0], 2))
        out[:, 0] = 1 - tmp_out
        out[:, 1] = tmp_out
        return out

    train = pickle.load(open('files/train_set.pkl', "rb"))
    feature_names = []
    for key, values in vectorizers.items():
        for f in vectorizers[key].get_feature_names():
            feature_names.append(key + ' | ' + f)
    print("numbers of overall available features in trainings set: ",len(feature_names))

    explanations=[]

    if 'lime' in method:
        if os.path.exists("files/explanation_lime.html"):
            os.remove("files/explanation_lime.html")
        
        explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=feature_names, discretize_continuous=False, class_names=['Safe', 'Fraudulent'])
        if model_type == "Booster":
            explanation = explainer.explain_instance(site_vector.flatten(), xgb_predict_proba)
        else:
            explanation = explainer.explain_instance(site_vector.flatten(), model.predict_proba)
        
        if platform.system() == 'Windows':
            explanation.save_to_file("files\\explanation_lime.html")
        else:
            explanation.save_to_file("files/explanation_lime.html")
        explanation = explanation.as_list()
        explanations.append(explanation)

    if 'shap' in method:   
        if os.path.exists("files/explanation_shap.png"):
            os.remove("files/explanation_shap.png")
        
        if model_type == "MLPClassifier":
            #FIXME: skipping explanation for Neural Net. TODO add shap.KernelExplainer for MLPClassifier 
            pass
        else:
            explainer = shap.TreeExplainer(model)
            explanation = explainer.shap_values(site_vector)
            if len(explanation) == 2: explanation = explanation[0]
            shap.summary_plot(explanation, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            if platform.system() == 'Windows':
                plt.savefig('files\\explanation_shap')
            else:
                plt.savefig('files/explanation_shap')
            explanations.append(explanation)

    return explanations

def main(modelloc, vectorizersloc, site, check_db, use_cache, do_feature_importance, do_scrape_images):

    def __nukedir_recursively(dir):
        """Force delete of directory and files recursively even if files/folders are still in use Arguments:
            dir {Str} -- folder path
        """
        if os.path.exists(dir):
            if dir[-1] == os.sep: dir = dir[:-1]
            files = os.listdir(dir)
            for file in files:
                if file == '.' or file == '..': continue
                path = dir + os.sep + file
                if os.path.isdir(path):
                    __nukedir_recursively(path)
                else:
                    os.unlink(path)
            os.rmdir(dir)

    model = pickle.load(open(modelloc, "rb"))
    vectorizers = pickle.load(open(vectorizersloc, "rb"))

    if check_db:
        score = site_database.main(table_name='sites_temp', get_score=site)  
    else:
        score = None
        
    verified = False
    if score is not None:
        verified = True
        if check_db and do_feature_importance:
            explanation = "**Currently feature importance is available only for URLs not already in the database.**"
        else:
            explanation = None
        return score, verified, explanation
    else:
        # read website names
        store_path = 'data/verify_sites/'
        results_dir = store_path + site + '/'

        #operate on cache or re-scrape?
        if use_cache and os.path.isdir(store_path + site + '/'):
            print('using local cache of scraped site')
        else:
            #no cache - remove old data
            try:
                __nukedir_recursively(results_dir)
            except Exception as e:
               print('Error while deleting directory:',results_dir, e)

            #download site via scrapy - blocking calls
            run_scrapy_spider(spider.HtmlSpider, input=site,output=store_path)
            run_scrapy_spider(spider.CssSpider,input=site,output=store_path)
            if do_scrape_images:
                run_scrapy_spider(spider.ImgSpider,input=site,output=store_path)
            
            #check output complete
            if os.path.exists(results_dir+ "index.html") and os.path.isdir(results_dir+ 'cssjs/'):
                print('scraping site is complete')
            else:
                print('failed scraping website')
                return None, None, None
           
        print("*" * 79)
        print("Prediction is being calculated ...")
        # load model and feature dictionary from file
        a = AggregateShops(do_whois=False)
        df_site = a.add_site(store_path + site + '/', status_id=None)

        for col in list(vectorizers):
            if col not in df_site.columns:
                df_site[col] = np.nan

        p = ProcessDataframes()
        site_vector = p.vectorize(df_site, vectorizers)
        model_type = type(model).__name__

        if model_type == "Booster":
            prediction = model.predict(xgb.DMatrix(site_vector))
        else:
            prediction = [model.predict_proba(site_vector)[0][1]]


        if do_feature_importance:
            print("Explanation is being calculated ...")
            explanations = feature_importance(model, vectorizers, site_vector, do_feature_importance, model_type)
        else:
            explanations = None

        return prediction, verified, explanations

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    Logger = logging.getLogger('main.stdout')

    Args = argparse.ArgumentParser(description="Verification of new eCommerce Site")
    Args.add_argument("-m", "--model", default="files/xgboost.model", help="Relative path to model")
    Args.add_argument("-v", "--vectorizers", default="files/vectorizers.dict", help="Relative path to feature dictionary")
    Args.add_argument("-u", "--url", default="akp-pflege.de", help="URL of site you want to verify")
    Args.add_argument("-f", "--feature-importance", default=None, dest='feature_importance', choices=['lime', 'shap'], nargs='+', help="Options are: lime, shap or no flag for None")
    Args.add_argument("--check-db", default=False, action="store_true", dest='check_db', help="Set flag if you want to check database first for site verification; otherwise do nothing")
    Args.add_argument("--use-cache", default=False, action="store_true", dest='use_cache', help="Set flag if you want to use locally cached version of scraped html files - re-scrapes only when site not available locally")
    Args.add_argument("--scrape-images", default=False, action="store_true", dest='scrape_images', help="Set flag if you want the crawler to also download all images")
    args = Args.parse_args()
    Logger.debug("model: {}, site: {}".format(args.model, args.vectorizers, args.url, args.check_db, args.use_cache, args.feature_importance))

    score, verified, explanation = main(args.model, args.vectorizers, args.url, args.check_db, args.use_cache, args.feature_importance, args.scrape_images)

    print('.\n.\n.')
    # Websites are rated from 0 (safe) to 1 (fraudulent)
    if verified is True:
        if score == 1:
            print('Website has been verified and is fraudulent')
        else:
            print('Website has been verified and is safe')
    else:
        if score[0] > 0.5:
            print('Website is most likely fraudulent. Has score of {}'.format(score[0]))
        else:
            print('Website is most likely safe. Has score of {}'.format(score[0]))
    if explanation is not None:
        print('\nFeature Importance:')
        print(explanation)
