import platform, logging, argparse, logging, pickle, os, sys, shutil, time, re
from collections import OrderedDict
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
import traceback
import verify
import requests
import json
import datetime as dt
from typing import List
import numpy as np
from w3lib.url import url_query_cleaner
from url_normalize import url_normalize
from urllib.parse import urlparse
import logoclassifier as kosoh_lc
from urllib.request import Request, urlopen
import pandas as pd
from datetime import datetime

"""turns the model predictions and explainability outputs (lime, shap) into one single dashboard
provides command line interface for generating the dashboard for a given site"""


def fetch_models_and_dict(basedir):
    """locates all models and a dictionary file in a given base dir
    Arguments:
        basedir {str} -- directory to search for files
    :returns: tuple (models, dicts) 
        WHERE
        list models is models file locations. 
        list dicts is dicts file locations  
    """
    model_extensions = ('.model')
    dict_extensions = ('.dict')
    models =[]
    dicts=[]

    for subdir, dirs, files in os.walk(basedir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in model_extensions:
                models.append(os.path.join(subdir, file))
            if ext in dict_extensions:
                dicts.append(os.path.join(subdir, file))

    return models, dicts

def read_input_csv(url_list, csv_input_file,models):
    """reads an input csv file into pandas dataframe. Expecting field: sites

    Args:
        file (Str): file path

    Returns:
        pandas dataframe: df containing sites column
    """
    #read the input data
    if len(url_list)>0:
       df = pd.DataFrame (url_list, columns=['sites'])
    else:
        df = pd.read_csv(csv_input_file) 
    #init used colums 
    df['error'] = None
    for model in models:
        model_name = os.path.basename(model).lower().split(".")[0]
        df[model_name] = np.nan
    print("reading input urls, received: ",df['sites'].count())
    return df


def fix_lime_html_javascript(limehtmlfile):
    """injects javascript to fix broken lime html output (bar formatting)
    
    Arguments:
        limehtmlfile {Str} -- path of html file
    """
    if platform.system() == 'Windows':
        js_modification = "resources\dashboard\explanation_javascrript_additions.txt"
    else:
        js_modification = "resources/dashboard/explanation_javascrript_additions.txt"

    # now add modifications via javascript
    with open(js_modification, "r") as f:
        javascript = f.read() 
        

    with open(limehtmlfile, "r+") as f:
        explhtml = f.read() # read everything in the file
        f.seek(0) # rewind
        #print("javascript: "+javascript)
        explhtml = explhtml.replace("</body></html>",javascript+"</body></html>")
        f.write(explhtml) # write the new line before


def handle_feature_importance_output(feature_importance, input_dir, output_dir, model_name, site):
    """move the explainability output of lime and shap to dashboard folder
    and create a screenshot of the lime html file
    
    Arguments:
        feature_importance {list} -- selection of choices e.g. lime, shap
    """
    lime_html = input_dir + os.path.sep + "explanation_lime.html"
    shap_png = input_dir + os.path.sep + "explanation_shap.png"
    out_base = output_dir+os.path.sep+site+os.path.sep
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    if "lime" in feature_importance and os.path.exists(lime_html):
        dest_lime_html = os.path.join(out_base, model_name+"_explanation_lime.html")
        dest_lime_png = os.path.join(out_base, model_name+"_explanation_lime.png")
        #fix broken lime html rendering
        fix_lime_html_javascript(lime_html)
        #copy to dashboard dir
        shutil.copyfile(lime_html, dest_lime_html)
        print("added",dest_lime_html)
        #render html as png 
        screenshot = take_screenshot("file://"+os.path.abspath(dest_lime_html),out_base, window_size="--window-size=1024,350")
        if screenshot:
            os.rename(screenshot,dest_lime_png)
            print("added",dest_lime_png)

    if "shap" in feature_importance and os.path.exists(shap_png):
        dest_shap_png = os.path.join(out_base, model_name+"_explanation_shap.png")
        shutil.copyfile(shap_png, dest_shap_png)
        print("added",dest_shap_png)

def convert_to_url(url):
        if url.startswith('file://'):
            return url
        if url.startswith('www.'):
            url = 'http://' + url[len('www.'):]
        if url.startswith('http://www.'):
            url = 'http://' + url[len('http://www.'):]
        if url.startswith('https://www.'):
            url= 'http://' + url[len('https://www.'):]
        if url.startswith('https://'):
            url = 'http://' + url[len('https://'):]
        if not url.startswith('http://'):
            url = 'http://' + url
        return url

def canonical_url(url):
    url = url_normalize(url)
    url = url_query_cleaner(url,parameterlist = ['utm_source','utm_medium','utm_campaign','utm_term','utm_content'],remove=True)

    url = url.replace("https://www.","")
    url = url.replace("http://www.","")
    url = url.replace("https://","")
    url = url.replace("http://","")
    if url.endswith('/'):
        url = url[:-1]
    return url

def check_site_is_online(url:str):
    #probes if a website returns status 200 otherwise raises exception
    try:
        req = Request("http://"+url, headers={'User-Agent': 'Mozilla/5.0'})
        urlopen(req,timeout=5).getcode()
    except Exception as err:
        print("website: %s is offline. err: %s"%(url,err))
        raise Exception("website: %s is offline. err: %s"%(url,err)) 

def take_screenshot(url:str, path_screenshots:str, *args, **kwargs):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('headless')
    chrome_options.add_argument('--disable-notifications')
    #required if run as root user
    chrome_options.add_argument('--no-sandbox')
    #chrome_options.add_argument("--start-maximized")

    window_size = kwargs.get('window_size', None)
    if window_size:
        chrome_options.add_argument(window_size)
    else:
        chrome_options.add_argument("--window-size=1024,768")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        #check if url starts with http
        url = convert_to_url(url)
        print('taking screenshot for url %s' % url)      
        #throw a TimeoutException whenever the page load takes more than 30 seconds
        driver.set_page_load_timeout(30)
        driver.get(url)

        element = WebDriverWait(driver, 10).until(
            lambda x: x.find_element_by_xpath("/html/head/title"))
        
        #make sure to give the site enough time to fully load
        time.sleep(2)

        os.makedirs(path_screenshots, exist_ok=True)
        # take a screenshot
        filename = os.path.join(path_screenshots,'screenshot.png')
        driver.get_screenshot_as_file(filename)
    
        driver.close()
        return os.path.abspath(filename)
    except:
        print ('errors taking screenshot')
    finally:
        driver.quit()


def build_html_dashboard(model_results:dict, site:str, output_dir:str, submit_results:bool, identify_logos:bool, site_dashboard_dir):
    """Takes the model results and builds the actual html dashboard file
    
    Arguments:
        model_results {dict} -- key: model name with value: prediction result
        site {str} -- shot url for the dashboard
        output_dir {str} -- base dir for dashboard
    """
    
    def get_placeholder_code(html_template_code,placeholder):
        """extracts the placeholder code from the html template file
        
        Arguments:
            html_template_code {Str} -- html template source code
            placeholder {str} -- placeholder base string e.g. DUPLICATE_JS_FOR_MODEL
        
        Returns:
            {Str} -- html/js code block for that placeholder from the teplate
        """
        #e.g. DUPLICATE_JS_FOR_MODEL
        start_token= "###"+placeholder+"_START###"
        end_token= "###"+placeholder+"_END###"
        result = re.search(start_token+"(.*)"+end_token, html_template_code,re.DOTALL)
        return result.group(1)


    def parse_metrics_file():
        """quick and dirty paring of the metrics file
        """

        metrics_loc = "files/metrics.txt".replace('/',os.path.sep)
        with open(metrics_loc, 'r') as f:
            data = f.read() 

        mydata = data.split("|")
        ret=[]
        items = []
        count=0
        count_elems =0
        for temp in mydata:
            #remove whitespaces
            temp = "".join(temp.split())
            if temp.startswith("+"):
                if count == 1:
                    ret.append(items)
                    items = []
            elif temp is not "":
                items.append(temp)
                count_elems +=1  
            
            count += 1
            if count_elems >9:
                count_elems = 0
                ret.append(items)
                items = []
        return ret

    def replace_model_metric_tokens(html_src,model_name):
        ret = html_src
        metrics = parse_metrics_file()
        #metrics[0]contains the keys as accuracy, precision, etc.
        keys = metrics[0]

        for j in range(1,len(metrics)):
            if metrics[j][0] == model_name:
                for i in range(1, len(keys)):
                    if i<6:
                        #convert to percent output fo precision, recall, etc.
                        ret = ret.replace("###"+keys[i]+"###",'%.1f' % (float(metrics[j][i])*100))
                    else:
                        ret = ret.replace("###"+keys[i]+"###",metrics[j][i])
        return ret


    def replace_model_tokens(html_src,model_name:str, prediction:float):
        #model name and scores
        ret=html_src.replace("###model-name###",model_name)
        ret=ret.replace("###legit-score###",'%.1f' % ((1-prediction)*100))
        ret=ret.replace("###fake-score###",'%.1f' % (prediction*100))  

        global aggregated_models_score
        global aggregated_models_count

        #eplanations
        if model_name == "xgboost":
            expl = "XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance."
            ret = ret.replace("###model-explainer-text###",expl)
            aggregated_models_score += prediction
            aggregated_models_count += 1
        elif model_name == "random_forest":
            expl = "Random Forest is a Machine Learning Algorithm based on Decision Trees. It essentially represents an assembly of a number N of decision trees, thus increasing the robustness of the predictions."
            ret = ret.replace("###model-explainer-text###",expl)
            aggregated_models_score += prediction
            aggregated_models_count += 1
        elif model_name == "single_tree":
            expl = "Clasification based on a single Decision Tree."
            ret = ret.replace("###model-explainer-text###",expl)
        elif model_name == "neural_net":
            expl = "Multi-layer Perceptron (MLP) is a supervised learning algorithm. MLPClassifier implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation."
            ret = ret.replace("###model-explainer-text###",expl)
        else:
            ret = ret.replace("###model-explainer-text###","")

        #f1,precision, recall, etc.
        ret = replace_model_metric_tokens(ret,model_name)

        return ret

    def replace_site_wide_tokens(html_src,site,submit_results,identify_logos, site_dashboard_dir):
        ret=html_src.replace("###shop-url###",site)
        ret=ret.replace("###dashboard-creation-date###",time.strftime("%d.%m.%Y %H:%M"))
        
        #call fake-shop db API to get manual checklist metadata
        db_id, db_date, db_type = get_manual_fakeshopdb_result(site)
        print("*" * 79)

        #check if no entry received
        if(db_id and db_date and db_type):
            # replace individual available items
            if db_id:
                ret = ret.replace("###fake-shop-db-id###",str(db_id))
            else:
                ret = ret.replace("###fake-shop-db-id###","unknown")
            if db_date:
                db_date = dt.datetime.strftime(db_date,"%d.%m.%Y %H:%M")
                ret = ret.replace("###fake-shop-db-added-date###",str(db_date))
            else:
                ret = ret.replace("###fake-shop-db-added-date###","unknown")
            if db_type:
                ret = ret.replace("###fake-shop-db-category###",str(db_type))
                if db_type=="Markenfälscher":
                    ret = ret.replace("###fake-shop-db###","counterfeiter")
                if db_type=="Fake-Shop":
                    ret = ret.replace("###fake-shop-db###","fake_shop")
            else:
                ret = ret.replace("###fake-shop-db-category###","unknown")
                ret = ret.replace("###fake-shop-db###","unknown")

            #set display block
            ret = ret.replace("###fake-shop-db-display###","block")
        else:
            #set display none
            ret = ret.replace("###fake-shop-db-display###","none")


        #call internet archive to add archived snapshot
        arch_time, arch_url = get_entry_from_wayback_machine(site, retry=1)

        def get_day_dif_to_today(arch_time):
            if arch_time==None:
                return -1
            else:
                today =  dt.date.today()
                archive_date = dt.datetime.strptime(arch_time,"%d.%m.%Y %H:%M")
                diff = today - archive_date .date()
                return diff.days

        if (not arch_url) and submit_results:
            #shop does not yet exist - submit site for internet archive snapshot
            submit_to_wayback_machine(site)
            arch_time, arch_url = get_entry_from_wayback_machine(site, retry=3)
        elif get_day_dif_to_today(arch_time) > 180 and submit_results:
            #if archive is older than 180 days re-request a snapshot
            submit_to_wayback_machine(site)
            arch_time, arch_url = get_entry_from_wayback_machine(site, retry=3)

        print("*" * 79)

        if arch_url and arch_time:
            ret = ret.replace("###waybackmachine-archive-link###",arch_url)
            ret = ret.replace("###waybackmachine-archive-date###",str(arch_time))
            ret = ret.replace("###waybackmachine-archive-display###","block")
        else:
            ret = ret.replace("###waybackmachine-archive-display###","none")

        def add_detected_images_to_html_dashboard(ret_html, template_html_code_block, detected_files, count):
            #iterate over all resulting files - and copy them to dashbaord dir
            for idx,f in enumerate(detected_files):
                copy_img_to_dir = os.path.abspath(site_dashboard_dir)
                file_type = f.split('.')[-1]
                file_name = "identified_logo"+str(idx)+"."+file_type
                img_copy_path = copy_img_to_dir+"/"+file_name
                shutil.copy(f,img_copy_path)
                #e.g. watchlist-internet.at/images/https%3A__www.weingrube.com_weingrube_weingrube-icon-euro-label.jpg
                ret_html = ret_html.replace("###DUPLICATE_HTML_FOR_TRUSTMARK_DETECTIONS_PLACEHOLDER###",template_html_code_block+"###DUPLICATE_HTML_FOR_TRUSTMARK_DETECTIONS_PLACEHOLDER###")
                ret_html = ret_html.replace("###kosoh_trustmark_detection-item###",file_name)
                count += 1
            return ret_html,count
            

        # KOSOH trustmark/payment provider image-analysis 
        if identify_logos:
            template_html_code_block = get_placeholder_code(html_template_src,"DUPLICATE_HTML_FOR_TRUSTMARK_DETECTIONS")
            ret = ret.replace(template_html_code_block,"")
            ret = ret.replace("###DUPLICATE_HTML_FOR_TRUSTMARK_DETECTIONS_START###","")
            ret = ret.replace("###DUPLICATE_HTML_FOR_TRUSTMARK_DETECTIONS_END###","")
            count = 0
            scraped_site_root = os.path.abspath(os.getcwd())+"/data/verify_sites/"+site+"/"
            train_data_supported = kosoh_lc.list_available_trainingsdata_location()
            print("applying KOSOH for trustmark/payment_provider logo classification")

            #treat ecommerge trustmark differently as calibrated threasholds exist
            found,found_files = find_ecg_trustmark_images(scraped_site_root)
            ret, count = add_detected_images_to_html_dashboard(ret, template_html_code_block, found_files, count)
            print("** trustmark/ecguetesiegel done. items identified: {}".format(len(found_files)))
            
            for train_data_names in train_data_supported:
                if(train_data_names!="trustmark/ecguetesiegel"):
                    #run classifier on all supported logos wihere trainingsdata exists
                    found,found_files = find_trustmark_and_payment_provider_images(scraped_site_root,train_data_supported.get(train_data_names))
                    ret, count = add_detected_images_to_html_dashboard(ret, template_html_code_block, found_files, count)
                    print("** {} done. items identified: {}".format(train_data_names, len(found_files)))

            ret = ret.replace("###DUPLICATE_HTML_FOR_TRUSTMARK_DETECTIONS_PLACEHOLDER###","")
            print("*" * 79)

            #check if we added any items?
            if(count > 0):
                ret = ret.replace("###kosoh_trustmark_detection-display###","block")
            else:
                ret = ret.replace("###kosoh_trustmark_detection-display###","none")
        else:
            ret = ret.replace("###kosoh_trustmark_detection-display###","none")

        return ret
    
    def delete_all_placeholders(html_src):
        html_src = html_src.replace("###DUPLICATE_JS_FOR_MODEL_START###","")
        html_src = html_src.replace("###DUPLICATE_JS_FOR_MODEL_END###","")
        html_src = html_src.replace("###DUPLICATE_JS_FOR_MODEL_PLACEHOLDER###","")
        html_src = html_src.replace("###DUPLICATE_HTML_FOR_MODEL_START###","")
        html_src = html_src.replace("###DUPLICATE_HTML_FOR_MODEL_END###","")
        html_src = html_src.replace("###DUPLICATE_HTML_FOR_MODEL_PLACEHOLDER###","")
        return html_src

    def replace_submit_fakeshop_db_items(html_src, site, model_name, fake_score):
        #submit aggregated model results to fake-shop website database
        if submit_results == True:
            # threashold for submission (0% risk-level i.e. submit all entries)
            if fake_score >= 0:
                print("submit {} to fake-shop db: risk-score {} model {}".format(site,'%.2f' % fake_score, model_name))
                db_submit_message = submit_result_to_fakeshopdb(site,fake_score)
                html_src = html_src.replace("###fake-shop-db-website-submitted-message###",str(db_submit_message))
            else:
                print("submit {} to fake-shop db: risk-score {} model {} - skip below threshhold".format(site,'%.2f' % fake_score, model_name))
                html_src = html_src.replace("###fake-shop-db-website-submitted-message###","below threshold")

            html_src = html_src.replace("###fake-shop-db-website-submitted-display###", "block")
        else:
            html_src = html_src.replace("###fake-shop-db-website-submitted-display###", "none")
        return html_src


    #parse dashboard_template.html
    html_template_loc = "resources/dashboard/dashboard_template.html".replace('/',os.path.sep)
    out_base = output_dir+os.path.sep+site+os.path.sep

    with open(html_template_loc) as f:
        html_template_src = f.read()

    placeholderjs = "###DUPLICATE_JS_FOR_MODEL_PLACEHOLDER###"
    placeholderhtml = "###DUPLICATE_HTML_FOR_MODEL_PLACEHOLDER###"

    # read code blocks from template to duplicate for model elements
    template_modeljs_code_block = get_placeholder_code(html_template_src,"DUPLICATE_JS_FOR_MODEL")
    template_modelhtml_code_block = get_placeholder_code(html_template_src,"DUPLICATE_HTML_FOR_MODEL")
    # remove those placeholders from the template
    html_template_src = html_template_src.replace(template_modeljs_code_block,"")
    html_template_src = html_template_src.replace(template_modelhtml_code_block,"")
    
    html_dashboard_src = html_template_src

    # sort predictions by value - highest (fake-shop score) first
    model_results = OrderedDict(sorted(model_results.items(), key=lambda kv: kv[1]))
    model_results = OrderedDict(reversed(list(model_results.items())))

    #now add the html specific to each model
    for model_name in model_results:
        fraud_score = model_results[model_name][0]

        #build html fragments
        modeljs_code_block = replace_model_tokens(template_modeljs_code_block,model_name,fraud_score)
        modelhtml_code_block = replace_model_tokens(template_modelhtml_code_block,model_name,fraud_score)

        #add them to dashboard
        html_dashboard_src = html_dashboard_src.replace(placeholderjs, modeljs_code_block+placeholderjs)
        html_dashboard_src = html_dashboard_src.replace(placeholderhtml, modelhtml_code_block+placeholderhtml)

    #submit aggregated model prediction to fakesho-db
    html_dashboard_src = replace_submit_fakeshop_db_items(html_dashboard_src,site,aggregated_models_name,aggregated_models_score/aggregated_models_count)

    #replace model independent tokens
    html_dashboard_src = replace_site_wide_tokens(html_dashboard_src,site,submit_results,identify_logos,site_dashboard_dir)
    #clean up the remaining placeholders
    html_dashboard_src = delete_all_placeholders(html_dashboard_src)

    #write dasbhoard to disk
    with open(out_base+'dashboard.html', 'w') as output_file:
        output_file.write(html_dashboard_src)

    print('created dashboard at',os.path.abspath(out_base+'dashboard.html'))


def build_json_index(out_dir):
    index_file = out_dir+os.path.sep+"folders.json"

    def get_dashboards_index_dirs(out_dir):
        """locates all dashboards and writes a json index file 
        """
        with_screenshot =[]
        without_screenshot=[]
        
        for dirpath, dirnames, files in os.walk(out_dir):
            if "dashboard.html" in files:
                if "screenshot.png" in files:
                    with_screenshot.append(dirpath)
                else:
                    without_screenshot.append(dirpath)
        
        return with_screenshot, without_screenshot

    def create_dashboards_index(out_dir, with_screenshots, without_screenshots):
        json = "var folders = \n\t{\n\ttype: 'dir',\n\tname: 'dashboard',\n\tchildren:["
        item = "\n\t\t{\n\t\ttype: 'dir',\n\t\tname: 'dashb_name',\n\t\timage: 'image_png',\n\t\tlink: 'link_url',\n\t\tchildren:[]\n\t\t}"
        
        def add_index_items(json,files:list,no_img:bool):
            count =0
            for ws in files:
                site_name = ws.split(os.path.sep)[-1]
                site_url="../"+site_name+"/dashboard.html"
                if no_img:
                    image_url = "../../resources/dashboard/no-preview.jpg"
                else:
                    image_url = "../"+site_name+"/screenshot.png"
                #replace and append to json
                json+=item.replace("dashb_name",site_name).replace("link_url",site_url).replace("image_png",image_url)
                if count<len(files)-1:
                    json+=","
                count+=1
            return json
        
        if len(without_screenshots)>0:
            json = add_index_items(json,without_screenshots,True)+",\n"
        json = add_index_items(json,with_screenshots,False)
        json+="\n\t]\n};"
        return json

    with_sh, without_sh = get_dashboards_index_dirs(out_dir)
    json = create_dashboards_index(out_dir,with_sh,without_sh)
    #write index to disk
    with open(index_file, 'w') as output_file:
        output_file.write(json)

    print('created index at',index_file)


def load_api_token():
        #load api key from disk - not provided in the repo(!)
        #parse dashboard_template.html
        try:
            api_key_loc = "resources/dashboard/fakeshopdb_api_access_token.txt".replace('/',os.path.sep)
            with open(api_key_loc) as f:
                api_key = f.read()
            return api_key
        except:
            return None

def get_manual_fakeshopdb_result(site):
    """queries the mal2 fake-shop database (ÖIAT fake-shop manual checklist evaluation)
    
    Arguments:
        site {Str} -- site to check fake-shop database for e.g. "elektronio.de"

    Returns:
        fakeshop db id {Str}, creation date {datetime}, type of fake-shop {Str}
    """

    api_url = FAKE_SHOP_API_HOST+"api/v1/website/?url="
    api_token = load_api_token()
    url = api_url+site
    
    db_id = None
    db_created_at = None
    db_ws_type = None

    if api_token:
        print("chcecking fake-shop db entry?",url)
    else:
        print("no fake-shop db API token provided - skipping")
        return db_id, db_created_at, db_ws_type
   
    headers = {'content-type': 'application/json', 'Authorization':'Token {}'.format(api_token) }
    resp =requests.get(url,headers=headers)
    
    #parse responds
    if resp.status_code == 200:
        #print(resp.json())
        resp_dict= resp.json()
        #there should be a results entry in the response json
        if resp_dict['results']:
            #iterate over the fake-shop db results and filter out wildcard matchings e.g. orf.at in .ff-wallendorf.at   
            for res_entry in resp_dict['results']:
                if(convert_to_url(res_entry['url']) == convert_to_url(site)):
                    #if a shop was analyzed and marked as fakeshop that's its corresponding fake shop db_id
                    db_id = res_entry['db_id']
                    db_url = res_entry['url']
                    db_created_at = res_entry['created_at']
                    db_created_at = dt.datetime.fromisoformat(db_created_at)
                    #ws_type: 2=Fake Shop 3=Markenfälscher
                    db_ws_type = res_entry['website_type']
                    if(db_ws_type == 2):
                        db_ws_type = "Fake-Shop"
                        pdf_link_details = FAKE_SHOP_API_HOST+"db/fake_shop/{}/details".format(db_id)
                    elif(db_ws_type == 3):
                        db_ws_type = "Markenfälscher"
                        pdf_link_details = FAKE_SHOP_API_HOST+"db/counterfeiter/{}/details".format(db_id)
                    else:
                        db_ws_type = "unknown category"
                        pdf_link_details = FAKE_SHOP_API_HOST
                    print("fake-shop db id: {}, creation date: {}, type: {}".format(db_id,db_created_at,db_ws_type))
                    print("fake-shop details:", pdf_link_details)
                    print("*" * 79)
                    return db_id, db_created_at, db_ws_type
            print("unknown site to fake-shop database")
        else:
            print("unknown site to fake-shop database")
    elif resp.status_code == 403:
        print("invalid credentials",resp.status_code,resp.text)
    else:
        print("error",resp.status_code,resp.text)
        
    return db_id, db_created_at, db_ws_type


def submit_result_to_fakeshopdb(site,model_score):
    """posts a site to website database of potential fake-shops for further manual expert evaluation)
    
    Arguments:
        site {Str} -- site to push to fake-shop website database for e.g. "elektronio.de"
        model_score {Float} -- risk_score of beeing fake determined by the mal2 prediction model

    Returns:
        submission_resp {Str} -- human readable response of status
    """

    def translate_model_score(model_score):
        #translates score to fake-shop db risk_types very low, low, below average, above average, high, very high, unknown
        #https://db-dev.malzwei.at/admin/mal2_db/websiteriskscore/
        range_low = np.arange(1, 100, 0.01)
        score = model_score * 100
        if np.logical_and(score >= 0, score < 10):
            print("risk-score: very low")
            #id for very low
            return 5
        elif np.logical_and(score >= 10, score < 25):
            print("risk-score: low")
            #id for low
            return 1
        elif np.logical_and(score >= 25, score < 50):
            print("risk-score: below average")
            #id for below average
            return 2
        elif np.logical_and(score >= 50, score < 80):
            print("risk-score: above average")
            #id for above average
            return 7
        elif np.logical_and(score >= 80, score < 90):
            print("risk-score: high")
            #id for high
            return 3
        elif np.logical_and(score >= 90, score <=100):
            print("risk-score: very high")
            #id for very high
            return 6
        else:
            print("risk-score: unknown")
            #id for unknown
            return 4

    api_url = FAKE_SHOP_API_HOST+'api/v1/website/'
    api_token = load_api_token()

    #reported_by https://db-dev.malzwei.at/admin/mal2_db/websitereportedby/
    json_data = {"url": "{}".format(convert_to_url(site)), "risk_score": translate_model_score(model_score), "reported_by": 3}
    submission_resp = ""

    if api_token:
        print("submitting {} with risk-score {}\nto: {}".format(site,'%.2f' % model_score,api_url))
    else:
        print("no fake-shop db API token provided - skipping")
        return submission_resp
   
    headers = {'content-type': 'application/json', 'Authorization':'Token {}'.format(api_token) }
    resp = requests.post(api_url, data=json.dumps(json_data), headers=headers)

    #parse responds
    if resp.status_code == 201:
        resp_dict= resp.json()
        #there should be a create_at entry in the response json
        if resp_dict['created_at']:
            #record date as return message
            submission_resp = dt.datetime.fromisoformat(resp_dict['created_at'])
            submission_resp = dt.datetime.strftime(submission_resp,"%d.%m.%Y %H:%M")
            print("recorded website id: {}, creation date: {}, url: {}".format(resp_dict['id'],submission_resp,resp_dict['url']))
        else:
            print("errors parsing the API response")
    #in case website already existed
    elif resp.status_code == 400:
        submission_resp = "website previously recorded for manual inspection"
        print("existing website in db - skipping result submission")
    elif resp.status_code == 403:
        print("invalid credentials",resp.status_code,resp.text)
    else:
        print("error",resp.status_code,resp.text)

    print("*" * 79)
    return submission_resp


def submit_to_wayback_machine(url):
    """submit a site to the internet wayback archive

    Arguments:
        url {Str} -- url to submit to internet arvhive for archiving snapshot
    """
    request_url = "https://web.archive.org/save/"+url
    headers = {'content-type': 'application/json'}
    print("submitting {} to wayback-machine {}".format(url, request_url))
    try:
        resp =requests.get(request_url,headers=headers)
        
        if resp.status_code == 200:
            print("successfully submitted site {} to the internet archive".format(url))
        else:
            print("error wayback-machine submission",resp.status_code,resp.text)
    except:
        print("failed wayback-machine submission")

def get_entry_from_wayback_machine(url,retry=3):
    """queries the internet wayback arhive on the latest snapshot

    Arguments:
        url {Str} -- url to request archived snapshot for e.g. google.at

    Keyword Arguments:
        retry {int} -- number of retries (default: {3})

    Returns:
        archive_date {Str} -- archive creation date - formatted for output %d.%m.%Y %H:%M
        archive_url {Str} -- link to waybackmachine

    """
    ret_arch_time = None
    ret_arch_url = None
    request_url = "https://archive.org/wayback/available?url="+url
    headers = {'content-type': 'application/json'}
    print("getting internet archive for {} from wayback-machine {}".format(url, request_url))
    #Note: API is only able to deliver latest (closest) snapshot
    
    #try three times - just in case we previously asked to create archive
    counter = 0
    while ret_arch_url == None and counter < retry:
        #send request
        resp =requests.get(request_url,headers=headers)
        #check on status code
        if resp.status_code == 200:
            resp_dict= resp.json()
            #print(resp_dict)
            if resp_dict['archived_snapshots']:
                timestamp = resp_dict['archived_snapshots']['closest']['timestamp']
                ret_arch_time = dt.datetime.strptime(timestamp,'%Y%m%d%H%M%S')
                ret_arch_time = dt.datetime.strftime(ret_arch_time,"%d.%m.%Y %H:%M")
                ret_arch_url = resp_dict['archived_snapshots']['closest']['url']
                print("received webarchive entry. archived: {} url: {}".format(ret_arch_time,ret_arch_url))
                return ret_arch_time, ret_arch_url
            else:
                print("no archive available at wayback-machine")
        else:
            print("error",resp.status_code,resp.text)
        counter += 1
        #sleep for 3 seconds
        time.sleep(3)
    return ret_arch_time, ret_arch_url

def find_ecg_trustmark_images(site_html_dir):
    """runs the KOSOH logoclassifier to check and identify if the ecommerce trustmark images are contained in the scraped site archive. classifier provides calibrated threshold for confidence for this case

    Arguments:
        site_html_dir {Str} -- root directory for the scraped html code of the given site

    Returns:
        contains {bool} -- indicates if trustmark was found True or not False
        found_files {Str} -- array of paths to images found or [] if none were found

    """
    if os.path.exists(site_html_dir):
        KOSOH_ecg = kosoh_lc.LogoClassifierECG()
        #contains = KOSOH_ecg.contains_logo("/media/sf_D_DRIVE/MAL2/_crawled_datasets/falter_online_shop/haengemattenshop.com/*")
        #similar = KOSOH_ecg.list_similar_images("/media/sf_D_DRIVE/MAL2/_crawled_datasets/falter_online_shop/haengemattenshop.com/*")
        abs_path = site_html_dir+"images"+os.path.sep+"*"
        #print("applying KOSOH imageclassifier on {}".format(abs_path))
        contains = KOSOH_ecg.contains_logo(abs_path)
        found_files = KOSOH_ecg.list_similar_images(abs_path)
        #print("successfully identified: {} logos at: {}".format(len(found_files), found_files))
        return contains,found_files
    else:
        print("no matching images found")
        return False,[]

def find_trustmark_and_payment_provider_images(site_html_dir, train_data_dir, threshold= 0.999):
    """runs the KOSOH logoclassifier to check and identify if trustmark or payment provider images are contained in the scraped site archive

    Arguments:
        site_html_dir {Str} -- root directory for the scraped html code of the given site
        train_data_dir {Str} -- dir with data for training the classifier on
        threshold {Float} -- min. required threshold for accpeting classification. default 0.999

    Returns:
        contains {bool} -- indicates if images were found True or not False
        found_files {Str} -- array of paths to images found or [] if none were found

    """
    if os.path.exists(site_html_dir):
        KOSOH_Classifier = kosoh_lc.LogoClassifierECG()
        KOSOH_Classifier.debug = False
        KOSOH_Classifier.show_figs = True
        KOSOH_Classifier.threshold = threshold
        KOSOH_Classifier.selector_metric = 0
        KOSOH_Classifier.gt_treshold = True
        
        abs_path = site_html_dir+"images"+os.path.sep+"*"
        #print("applying KOSOH imageclassifier on {}".format(abs_path))
        contains = KOSOH_Classifier.compare_multiple(path_test = abs_path,path_train_folder=train_data_dir+"*")[0]
        found_files = KOSOH_Classifier.list_similar_images(abs_path)
        #print("successfully identified: {} logos at: {}".format(len(found_files), found_files))
        return contains,found_files
    else:
        print("no matching images found")
        return False,[]
   
def main(url:str,models:List[str],dicts:List[str],output_dir:str, feature_importance:List[str] = None ,use_cache:bool = False,submit_results:bool = False, identify_logos:bool = False):
    
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

    if len(dicts)>0:
        dictv = dicts[0]
        print("vectorizer:", dictv)
    else:
        print("unable to localize the dictionary in",args.model_dir)
        sys.exit(2)

    if len(models)>0:
        print("models:",models)
    else:
        print("unable to localize at least one model in",output_dir)
        sys.exit(2)

    #delete old content
    site_dashboard_dir = output_dir+os.path.sep+url+os.path.sep
    if os.path.isdir(site_dashboard_dir):
        try:
            #shutil.rmtree(site_dashboard_dir)
            __nukedir_recursively(site_dashboard_dir)
        except:
            print('Error while deleting dashboard:',site_dashboard_dir)
    
    #init output dir
    os.makedirs(os.path.dirname(site_dashboard_dir), exist_ok=True)

    #dict of model name and scores
    d_scores ={}
    #now iterate over all models
    for model in models:
        model_name = os.path.basename(model).lower().split(".")[0]
        #call mal2 verify component 
        score, verified, explanation = verify.main(model, dictv, url, check_db=False, use_cache=use_cache, do_feature_importance=feature_importance, do_scrape_images=identify_logos)
        d_scores[model_name]=score
        print("prediction",model_name,score, url)

        #move shap and lime output to dashboard dir
        if feature_importance:
            handle_feature_importance_output(feature_importance,args.model_dir,output_dir,model_name,url)

        #force scrape only one 
        use_cache=True

    #take screenshot
    print("*" * 79)
    screenshot = take_screenshot(url,site_dashboard_dir)
    if screenshot:
        print("screenshot",screenshot)

    print("*" * 79)
    #build the htlm dashboard
    build_html_dashboard(d_scores,url,output_dir,submit_results,identify_logos,site_dashboard_dir)

    print("*" * 79)
    #build the index json
    build_json_index(output_dir)

    print('.\n.\n.')
    
    #return the predictions
    return url, d_scores


#fake-shop db API base URL
FAKE_SHOP_API_HOST =  "https://db.malzwei.at/"
#FAKE_SHOP_API_HOST_DEV =  "https://db-dev.malzwei.at/"

#init aggregated model prediction results
aggregated_models_score = 0
aggregated_models_count = 0
aggregated_models_name= "aggregated_prediction"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Logger = logging.getLogger('main.stdout')

    Args = argparse.ArgumentParser(description="Dashboard builder for eCommerce Site to verify")
    Args.add_argument("-u", "--urls", dest= 'urls', default=[], nargs='*', help="URL(s) of the site to check.  model based risk-score(s) are predicted on the base url.")
    Args.add_argument("-c", "--input-csv", dest='input_csv', default=None, nargs='?', help="Relative path of CSV input file with URLs in the field 'site' to check. model based risk-score(s) are predicted on the base url of every site.")
    Args.add_argument("-m", "--model-dir", dest='model_dir', default="files", help="Relative path to location holding the models and dict")
    Args.add_argument("-d", "--dashboard-dir", dest='dashboard_dir', default="dashboard", help="Relative path to location for dasbhoard results")
    Args.add_argument("-f", "--feature-importance", dest='feature_importance', default=None, choices=['lime', 'shap'], nargs='+', help="Options are: lime, shap or no flag for None")
    Args.add_argument("--use-cache", default=False, action="store_true", dest='use_cache', help="Set flag if you want to use locally cached version of scraped html files - re-scrapes only when site not available locally")
    Args.add_argument("--submit-results", default=False, action="store_true", dest='submit_results', help="Set flag if you want to submit the results of the model prediction to the fake-shop database for manual inspection")
    Args.add_argument("--identify-logos", default=False, action="store_true", dest='identify_logos', help="Set flag if you want to scrape images and apply the KOSOH trustmark/payment-provider image identifier")
    args = Args.parse_args()
    Logger.debug("site(s): {}, input-csv: {}, model-dir: {}, dashboard-dir: {}, feature-importance: {}, use-cache: {}, sumit-results: {}, identify-logos: {} ".format(args.urls,args.input_csv,args.model_dir, args.dashboard_dir, args.feature_importance, args.use_cache, args.submit_results, args.identify_logos))

    #timestamp start processing
    time_start = datetime.utcnow()

    #either -u or -c allowed
    if len(args.urls)<1 and args.input_csv == None:
        print("ERROR: mandatory parameter missing, please specify either --url or --input-csv as input")
        print("*" * 79)
    elif len(args.urls)>0 and args.input_csv != None:
        print("ERROR: please specify either --urls or --input-csv as input, not both")
        print("*" * 79)
    elif args.input_csv != None and not (os.path.exists(args.input_csv)):
        print("ERROR: invalid --input_csv location: ",args.input_csv)
    else:
        try:
            #read all available models and dictionary
            models, dicts = fetch_models_and_dict(args.model_dir)
            if len(models)<1 or len(dicts)<1:
                raise Exception("no models or dicts found in {}".format(args.model_dir))
            print("*" * 79)

            # init dataframe with input urls to check
            df = read_input_csv(args.urls, args.input_csv, models)
            print("*" * 79)

            for i,row in df.iterrows():
                print("PROCESSING item", i, row['sites'])
                try:
                    #dashboard due to folder structure only operates on netloc - verify itself is able to also work on url paths
                    base_url = convert_to_url(row['sites'])
                    base_url = urlparse(base_url).netloc
                    #cleanup trailing paths and www
                    base_url = canonical_url(base_url)
                    print("operate on base url: "+base_url)
                
                    check_site_is_online(base_url)
                    #call the dashboard builder main
                    url, predictions = main(base_url,models,dicts,args.dashboard_dir,args.feature_importance,args.use_cache, args.submit_results, args.identify_logos)
                    print("PREDICTED risk-scores for {}".format(url),predictions)

                    #update dataframe with results
                    for model_name in predictions.keys():
                        df[model_name].at[i] = predictions[model_name][0]
                    
                except Exception as e:
                    print("FAILED: to get prediction: ",e)
                    #traceback
                    #traceback.print_exc()
                    df["error"].at[i]=str(e)

                #update csv file
                df.to_csv(time_start.strftime("%s")+"_results.csv", index=False)
                print("*" * 79)
                print("*" * 79)

        except Exception as e:
            print("FAILED: to init dashboard: ",e)
