import os
from datetime import datetime
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import TimeoutError, TCPTimedOutError, DNSLookupError
import pandas as pd


class ScrapySpiderItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    url = scrapy.Field()
    screenshot_filename = scrapy.Field()
    output_path = scrapy.Field()
    do_screenshot = scrapy.Field()
    timestamp = scrapy.Field()
    price = scrapy.Field()

class CssSpider(CrawlSpider):
    name = "CSS"
    tags = ['script', 'link', 'img', 'style', 'picture', 'svg', 'a']
    attrs = ['src', 'href']
    # Rules how to get css and js

    #allow_list = [".css", ".js", '.jpg', 'png', '.tif', '.tiff', '.png', '.gif', '.jpeg', 'jpg', '.jif', '.jfif', 'pdf',
    #              '.html']
    allow_list = ['.css', 'css', '.js', 'js']
    rules = (
        Rule(LinkExtractor(allow=allow_list, tags=tags, attrs=attrs, deny_extensions=[]), callback='parse_css',
             follow=False),
    )

    def __init__(self, input=None, output=None):
        super(CssSpider, self).__init__(CrawlSpider)

        self.input = input
        self.output = output

    def start_requests(self):
        input = self.input
        output_path = self.output
        print("Now extracting css/js ...")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # read list of urls
        urls = [input]

        # only process if new
        for url in urls:
            if url.startswith("http"):
                scrape_url = url
            else:
                scrape_url = "http://" + url

            yield scrapy.Request(url=scrape_url, callback=self.parse)

    def parse_css(self, response):
        output_path = self.output
        # self.logger.info('Hi, this is an css page! %s', response.url)

        if not os.path.exists(output_path + "/" + self.input + '/cssjs'):
            os.makedirs(output_path + "/" + self.input + '/cssjs')

        filename = output_path + "/" + self.input + '/cssjs/%s' % response.url.replace("/", "_").replace(":", "_").replace('?','q')

        with open(filename, 'wb') as f:
            f.write(response.body)

        item = ScrapySpiderItem()
        item['name'] = response.body
        item['url'] = response.url
        item['output_path'] = output_path
        item['do_screenshot'] = False
        item['timestamp'] = False

        return item

    # error handling
    def errback_html(self, failure):
        self.logger.error(repr(failure))

        if failure.check(HttpError):
            response = failure.value.response
            print('HttpError on %s' % response.url)

        elif failure.check(DNSLookupError):
            # this is the original request
            request = failure.request
            print('DNSLookupError on %s' % request.url)


        elif failure.check(TimeoutError, TCPTimedOutError):
            request = failure.request
            print('TimeoutError on %s' % request.url)


class ImgSpider(CrawlSpider):
    name = "IMG"
    tags = ['img', 'link', 'style', 'picture', 'source','svg', 'a']
    attrs = ['src', 'href', 'srcset']
    # Rules how to get css and js

    allow_list = ['.png', 'jpg', '.jpeg', 'tif', '.tiff', '.gif','.jif', '.jfif', '.svg', '.webp']
    rules = (
        Rule(LinkExtractor(allow=allow_list, tags=tags, attrs=attrs, deny_extensions=[]), callback='parse_img',
             follow=False),
    )

    def __init__(self, input=None, output=None):
        super(ImgSpider, self).__init__(CrawlSpider)
        self.input = input
        self.output = output

    def start_requests(self):
        input = self.input
        output_path = self.output
        print("Now extracting images ...")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # read list of urls
        urls = [input]

        # only process if new
        for url in urls:
            if url.startswith("http"):
                scrape_url = url
            else:
                scrape_url = "http://" + url

            yield scrapy.Request(url=scrape_url, callback=self.parse)

    def parse_img(self, response):
        output_path = self.output
        # self.logger.info('Hi, this is an css page! %s', response.url)

        if not os.path.exists(output_path + "/" + self.input + '/images'):
            os.makedirs(output_path + "/" + self.input + '/images')

        filename = output_path + "/" + self.input + '/images/%s' % response.url.replace("/", "_").replace(":", "_").replace('?','q')

        with open(filename, 'wb') as f:
            f.write(response.body)

        item = ScrapySpiderItem()
        item['name'] = response.body
        item['url'] = response.url
        item['output_path'] = output_path
        item['do_screenshot'] = False
        item['timestamp'] = False

        return item

    # error handling
    def errback_html(self, failure):
        self.logger.error(repr(failure))

        if failure.check(HttpError):
            response = failure.value.response
            print('HttpError on %s' % response.url)

        elif failure.check(DNSLookupError):
            # this is the original request
            request = failure.request
            print('DNSLookupError on %s' % request.url)


        elif failure.check(TimeoutError, TCPTimedOutError):
            request = failure.request
            print('TimeoutError on %s' % request.url)

class HtmlSpider(CrawlSpider):
    name = "html"

    def __init__(self, input=None, output=None):
        super(HtmlSpider, self).__init__(CrawlSpider)

        self.input = input
        self.output = output

    def start_requests(self):
        print("*" * 79)
        print("Scraping the site: %s" % self.input)

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        # read list of urls
        urls = [self.input]

        for url in urls:
            if url.startswith("http"):
                scrape_url = url
            else:
                scrape_url = "http://" + url

            yield scrapy.Request(url=scrape_url, callback=self.parse_html, meta={"url_org": url},
                                 errback=self.errback_html)

    def parse_html(self, response):
        if not os.path.exists(self.output + "/" + self.input):
            os.makedirs(self.output + "/" + self.input)

        # html to file
        #filename = output_path + "/" + page + '/MAIN_%s.html' % response.url.replace("/", "_")
        filename = self.output + "/" + self.input + '/index.html'
        with open(filename, 'wb') as f:
            f.write(response.body)

        # get current git hash
        try:
            log_filename = self.output + "/" + self.input + '/log.csv'
            git_hash = os.popen("git log --pretty=format:'%h' -n 1").read()
            log = {'crawl_started': 'html crawled at',
                   'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   'git_hash': git_hash
                   }

            # write header only once
            if os.path.exists(log_filename):
                pd.DataFrame(log, index=[0]).to_csv(log_filename, mode='a', header=False)
            else:
                pd.DataFrame(log, index=[0]).to_csv(log_filename, mode='a', header=True)
        except Exception as e:
            pass

        item = ScrapySpiderItem()
        item['name'] = response.body
        item['url'] = response.url
        item['output_path'] = self.output + "/" + self.input
        item['do_screenshot'] = True
        item['timestamp'] = False

        return item

    # error handling
    def errback_html(self, failure):
        if failure.check(HttpError):
            # these exceptions come from HttpError spider middleware
            # you can get the non-200 response
            response = failure.value.response
            print('HttpError on %s' % response.url)

        elif failure.check(DNSLookupError):
            # this is the original request
            request = failure.request
            print('DNSLookupError on %s' % request.url)

        elif failure.check(TimeoutError, TCPTimedOutError):
            request = failure.request
            print('TimeoutError on %s' % request.url)
