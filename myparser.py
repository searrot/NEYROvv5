from selenium import webdriver
import requests
import urllib
import time
import os
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.firefox.options import Options
from typing import List


class Parser():

    def __init__(self, link, xpath, image_path, driver_path):
        self.driver_path = driver_path
        self.link = link
        self.xpath = xpath
        self.image_path = image_path


    def connect_driver(self):
        options = Options()
        options.headless = True
        self.driver = webdriver.Firefox(executable_path=self.driver_path, options=options)
        self.driver.implicitly_wait(5)


    def start(self):
        self.driver.get(self.link)
        self.driver.implicitly_wait(5)
        self.card = self.driver.find_element_by_xpath(self.xpath)
        self.driver.implicitly_wait(5)
        self.time_post = self.card.find_element_by_xpath('.//time').get_attribute('datetime')
        self.last_time = self.time_post
    

    def get_tweet(self):
        self.driver.implicitly_wait(10)
        self.container:WebElement = self.driver.find_element_by_xpath(self.xpath)
        self.time_post = self.container.find_element_by_xpath('.//time').get_attribute('datetime')


    def get_text(self):
        try:
            self.tweet_text = self.card.find_element_by_xpath('.//div[2]/div[2]/div[1]').text
            return self.tweet_text
        except Exception as e:
            print('_________________________________________________________________________________________________\n')
            print('                              TEXT PARSING ERROR\n')
            print(f'{e}\n')
            print('_________________________________________________________________________________________________\n')


    def get_image(self):
        try:
            self.driver.implicitly_wait(10)
            time.sleep(0.3)
            images:List[WebElement] = self.container.find_elements_by_xpath('.//img')
            time.sleep(0.3)
            self.driver.implicitly_wait(10)
            del images[0]
            if len(images) < 2:
                imname = 'testing'
                src = images[0].get_attribute('src')
                urllib.request.urlretrieve(src, f'{self.image_path}{imname}.jpg')
                print('IMAGE SAVE SUCCESsS')
            else:
                num = 0
                for element in images:
                    imname = f'testing{num}'
                    src = element.get_attribute('src')
                    urllib.request.urlretrieve(src, f'{self.image_path}{imname}.jpg')
                    print('IMAGE SAVE SUCCESsS')
        except Exception as e:
            print('_________________________________________________________________________________________________\n')
            print('                              IMAGE PARSING ERROR\n')
            print(f'{e}\n')
            print('_________________________________________________________________________________________________\n')
