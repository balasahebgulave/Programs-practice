import unittest
from selenium import webdriver
import requests, time
import os
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys

# Refrence link - https://devopsqa.wordpress.com/2018/11/20/zalenium-docker-selenium-grid/

try:
    driver = webdriver.Remote(
            command_executor='http://108.62.118.197:4444/wd/hub',
            desired_capabilities=DesiredCapabilities.FIREFOX)   
    driver.maximize_window()

    driver.get('https://login.yahoo.com/')

    username = driver.find_element_by_name('username')
    username.send_keys('balasahebgulave')
    username.send_keys(Keys.RETURN)
    time.sleep(5)
    password = driver.find_element_by_name('password')
    password.send_keys('9766058622')
    password.send_keys(Keys.RETURN)
    time.sleep(5)
    driver.quit()

except Exception as e:
    raise e



