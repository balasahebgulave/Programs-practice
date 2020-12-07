from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep

options = Options()
driver = webdriver.Chrome(options=options, executable_path='chromedriver.exe')
wait = WebDriverWait(driver, 10)
driver.get('https://login.yahoo.com')
driver.maximize_window()
sleep(5)
driver.find_element_by_xpath('//*[@id="login-username"]').send_keys('nicolasimpsonfir@yahoo.com')
sleep(5)
driver.find_element_by_xpath('//*[@id="login-username"]').send_keys(Keys.RETURN)
sleep(5)
iframe = driver.find_element_by_xpath("//iframe[@id='recaptcha-iframe']")
driver.switch_to.frame(iframe)

iframe1 = driver.find_element_by_xpath("//iframe[@role='presentation']")
driver.switch_to.frame(iframe1)

checkbox = driver.find_element_by_id("recaptcha-anchor").click()
print(checkbox)
sleep(5)
driver.find_element_by_xpath('//*[@id="login-passwd"]').send_keys('hxxdQRataD')
sleep(5)
wait.until(EC.presence_of_element_located((By.ID, 'login-passwd'))).send_keys(Keys.RETURN)
