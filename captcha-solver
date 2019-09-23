from selenium import webdriver
from selenium.webdriver.support.ui import Select
from time import sleep
import time
import re
import requests
from bs4 import BeautifulSoup



def check_captcha(driver):
	try:
		content = driver.page_source
		soup = BeautifulSoup(content,'html.parser')
		d = soup.find('iframe').attrs
		out = re.search(r'siteKey=.*\&',d['src'])
		if out:
			print(out)
			out = out.group()
			out = out.rstrip('&').lstrip('siteKey=')
			out = out.split('&')[0]			
			captcha = out

	except Exception as e:
		captcha = False
		print(f"Error:\t{e}")
	if not captcha:
	    return False
	return captcha

def solve_captcha(driver, API_KEY, SITE_KEY, SITE_URL):
	response = 'Hello World!'

	url=f"http://2captcha.com/in.php?key={API_KEY}&method=userrecaptcha&googlekey={SITE_KEY}&pageurl={SITE_URL}"
	resp = requests.get(url) 
	if resp.text[0:2] != 'OK':
	    quit('Error. Captcha is not received')
	captcha_id = resp.text[3:]

	print('---------captcha_id---------',captcha_id)

	# fetch ready 'g-recaptcha-response' token for captcha_id  
	fetch_url = f"http://2captcha.com/res.php?key={API_KEY}&action=get&id={captcha_id}"
	for i in range(1, 20):	
		time.sleep(5) # wait 5 sec.
		resp = requests.get(fetch_url)
		if resp.text[0:2] == 'OK':
			print('---------captcha_token---------',resp.text)
			response = resp.text[3:]			
			break
	response1 = response
	print(f"Reponse:\t{response1}")

	frame = driver.find_element_by_id('recaptcha-iframe')
	driver.switch_to.frame(frame)
	jsript = 'document.getElementById("g-recaptcha-response").innerHTML="%s"; document.getElementById("recaptcha-submit").removeAttribute("disabled") '%(response1)
	#value = response
	driver.execute_script(jsript)
	time.sleep(20)
	driver.find_element_by_xpath('//button[@id="recaptcha-submit"]').click()
	time.sleep(20)

	return True

def captcha_solver(SITE_URL):

	PROXY = "181.214.250.81:3129"  # IP:PORT or HOST:PORT
	chrome_options = webdriver.ChromeOptions()
	chrome_options.add_argument('--proxy-server=%s' % PROXY)

	driver = webdriver.Chrome('C:/Users/balasahebg/Desktop/Balasaheb/chromedriver.exe', options=chrome_options)
	driver.get(SITE_URL)

	# driver.get('https://login.yahoo.com/account/create')

	sleep(2)

	driver.find_element_by_xpath('//*[@id="usernamereg-firstName"]').send_keys("Nilima")
	sleep(1)

	driver.find_element_by_xpath('//*[@id="usernamereg-lastName"]').send_keys("Joshi")
	sleep(1)

	driver.find_element_by_xpath('//*[@id="usernamereg-yid"]').send_keys("JoshiNilima3686")
	sleep(1)

	driver.find_element_by_xpath('//*[@id="usernamereg-password"]').send_keys("Toothpaste#123")
	sleep(1)

	select = Select(driver.find_element_by_xpath('//*[@id="regform"]/div[3]/div[2]/div/select'))
	select.select_by_value("IN")
	sleep(1)

	driver.find_element_by_xpath('//*[@id="usernamereg-phone"]').send_keys("8668827511")
	sleep(1)

	driver.find_element_by_xpath('//*[@id="usernamereg-month"]').send_keys("September")
	sleep(1)

	driver.find_element_by_xpath('//*[@id="usernamereg-day"]').send_keys("8")
	sleep(1)

	driver.find_element_by_xpath('//*[@id="usernamereg-year"]').send_keys("1988")
	sleep(1)

	driver.find_element_by_xpath('//*[@id="reg-submit-button"]').click()
	sleep(10)
	
	captcha = check_captcha(driver)

	API_KEY = 'abcdefgc88c59e932b982888f8db4b2d597f145'
	SITE_URL = driver.current_url


	if captcha != False:
		SITE_KEY = captcha
		print(SITE_KEY)
		recaptcha_response = solve_captcha(driver, API_KEY, SITE_KEY, SITE_URL)
		print(recaptcha_response)
		return True
	else:
		return False

	

if __name__ == "__main__":
	print(captcha_solver('https://login.yahoo.com/account/create'))
