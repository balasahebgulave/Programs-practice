from selenium import webdriver
from bs4 import BeautifulSoup
chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--headless')

class LinkedinSearch():
    def __init__(self):
        self.driver = webdriver.Chrome(r'C:\Users\rnt1013\Desktop\Advanced_Python\MattermarkScrapySelenium\chromedriver.exe',chrome_options=chrome_options)
        self.driver.implicitly_wait(10)
        self.driver.get('https://www.linkedin.com/hp/')
        self.driver.find_element_by_xpath('//*[@id="login-email"]').send_keys('pallati.charan31@gmail.com')
        self.driver.find_element_by_xpath('//*[@id="login-password"]').send_keys('charan')
        self.driver.find_element_by_xpath('//*[@id="login-submit"]').click()

    def get_business_lines(self, companies):
        for index,company in enumerate(companies): 
            print('-------index-------',index,'----->',company)
            self.driver.get(f"https://www.linkedin.com/company/{company}/")
            soup = BeautifulSoup(self.driver.page_source, 'html')
            try:
                industry_type = soup.find(class_="org-top-card-summary__info-item org-top-card-summary__industry").text.strip()
                with open("website_with_domain.tsv","a") as f:
                    f.write(company+"\t"+industry_type+"\n")
            except:
                with open("website_with_domain.tsv","a") as f:
                    f.write(company+"\t"+'Not Found'+"\n")
        
test = LinkedinSearch() 
test.get_business_lines(['amdocs','infosys'])
