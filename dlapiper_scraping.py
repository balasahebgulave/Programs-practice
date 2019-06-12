from bs4 import BeautifulSoup
import requests
from selenium import webdriver
driver = webdriver.Chrome(r'C:\Users\rnt1013\Desktop\Advanced_Python\MattermarkScrapySelenium\chromedriver.exe')
url = 'https://www.dlapiper.com/en/asiapacific/people/#sort=relevancy'
# data = requests.get(url)
f = open('dlapiper.tsv','w')
for index in range(0,5000,10):
    suburl = url+f'&first={index}'
    driver.get(suburl)
    data = driver.page_source
    soup = BeautifulSoup(data,'lxml')
    main_div = soup.find_all(class_="coveo-result-frame")
    for i in range(len(main_div)):
        try:
            name = main_div[i].find(class_="CoveoResultLink").text
        except:
            name='None'
        try:
            role = main_div[i].find(attrs={"data-field" : "@dlaprofessionallevel"}).text
        except:
            role='None'
        try:
            location = main_div[i].find(attrs={"data-facet" : "offices"}).text
        except:
            location='None'
        try:
            phone = main_div[i].find(attrs={"data-helper" : "phoneNumber"}).text
        except:
            phone='None'
        try:
            email = main_div[i].find(attrs={"data-helper" : "email"}).text
        except:
            email='None'
        print(name)
        print(role)
        print(location+phone)
        print(email)
        try:
            f.write(name+'\t'+role+'\t'+location+'\t'+phone+'\t'+email+'\n')
        except:
            f.write('name'+'\t'+'role'+'\t'+'location'+'\t'+'phone'+'\t'+'email'+'\n')
        
        print(suburl)
        driver.close()
