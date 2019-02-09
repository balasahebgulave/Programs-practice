from django.shortcuts import render , HttpResponse
import subprocess
import os 
import shutil
import json
import time
from django.core.files.storage import FileSystemStorage
from Extrating_project.settings import BASE_DIR 
path = BASE_DIR + '\\demoscrapy\\demoscrapy\\spiders\\'


def home(request):

	if request.method == 'POST':
		category = request.POST.get('category')
		
		if category == 'newyork':		
			prc = subprocess.Popen(f"scrapy runspider {path}newyork.py -o newyork.csv", shell=True)
			prc.wait()
		if category == 'politics':			
			prc = subprocess.Popen(f"scrapy runspider {path}politics.py -o politics.csv", shell=True)
			prc.wait()
		if category == 'law':			
			prc = subprocess.Popen(f"scrapy runspider {path}law.py -o law.csv", shell=True)
			prc.wait()
		if category == 'us':			
			prc = subprocess.Popen(f"scrapy runspider {path}US.py -o us.csv", shell=True)
			prc.wait()
		
		filename = category+'.csv'
		
		source = BASE_DIR+'\\'+category+'.csv'
		destination = BASE_DIR+'\\Information\\static\\'+category+'.csv'
		shutil.copy2(source , destination)

		return render(request , 'home.html' , {'filename':filename, 'source':source})

	return render(request , 'home.html')
  
  
  
  
  
	
# subprocess file

'''


# -*- coding: utf-8 -*-
import scrapy
import json

class NewyorkSpider(scrapy.Spider):
    name = 'newyork'
    # allowed_domains = ["wsj.com/public/page/new-york-main.html"]
    start_urls = [r'https://www.wsj.com/public/page/new-york-main.html']

    def parse(self, response):
    	urls=response.xpath('//*[@id="root"]/div/div/div/div[2]/div/div/div[2]/div[1]').css("a::attr(href)").extract()
    	unique_urls=set()
    	for i in urls:
    		unique_urls.add(i)
    	for j in unique_urls:
    		yield scrapy.Request(url=j,callback=self.parse_details)            
        
    def parse_details(self,response):
        body =[]
        content = response.css('div[class="wsj-snippet-body"]>p::text').extract()
        if type(content)==list:
            if len(content)>=1:
                for con in response.css('div[class="wsj-snippet-body"]>p::text').extract():
                    body.append(con)
            else:
                body.append("Content not available")
        else:
            body=content
        js=response.css('script[type="application/ld+json"]::text').extract_first().replace("\n  ","").replace("\n","")
        js=json.loads(js)
        return {"topic":response.css('a[itemprop="item"]::text').extract_first().strip(),"contents":{"Headline":js["headline"],
               "subtitle":response.css('h2[class="sub-head"]::text').extract_first(),
                                         "Author":js["author"]["name"],
                                         "date created":js["dateCreated"],
                                         "Image_url":js["image"]["url"],
                                         "Content":body
                                         }}

      
        
        





'''
	
  
  
  
  # template
  
  
  '''
  
  
  {% extends 'base.html' %}
{% block content %}

<head>
	<title></title>
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">

</head>
<body>
<div class="row" style="margin-bottom: 5% ; height: 380px;">
	<div class="col-md-4" style=" margin-left: 5% ;position: fixed; margin-top: 5%">
        <form method="post" enctype="multipart/form-data">
            {% csrf_token %}
            <div>


            <label><h3>Select Category : </h3></label>
            <select style="color: blue" class="custom-select" name = "category"  required >
            	<option>Select Here</option>
            	<option value="politics">Politics</option>
            	<option value="law">Law</option>
            	<option value="us">Us</option>
            	<option value="newyork">NewYork</option>
             </select><br>
             
            <button class="btn btn-primary" type="submit">Submit</button>
        </form><br>
        </div>

<div style="margin-top: 10%">
	{% if filename %}
	<h3>Download {{filename}}</h3>
	<a href="/static/{{filename}}" download><button class="btn btn-primary" type="submit">Download</button></a>
	{% endif %}
</div>
</div>
</body>

{% endblock %}


  
  
  '''


