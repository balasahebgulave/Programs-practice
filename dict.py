
# program for converting dict value to string.

d={"value":[{"optional":"Yes","occasion":"New Year's Day","weekDay":"Monday","date":"2018-01-01"},
{"optional":"No","occasion":"Republic Day","weekDay":"Friday","date":"2018-01-26"},
{"optional":"No","occasion":"Holi","weekDay":"Saturday","date":"2018-03-03"}]

d3=[]
title='This is title'
for j in d['value']:
	d1={'Title':'','Descriptions':[]}
	d2={}
	s=[]
	result=''
	for m,n in j.items():
		s.append(str(m).title()+':'+str(n)+'<br>')
		result=(''.join(s))
	d2['description_element']=result
	d1['Descriptions'].append(d2)
	d1['Title']=title
	d3.append(d1)

print(d3)


OR


d1={'description_top':[]}
d2={}
for i,j in enumerate(d['value']):
	d2['description_element']=j
	d1['description_top'].append(d2)
print(d1)










apidata=[{"staffID":1004,"status":"Pending","sequenceNo":5,"from":"2018-10-15","to":"2018-10-16","days":2,"leaveType":"Pl","reason":"going out","name":"Gauri Walzade"},
{"staffID":1004,"status":"Pending","sequenceNo":6,"from":"2018-09-26","to":"2018-09-27","days":2,"leaveType":"Pl","reason":"Casual or Unwell","name":"Gauri Walzade"},
{"staffID":1004,"status":"Pending","sequenceNo":7,"from":"2018-09-26","to":"2018-09-27","days":2,"leaveType":"Fl","reason":"not well","name":"Gauri Walzade"},
{"staffID":1004,"status":"Pending","sequenceNo":8,"from":"2018-09-26","to":"2018-09-27","days":2,"leaveType":"Fl","reason":"not well","name":"Gauri Walzade"}]



d={'ImageURL': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTd2NZOVTCJkf_awtgASCJFsv0dj_4Uo4oHzni8T5Cyot03b9AQ', 
'Title': 'Panda', 
'API': 'checked', 
'Buttons': [{'ActionType': 'Phone', 'ActionValue': '989898678', 'ButtonTitle': 'View Panda'}]}




final_result=[]
for dic in apidata:
	result=[]
	for key,value in dic.items():
		str_data=str(key)+':'+str(value)+'<br>'
		result.append(str_data)
		process_data=''.join(result)
	x= {}
	for k,v in d.items():
		x[k] = v
	x["Description"] = process_data
	final_result.append(x)
print(final_result)










