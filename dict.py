
# program for converting dict value to string.

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
