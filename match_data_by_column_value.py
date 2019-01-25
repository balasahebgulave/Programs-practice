import pandas as pd
file = '6_DataSet'
linkedin_data = pd.read_csv(file+'.csv',encoding = 'latin1')
mattermark_data = pd.read_csv('MatterMarkData_Set_Master.csv')

# Name	Website	Stage	Investors	Total Funding	Last funding

mattermark_dict = {}

for ind in range(len(mattermark_data)):
	data = list(mattermark_data.iloc[ind,:])
	mattermark_dict[data[1].lower()] = data

f = open(file+'mattermark.tsv', 'w')

for ind in range(len(linkedin_data)):
	data = list(linkedin_data.iloc[ind,:])
	url = str(data[1]).strip()
	url = url.replace('www.','').lower()

	if url in mattermark_dict.keys():
		x = str(mattermark_dict[url][:])
		f.write( '\t' + url + '\t' + str(mattermark_dict[url][3]) + '\t' + str(mattermark_dict[url][4]) + '\t' + str(mattermark_dict[url][5]) + '\n')
	else:
		f.write( '\t' + url + '\n' )
		
		
		
		
