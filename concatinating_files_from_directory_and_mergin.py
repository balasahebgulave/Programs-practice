import pandas as pd
import os

files = os.listdir(r'C:\Users\rnt1013\Desktop\WEB_CLASSIFICATION\Company Categorization\Completed files')
total_files = []
for index,file in enumerate(files):
    filename = pd.read_excel(f'C:/Users/rnt1013/Desktop/WEB_CLASSIFICATION/Company Categorization/Completed files/{file}')
    total_files.append(filename)
data = pd.concat(total_files,axis=0)

data.to_csv('labeld.tsv',index=None) 

labeled_data = pd.read_csv('labeld.tsv')
labeled_data.head()


featured_data = pd.read_excel('with_out_notfound.xlsx',header=None)

fun = lambda x:x.replace('http://','').replace('https://','').replace('www.','').split('/')[0]
featured_data[1] = featured_data[1].apply(fun)
featured_data.reset_index(inplace=True,drop=True)
featured_data.head()


final = {'corpus':[],'domain':[],'industry_type':[]}
for ind, url in enumerate(labeled_data.Domains):
    for index, link in enumerate(featured_data[1]):
        if url == link:
            final['corpus'].append(featured_data[0].loc[index])
            final['domain'].append(url)
            final['industry_type'].append(labeled_data.Industry.loc[ind])
  
  
data = pd.DataFrame(list(zip(final['corpus'],final['domain'],final['industry_type'])),columns=['corpus','domain','industry_type'])
data = data.drop_duplicates()
data.reset_index(inplace=True,drop=True)
data.head()            



f = []
for i,j in enumerate(data.industry_type.isnull()):
    if j == False:
        f.append(data.loc[i])
        
clean = lambda x:str(x).strip().lower()
final_record = pd.DataFrame(f)
final_record.reset_index(inplace=True, drop=True)
final_record['corpus'] = final_record.corpus.apply(clean)


final_record.to_csv('final.csv')
