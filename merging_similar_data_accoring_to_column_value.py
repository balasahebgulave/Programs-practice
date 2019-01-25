import pandas as pd
data1 = pd.read_csv('Data_Set_21(1).csv',encoding = 'latin1')
data2 = pd.read_csv('MatterMarkData_Set_21.csv')

Company_Name = []
Company_Url = []
Company_Description = []
LinkdIn_url = []
Address = []
Current_Head_Count = []
Six_Month_Head_Count = []
One_Year_Head_Count = []
Two_Year_head_Count = []
Industry = []
Name = []
Website = []
Stage = []
Investors = []
Total_Funding = []
Last_Funding = []


l =[]
for k, i in enumerate(data1['Company Url']):
    i =str(i)i = str(i).lower().strip()
    i = i.replace('www.','')
    i = i.replace('http://www.','')
    i = i.replace('https://www.','')
    i = i.replace('https://','')
    i = i.replace('http://','')
    i = i.split('/')[0]
    l.append(i)


l1 = []
for k, i in enumerate(data2['Website']):
    i = str(i).lower().strip()
    if i in l:
        l1.append(i)
        a = list(data2.iloc[k,:])
        Name.append(a[0])
        Website.append(a[1])
        Stage.append(a[2])
        Investors.append(a[3])
        Total_Funding.append(a[4])
        Last_Funding.append(a[5])
        
        for m in range(len(l)):
            if l[m] == i:
                b = list(data1.iloc[m,:])
                Company_Name.append(b[0])
                Company_Url.append(b[1])
                Company_Description.append(b[2])
                LinkdIn_url.append(b[3])
                Address.append(b[4])
                Current_Head_Count.append(b[5])
                Six_Month_Head_Count.append(b[6])
                One_Year_Head_Count.append(b[7])
                Two_Year_head_Count.append(b[8])
                Industry.append(b[9])


data = pd.DataFrame(list(zip(Company_Name ,Company_Url,Company_Description,LinkdIn_url,Address,Current_Head_Count,Six_Month_Head_Count,One_Year_Head_Count,Two_Year_head_Count,Industry,Name,Website,Stage,Investors,Total_Funding,Last_Funding)))

data.to_csv('set_21.csv',sep=',')
