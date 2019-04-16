import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

student_response= pd.read_excel('student_test.xlsx')
correct_options = pd.read_excel('question_options.xlsx')
data = student_response.merge(correct_options, on='question_id')
data = data[data['correct_ans']=='Y']
data = data.reset_index(drop=True)


d = {}
for p in data.email_id:
    d[p] = {'questions':[],'student_answer':[],'right_answer':[]}

for i,j in enumerate(data.email_id):
    j = data.email_id.loc[i]
    d[j]['questions'].append(data.question_id.loc[i])
    d[j]['student_answer'].append(data.option_id_x.loc[i])
    d[j]['right_answer'].append(data.option_id_y.loc[i])
    
each_result = {}
for p in data['question_id'].unique():
        each_result[p] = 0
print(each_result)


user = d.keys()
result = {}

for i in user:
    result[i] = 0
    var = {}
    for p in d[i]['questions']:
        var[p] = []
    for m, n, o in zip(d[i]['questions'], d[i]['student_answer'],d[i]['right_answer']):
        var[m].append((n,o))
#     print(var)
    for q in var:
        if len(var[q])>1:
            real = 0
            for s in range(len(var[q])):
                if var[q][s][0] == var[q][s][1]:
#                     print(var[q])
                    real += 1
            if real == len(var[q]):
                result[i]+=1
                each_result[q]+= 1
#             print(real)
#             print(len(var[q]))
        if var[q][0][0] == var[q][0][1]:
            result[i]+=1
#             print(q)
            each_result[q]+=1
#         print(var[q])
    
print(result)
print(each_result)

