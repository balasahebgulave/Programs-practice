import random

mylist = [{'q1': 'question1'}, {'q2': 'question2'}, {'q3': 'question3'}, {'q4': 'question'}, {'q5': 'question5'},
          {'q6': 'question6'}, {'q7': 'question7'}]

random_index = random.sample([i for i in range(len(mylist))], 5)

print(random_index)
