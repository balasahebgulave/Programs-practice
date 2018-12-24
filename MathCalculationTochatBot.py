import re

string = 'show 105+5-10'

addition = ['sum','summation','plus', 'add', 'addition' '+' ]
substract = ['minus' , 'substract' , 'substraction', 'sub', '-' ]
multiply = ['multiply', 'multiplication', 'into', 'times', '*' ]
divide = ['divide', 'division' , 'by' , '/']


def expression(string):
	print('expression')
	pattern1 = re.compile('[\d\(\)\+\-\*\/\.]')
	match1 = pattern1.findall(string)
	match1 = ''.join(match1)
	result = eval(match1)
	print(result)
	

def keyword(string):
	pattern = re.compile('[0-9]{1,111}')
	match = pattern.findall(string)
	for i in string.split():
		if i in addition:
			result = sum([int(j) for j in match ])
			print(result)

		if i in substract and len(match) == 2:
			a = [int(j) for j in match]
			for j in a:
				result = a[-2]-a[-1]
			print(result)
		if i in multiply :
			a = [int(j) for j in match]
			result = 1
			for j in a:
				result = result * j
			print(result)
		if i in divide and len(match):
			a = [int(j) for j in match]
			for j in a:
				result = a[-2]/a[-1]
			print(result)



keyword(string)

expression(string)
