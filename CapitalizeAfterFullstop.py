string = 'hi how are you. i am fine today. happy. how are you. where are you.'
string = string.split()
for i,j in enumerate(string):
	if i == 0:
		string[i] = string[i].capitalize()
	if j.endswith('.') and i != len(string)-1:
		string[i+1] = (string[i+1].capitalize())
print(' '.join(string))
		
		
