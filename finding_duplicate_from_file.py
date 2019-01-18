f = open('demo.txt','r')
data = []
s = set()
unique = set()
duplicate = {}
for i in f:
    unique.add(i)
    data.append(i)    
    duplicate[i] = data.count(i)
for j in duplicate:
    if duplicate[j] == 2:
        s.add(j)
print('Duplicates : ',len(s))
print('Unique : ',len(unique))
