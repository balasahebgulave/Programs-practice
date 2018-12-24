import re

text = 'rohit@gmail.com , bala@gmail.com , rohit@rnt.ai'

pattern = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-zA-Z]+')

matches = pattern.findall(text)

print(matches)
