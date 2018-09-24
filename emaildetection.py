import re

email = 'bala@gmail.com'

match = re.search(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', email, re.I)

print(match.group())
