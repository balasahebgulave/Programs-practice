import pandas as pd
def clean_csv(csv_name):
    data = pd.read_csv(csv_name, encoding = 'latin1')
#     pattern = re.compile('\S[A-Za-z0-9]\S{0,200}')
    pattern = re.compile('[A-Za-z0-9?*-+. , ^&)(]+[A-Za-z0-9?*-+. , ^&)(]{0,200}')
    fun = lambda x:' '.join(pattern.findall(x)).strip()
    for col in data:
        data[col]= data[col].apply(fun)
    return data
