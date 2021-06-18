import requests, re, pandas as pd, matplotlib.pyplot as mp
url = 'https://blog.csdn.net/Yellow_python/article/details/88823956'
header = {'User-Agent': 'Opera/8.0 (Windows NT 5.1; U; en)'}
r = requests.get(url, headers=header)
contain = re.findall('<pre><code>([\s\S]+?)</code></pre>', r.text)[0].strip()
df = pd.DataFrame([i.split(',') for i in contain.split('\n')],
                  columns=['province', 'city', 'longitude', 'latitude'])
df['longitude'] = pd.to_numeric(df['longitude'])
df['latitude'] = pd.to_numeric(df['latitude'])
mp.scatter(df['longitude'], df['latitude'], alpha=0.2)
mp.show()
df.to_csv('CNprovince.csv', index=None)