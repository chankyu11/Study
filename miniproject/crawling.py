# from bs4 import BeautifulSoup
# from urllib.request import urlopen

# response = urlopen('https://zum.com/#!/home/')
# soup = BeautifulSoup(response, 'html.parser')
# i = 1
# f = open("./miniproject/0827/rankk.txt", 'w')
# for anchor in soup.select("span.keyword"):
#     data = str(i) + "위 : " + anchor.get_text() + "\n"
#     i = i + 1
#     f.write(data)
# f.close()

import requests 
from bs4 import BeautifulSoup 
from urllib.request import urlopen 

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'} 
url = 'https://datalab.naver.com/keyword/realtimeList.naver?where=main' 
res = requests.get(url, headers = headers) 
soup = BeautifulSoup(res.content, 'html.parser') 
data = soup.select('span.item_title') 
# f = open("./miniproject/0827/rank.txt", 'w')

f = open("./miniproject/0827/rank.txt", 'w')
i = 1

for item in data:
    data = "%d위 : "%i + item.get_text() + "\n"
    i = i + 1
    f.write(data)
f.close()
