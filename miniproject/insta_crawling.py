from urllib.request import urlopen
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# user_input = "아이유"

baseurl = 'https://www.instagram.com/explore/tags/'
plusurl = "증명사진"
url = baseurl+ quote_plus(plusurl)

driver = webdriver.Chrome("./miniproject/chromedriver")
driver.get(url)

time.sleep(3)

#페이지 스크롤 다운
body = driver.find_element_by_css_selector('body')
for i in range(4):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(4)

html = driver.page_source
soup = BeautifulSoup(html)

# insta = soup.select('.v1Nh3.kIKUG._bz0w')
insta = soup.select('.v1Nh3.kIKUG._bz0w')

# print(insta)

n = 1
for i in insta:
    print('https://www.instagram.com'+ i.a['href'])
    img_url = i.select_one('.KL4Bh').img['src']
    with urlopen(img_url) as f:
        with open('./miniproject/img/insta/'+ plusurl + str(n) + '.jpg', 'wb') as h:
            img = f.read()
            h.write(img)
    n += 1


driver.close()


print("끝")