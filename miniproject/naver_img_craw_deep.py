from google_images_download import google_images_download as gid
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as soups
import time
from tqdm import tqdm

print("크롤링 시작")

search_names = ['증명사진']

for search_name in search_names:
    
    # 웹 접속
    base_url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' + str(search_name)
    browser = webdriver.Chrome("./miniproject/chromedriver")
    browser.implicitly_wait(10)

    browser.get(base_url)

    #페이지 스크롤 다운
    body = browser.find_element_by_css_selector('body')
    for i in range(10):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(4)


    # 이미지 링크 수집
    imgs = browser.find_elements_by_css_selector('img._img')
    result = []
    for img in tqdm(imgs):
        if "http" in img.get_attribute('src'):
            result.append(img.get_attribute('src'))
    # print(result)
    # print("수집 완료")

    # 폴더 생성
    import os
    if not os.path.isdir('./miniproject/img/{}'.format(search_name)):
        os.mkdir('./miniproject/img/{}'.format(search_name))
    # print("폴더 생성")

    # 링크로 다운로드
    from urllib.request import urlretrieve

    for index, link in tqdm(enumerate(result)):
        start = link.rfind('.')
        end = link.rfind('&')
        filetype = link[start:end]
        # print(link[0][start:end])

        urlretrieve(link, './miniproject/img/{}/{}{}{}'.format(search_name, search_name, index, filetype))

    print("다운로드 완료")
