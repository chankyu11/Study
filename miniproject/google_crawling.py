from google_images_download import google_images_download as gid
from selenium import webdriver
from bs4 import BeautifulSoup as soups

def search_selenium(search_name, search_path, search_limit) :
    search_url = "https://www.google.com/search?q=" + str(search_name) + "&hl=ko&tbm=isch"
    
    browser = webdriver.Chrome("./miniproject/chromedriver")
    browser.get(search_url)
    
    image_count = len(browser.find_elements_by_tag_name("img"))
    
    print("로드된 이미지 개수 : ", image_count)

    browser.implicitly_wait(2)

    for i in range( search_limit ) :
        image = browser.find_elements_by_tag_name("img")[i]
        image.screenshot("./miniprojet/animal" + str(i) + ".png")

    browser.close()

if __name__ == "__main__" :

    search_name = '삼국지 유비'
    search_limit = 10
    search_path = "Your Path"
    # search_maybe(search_name, search_limit, search_path)
    search_selenium(search_name, search_path, search_limit)