# 모두 안됨.

# from google_images_download import google_images_download

# response = google_images_download.googleimagesdownload()   #class instantiation

# arguments = {"keywords":"coke, tiger, bear","limit":5,"print_urls":True}   #creating list of arguments
# paths = response.download(arguments)   #passing the arguments to the function
# print(paths)   #printing absolute paths of the downloaded images


# https://data-make.tistory.com/172
from google_images_download import google_images_download
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def imgcraw(keyword, dir):
    response = google_images_download.googleimagesdownload()

    arguments = {"keyword": keyword,
                "limit": 50,
                "print_url": True,
                "no_directory": True,
                "output_directory": dir}

    paths = response.download(arguments)
    print(paths)

imgcraw("dog",'./miniproject/0827/downloadTest')


