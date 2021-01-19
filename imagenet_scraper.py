import cv2 as cv
import nltk
from nltk.corpus import wordnet as wn
from selenium import webdriver 
import numpy as np
import os
import urllib
import re
import sys

nltk.download('wordnet')

def get_wnid(word):
    l = wn.synsets(word)
    if not l:
        return '404'
 
    offset = l[0].offset()
    wnid = "n{:08d}".format(offset)
    return wnid 

def get_urls(wnid):
    driver = webdriver.Chrome()
    driver.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='+wnid)
    search = driver.find_element_by_tag_name('body')
    res = search.text
    driver.close()

    url_list = res.split('\n')
    url_list = list(filter(lambda item: re.match("(http|https)://", item), url_list))

    return url_list

def store_urls(urlset, foldername, term='img'):
    original_path = os.getcwd()
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    
    os.chdir(foldername)

    c = 0
    for url in urlset:
        try: 
            res = urllib.request.urlopen(url)
            img = np.asarray(bytearray(res.read()), dtype='uint8')
            img = cv.imdecode(img, cv.IMREAD_COLOR)
        except Exception:
            # Invalid/broken url
            continue

        if not (img is None):
            c += 1
            filename = f'{term}_{str(c)}.jpg'
            cv.imwrite(filename, img)
        else: 
            print('Empty img =>', url)
            break
    print(len(os.listdir('.')), 'images succesfully stored in ', foldername)

    os.chdir(original_path)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('No search term was provided.')
    else: 
        search_term = sys.argv[1]

        wnid = get_wnid(search_term)
        print("wnid for given term: ", search_term, " => ", wnid)

        urls = np.array(get_urls(wnid))
        np.random.shuffle(urls)

        test_size = int(np.floor(len(urls) * 0.2))
        test = urls[:test_size]
        train = urls[test_size:]

        print("Storing training set with => ", len(train), " images")
        print("Storing training set with => ", len(test), " images")

        store_urls(urlset = test, foldername = './test', term = search_term)
        store_urls(urlset = train, foldername = './train', term = search_term)
