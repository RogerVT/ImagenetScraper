import cv2 as cv
from nltk.corpus import wordnet as wn
from selenium import webdriver 
import numpy as np
import os
import urllib

def get_wnid(word):
    l = wn.synsets(word)
    if not l:
        print('here')
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
    if res == '':
        print('No links available!')
        return []

    return res.split('\n')

def store_urls(urlset, foldername, term='img'):
    os.mkdir(foldername)
    os.chdir(foldername)
    c = 0
    for url in urlset:
        res = urllib.request.urlopen(url)
        img = np.asarray(bytearray(res.read()), dtype='uint8')
        img = cv.imdecode(img, cv.IMREAD_COLOR)
        if not (img is None):
            c+=1
            filename = term+'_'+str(c)+'.jpg'
            cv.imwrite(filename, img)
        else: 
            print('Empty img =>', url)
            break
    print(len(os.listdir(foldername)), 'images succesfully stored in ', foldername)


search_term = 'dog'
wnid = get_wnid(search_term)
print("wnid for given term: ", search_term, " => ", wnid)
urls = np.array(get_urls(wnid))
np.random.shuffle(urls)
test_size = int(np.floor(len(urls) * 0.2))
test = urls[:test_size]
train = urls[test_size:]
print("Storing training set with => ", len(train), " images")
print("Storing training set with => ", len(test), " images")
store_urls(urlset=test, foldername='./test', term=search_term)
