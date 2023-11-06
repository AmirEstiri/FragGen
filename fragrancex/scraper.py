import json
from tqdm import tqdm
import time
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By

def clean_white_spaces(s):
    return re.sub('\s+', ' ', s)

### Header ###
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
}

def save_fragrance_links():
    """
    Extract names and links for all fragrances.
    """
    url = lambda page: f'https://www.fragrancex.com/search/search_results?currentPage={page}&searchSortExpression=2'
    href_dataset = {}
    browser = webdriver.Firefox()
    browser.get('http://selenium.dev/')
    browser.get(url(1))
    time.sleep(10)
    for j in range(2, 399):
        print(f'Page: {j}, {len(href_dataset)}')
        for i in range(50):
            try:
                frag_link = browser.find_element(By.CSS_SELECTOR, f'div.c3-4-of-12:nth-child({i+1}) > div:nth-child(1) > div:nth-child(1) > a:nth-child(1)').get_attribute('href')
                name = frag_link.split('/')[-1]
                href_dataset[name] = frag_link
            except:
                pass
        browser.get(url(j)) 
        time.sleep(5)
    browser.close()
    json_dataset = json.dumps(href_dataset)
    os.makedirs(os.path.dirname("fragrancex/links"), exist_ok=True)
    f = open("fragrancex/links/href_dataset.json", "w")
    f.write(json_dataset)
    f.close()

def extract_all_perfume_data():
    os.makedirs(os.path.dirname("fragrancex/fragrances"), exist_ok=True)
    links = json.load(open('fragrancex/links/href_dataset.json', 'r'))
    for i, (name, link) in enumerate(links.items()):
        print(f'Extracting {i+1}/{len(links)}: {name}')
        extract_fragrance_data(link)


def extract_fragrance_data(url):
    """
    Extract information of fragrance from its link.
    """
    frag_data = {}
    browser = webdriver.Firefox()
    browser.get('http://selenium.dev/')
    browser.get(url)
    time.sleep(5)

    # Extract name of perfume
    try:
        perfume_name = browser.find_element(By.CSS_SELECTOR, f'.perfume-name').get_attribute('innerHTML')
    except:
        print('Perfume name not found')
    frag_data['name'] = perfume_name

    # Extract brand and gender
    try:
        brand_name = browser.find_element(By.CSS_SELECTOR, f'.ga-product-brand').get_attribute('innerHTML')
        gender = browser.find_element(By.CSS_SELECTOR, f'.brand-name').get_attribute('innerHTML').split(' ')[-1]
    except:
        print('Brand/gender not found')
    frag_data['brand'] = brand_name
    frag_data['gender'] = gender[:-1]

    # Extract about description
    about = ""
    try:
        about_element = browser.find_element(By.CSS_SELECTOR, f'.faq-description')
        for txt in about_element.find_elements(By.CSS_SELECTOR, f'*'):
            about += txt.get_attribute('innerHTML')
            about += '\n'
    except:
        print('About not found')
    frag_data['about'] = about[:-1]

    # Extract FAQ description
    faq = ""
    try:
        
        faq_element = browser.find_element(By.CSS_SELECTOR, f'.faq-info > section:nth-child(2)')
        for txt in faq_element.find_elements(By.CSS_SELECTOR, f'*'):
            faq += txt.get_attribute('innerHTML')
            faq += '\n'
    except:
        print('FAQ not found')
    frag_data['faq'] = faq[:-1]

    # Extract attributes
    attributes = {}
    try:
        trs = browser.find_element(By.CSS_SELECTOR, f'.table').find_elements(By.TAG_NAME, 'tr')
        for tr in trs:
            tds = tr.find_elements(By.TAG_NAME, 'td')
            if len(tds) > 0:
                k = clean_white_spaces(tds[0].get_attribute('innerHTML'))[1:-1]
                v = clean_white_spaces(tds[1].get_attribute('innerHTML'))[1:-1]
                attributes[k] = v
    except:
        print('Table not found')

    #TODO: Extract rating
    #TODO: Extract number of ratings
    #TODO: Extract reviews

    browser.close()
    f = open(f"fragrancex/fragrances/{url.split('/')[-1]}.json", "w")
    f.write(json.dumps(frag_data))
    f.close()

# links = json.load(open('fragrancex/links/href_dataset.json', 'r'))
# extract_fragrance_data('https://www.fragrancex.com/products/armaf/club-de-nuit-intense-cologne')
extract_all_perfume_data()