import json
from tqdm import tqdm
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from utils import clean_white_spaces, clean_html

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
    os.makedirs("fragrancex/links", exist_ok=True)
    f = open("fragrancex/links/href_dataset.json", "w")
    f.write(json_dataset)
    f.close()


def extract_all_perfume_data():
    os.makedirs("fragrancex/fragrances", exist_ok=True)
    extracted_fragrances = os.listdir("fragrancex/fragrances")
    links = json.load(open('fragrancex/links/href_dataset.json', 'r'))
    for i, (name, link) in enumerate(links.items()):
        if f'{name}.json' in extracted_fragrances:
            continue
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
    name = ""
    try:
        name = browser.find_element(By.CSS_SELECTOR, f'.perfume-name').get_attribute('innerHTML')
    except:
        print('Perfume name not found')
    frag_data['name'] = name

    # Extract brand and gender
    gender = ""
    brand = ""
    try:
        brand = browser.find_element(By.CSS_SELECTOR, f'.ga-product-brand').get_attribute('innerHTML')
        gender = browser.find_element(By.CSS_SELECTOR, f'.brand-name').get_attribute('innerHTML').split(' ')[-1]
    except:
        pass
        # print('Brand/gender not found')
    frag_data['brand'] = brand
    frag_data['gender'] = gender[:-1]

    # Extract about description
    about = ""
    try:
        about_element = browser.find_element(By.CSS_SELECTOR, f'.faq-description')
        for txt in about_element.find_elements(By.CSS_SELECTOR, f'*'):
            about += txt.get_attribute('innerHTML')
            about += '\n'
    except:
        pass
        # print('About not found')
    frag_data['about'] = about[:-1]

    # Extract FAQ description
    faq = ""
    try:
        faq_element = browser.find_element(By.CSS_SELECTOR, f'.faq-info > section:nth-child(2)')
        for txt in faq_element.find_elements(By.CSS_SELECTOR, f'*'):
            faq += txt.get_attribute('innerHTML')
            faq += '\n'
    except:
        pass
        # print('FAQ not found')
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
        pass
        # print('Table not found')
    frag_data['attributes'] = attributes

    #Extract price
    prices = {}
    try:
        products = browser.find_elements(By.CSS_SELECTOR, f'.product')
        for product in products:
            listing_price = float(product.find_element(By.CSS_SELECTOR, f'.sale-price-val').get_attribute('innerHTML'))
            listing_description = clean_html(product.find_element(By.CSS_SELECTOR, f'.listing-description').get_attribute('innerHTML'))[1:-1]
            prices[listing_description] = listing_price
    except:
        pass
        # print('Price not found')
    frag_data['prices'] = prices

    # Extract rating and number of ratings
    rating = 0.0
    rating_count = 0.0
    try:
        rating_count = int(browser.find_element(By.CSS_SELECTOR, f'div.review-count:nth-child(3)').get_attribute('innerHTML').split(' ')[0])
        rating = float(browser.find_element(By.CSS_SELECTOR, f'.review-snippet > div:nth-child(1)').get_attribute('innerHTML'))
    except:
        pass
        # print('Rating not found')
    frag_data['rating'] = rating
    frag_data['rating_count'] = rating_count

    #Extract reviews
    reviews = []
    try:
        start = 0
        while True:
            print(len(browser.find_elements(By.CSS_SELECTOR, f'.review')))
            reviews_div = browser.find_elements(By.CSS_SELECTOR, f'.review')[start:]
            for review_div in reviews_div:
                review_text_div = review_div.find_element(By.CSS_SELECTOR, f'.review-text')
                headline = review_text_div.find_element(By.CSS_SELECTOR, f'.headline').get_attribute('innerHTML')
                comment = review_text_div.find_element(By.CSS_SELECTOR, f'.comment').get_attribute('innerHTML')
                reviews.append({
                    'headline': headline,
                    'comment': comment
                })
            start = len(reviews)
            browser.find_element(By.CSS_SELECTOR, f'button.load-more-reviews').click()
            time.sleep(1)
    except:
        pass
        # print('Reviews not found')
    frag_data['reviews'] = reviews

    browser.close()
    f = open(f"fragrancex/fragrances/{url.split('/')[-1]}.json", "w")
    f.write(json.dumps(frag_data))
    f.close()

# links = json.load(open('fragrancex/links/href_dataset.json', 'r'))
extract_all_perfume_data()