import requests
from bs4 import BeautifulSoup
import json

### Global URL and Header ###
url = 'https://www.fragrantica.com/'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
}

def extract_designers():
    """
    Extract name and link for all designers from fragrantica.com
    """
    designers = []
    designer_hrefs = []

    response = requests.get(url+"designers/", headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    for i in range(2, 122):
        designer_element = str(soup.select(f'div.designerlist:nth-child({i}) > a:nth-child(2)')[0])
        designer_href = designer_element.split()[1]
        designer_hrefs.append(url+designer_href[7:-2])
        designers.append(designer_href[17:-7])
    
    return designer_hrefs, designers



def extract_fragrances_designer(href):
    """
    Extract all fragrances from a designer
    """
    frag_hrefs = []

    response = requests.get(href, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    index = 3
    empty = False
    while True:
        if index % 100 == 0:
            print(index)
        frag_element = \
            soup.select(f'div.text-left:nth-child({index}) > div:nth-child(1) > div:nth-child(3) > h3:nth-child(1) > a:nth-child(1)')

        if len(frag_element) == 0 and empty:
            # If more than one empty consecutive element noticed after extracting fragrances, exit the loop
            break

        if len(frag_element) == 0:
            empty = True

        else:
            empty = False
            frag_href = str(frag_element[0]).split()[1]
            frag_hrefs.append(url+frag_href[7:-2])

        index += 1
    
    return frag_hrefs


def extract_save_links():
    """
    Extract names and links for all fragrances and save them as json in dict format:
    {designer_name: [fragrance_name1, fragrance_name2, ...], ...}
    """
    href_dataset = {}

    designer_hrefs, designers = extract_designers()

    for ref, name in zip(designer_hrefs, designers):
        href_dataset[name] = extract_fragrances_designer(ref)
        print(f"Finished: {len(href_dataset[name])}")

    json_dataset = json.dumps(href_dataset)
    f = open("href_dataset.json", "w")
    f.write(json_dataset)
    f.close()


def extract_comments(frag_ref):
    comments = []

    response = requests.get(frag_ref, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # "#dd1765803 > div:nth-child(2) > div:nth-child(1)"
    # "#dd1723982 > div:nth-child(2) > div:nth-child(1)"
    # "#dd1723016 > div:nth-child(3) > div:nth-child(1)"
    # "#dd1660041 > div:nth-child(2) > div:nth-child(1)"
    # "#dd2165 > div:nth-child(2) > div:nth-child(1)"
    


extract_save_links()