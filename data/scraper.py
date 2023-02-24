import requests
from bs4 import BeautifulSoup
import json
import ast
import time
import os

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


def save_fragrance_links():
    """
    Extract names and links for all fragrances and save them as json in dict format:
    {designer_name: [fragrance_name1, fragrance_name2, ...], ...}
    Website blocks too many consecutive requets. You should run this function multiple times
    until all of the data is collected.
    """
    href_dataset = {}

    designer_hrefs, designers = extract_designers()
    f = open("data/links/href_dataset.json")
    href_dataset = json.load(f)
    f.close()

    for ref, name in zip(designer_hrefs, designers):
        if len(href_dataset[name]) == 0:
            print(f"{name}, {ref}")
            href_dataset[name] = extract_fragrances_designer(ref)
            print(f"Finished: {len(href_dataset[name])}")

    json_dataset = json.dumps(href_dataset)
    f = open("data/links/href_dataset.json", "w")
    f.write(json_dataset)
    f.close()


def save_page_binary_content():
    """
    Extract response from file and save binary content to different files
    """
    f = open("data/links/hrefs.txt", "r")
    link_list_txt = f.read()
    link_list_txt = link_list_txt.replace('\n', '')
    link_list = ast.literal_eval(link_list_txt)
    print(len(link_list))

    for i, link in enumerate(link_list):
        response = requests.get(link, headers=headers)

        if response.status_code == 200:
            print(link.split('/')[-1][:-5])
            f = open(f"data/pages/{link.split('/')[-1][:-5]}.xls", "wb")
            f.write(response.content)
            f.close()
        
        else:
            print("Wait for 10 mins!")
            time.sleep(600)
            response = requests.get(link, headers=headers)
            print(link.split('/')[-1][:-6])
            f = open(f"data/pages/{link.split('/')[-1][:-5]}.xls", "wb")
            f.write(response.content)
            f.close()


def read_frag_content(f_name):

    f = open(f"data/pages/{f_name}", "rb")
    content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    comment = soup.select("#dd1766331 > div:nth-child(2) > div:nth-child(1)")
    print(comment)



# save_page_binary_content()

read_frag_content("Acqua-di-Parma-Colonia-1681.xls")
