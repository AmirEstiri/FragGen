import requests
from bs4 import BeautifulSoup
import json
import ast
import time
import os
import re

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

    for i, f_name in enumerate(os.listdir("data/pages/")):
        for l in link_list:
            if re.search(f"^.*{f_name[:-4]}.*", l):
                link_list.remove(l)

    print(len(link_list))

    for i, link in enumerate(link_list):
        retry = True

        while retry:
            response = requests.get(link, headers=headers)

            if response.status_code == 200:
                retry = False
                print(link.split('/')[-1][:-5])
                f = open(f"data/pages/{link.split('/')[-1][:-5]}.xls", "wb")
                f.write(response.content)
                f.close()
            
            else:
                print("Wait for 10 mins!")
                time.sleep(600)


def extract_frag_data(f_name):

    f = open(f"data/pages/{f_name}", "rb")
    content = f.read()
    soup = BeautifulSoup(content, 'html.parser')

    # Extract notes
    frag_notes = {'top': [], 'middle': [], 'base': [], 'all': []}
    element = soup.select("#pyramid > div:nth-child(1) > div:nth-child(1)")
    if len(element) > 0:
        element = element[0]
        l = len(element.contents[1])-1
        if l == 1:
            for e in element.contents[1].contents[1].contents[0]:
                frag_notes['all'].append(e.contents[1].contents[1])
        else:
            for i in range(1, l):
                for e in element.contents[1].contents[i].contents[0]:
                    if e == 'Top Notes':
                        for e in element.contents[1].contents[i+1].contents[0]:
                            frag_notes['top'].append(e.contents[1].contents[1])
                for e in element.contents[1].contents[i].contents[0]:
                    if e == 'Middle Notes':
                        for e in element.contents[1].contents[i+1].contents[0]:
                            frag_notes['middle'].append(e.contents[1].contents[1])
                for e in element.contents[1].contents[i].contents[0]:
                    if e == 'Base Notes':
                        for e in element.contents[1].contents[i+1].contents[0]:
                            frag_notes['base'].append(e.contents[1].contents[1])

    # Extract accords
    frag_accords = {}
    element = soup.select(".bg-white > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > div:nth-child(3)")[0]
    for e in element:
        accord_name = e.contents[0].contents[0]
        accord_value = float(e.contents[0].get('style').split(':')[-1][1:-2])
        frag_accords[accord_name] = accord_value

    # Extract name
    element = soup.select("h1.text-center")[0]
    frag_name = element.contents[0]

    # Extract sex
    element = soup.select("h1.text-center > small:nth-child(1)")[0]
    frag_sex = element.contents[0]

    # Extract rating
    element = soup.select("div.grid-margin-y:nth-child(4)")[0]
    if len(element.contents[2].contents) > 0:
        rating = float(element.contents[2].contents[0].contents[0].contents[1].contents[0])
        total_votes = int(element.contents[2].contents[0].contents[0].contents[5].contents[0].replace(",",""))
    else:
        rating = 0.0
        total_votes = 0

    # Extract similar fragrances
    element = soup.select("#main-content > div.grid-x.grid-margin-x > div.small-12.medium-12.large-9.cell > div")[0]
    
    for j in range(100):
        if re.search("^.*This perfume reminds me of.*", str(element.contents[j])):
            break

    similar_frags = []
    for i, e in enumerate(element.contents[j].contents[0].contents[2]):
        if i%2 == 1:
            similar_frags.append(
                {'Company': e.contents[1].contents[3].contents[0], 'Model': e.contents[1].contents[6][:-1]}
            )

    # Extract designer name
    element = soup.select(".bg-white > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > p:nth-child(1) > a:nth-child(1) > span:nth-child(1)")[0]
    designer = element.contents[0]

    # TODO extract winter, spring, summer, fall, day, night
    # element = soup.select("div.carousel:nth-child(2)")

    # TODO extract comments
    # element = soup.select("#all-reviews")
    # print(element)

    # TODO Extract longevity, sillage, gender, price value
    # element = soup.select(".bg-white > div:nth-child(7)")[0]
    # longevity = float(element.contents[0].contents[3].contents[0].contents[0].contents[1].contents[0]) / float(element.contents[0].contents[3].contents[0].contents[0].contents[3].contents[0])
    # sillage = float(element.contents[0].contents[4].contents[0].contents[0].contents[1].contents[0]) / float(element.contents[0].contents[4].contents[0].contents[0].contents[3].contents[0])
    # price_value = element.contents[0].contents[7].contents[0].contents[0].contents[1].contents[0]
    
    return frag_name, designer, frag_sex, frag_accords, frag_notes, rating, total_votes, similar_frags



def extract_all_fragrances_dataset():

    for i, f_name in enumerate(os.listdir("data/pages/")):
        print(f_name)
        frag_name, designer, frag_sex, frag_accords, frag_notes, rating, total_votes, similar_frags =\
             extract_frag_data(f_name)
        f = open(f"data/fragrances/{f_name[:-4]}.json", "w")
        f.write(
            json.dumps(
                {
                    'name': frag_name, 'designer': designer, 'sex': frag_sex, 
                    'accords': frag_accords, 'notes': frag_notes, 'rating': float(rating), 
                    'total_votes': int(total_votes), 'similar_fragrances': similar_frags
                }
            )
        )
        f.close()
        


# save_page_binary_content()

extract_all_fragrances_dataset()
