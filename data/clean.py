import json


def create_links_list_dataset():
    """
    Extract all links from dictionary, put it in a list and save it
    """
    links_list = []
    
    f = open("data/links/href_dataset.json")
    href_dict = json.load(f)
    f.close()
    
    for v in href_dict.values():
        for link in v:
            links_list.append(link)
    
    f = open("data/links/hrefs.txt", "w")
    f.write(str(links_list))
    f.close()


create_links_list_dataset()