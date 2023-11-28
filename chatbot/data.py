import json

# TODO: Group similar perfumes into the same document
def create_frag_template(name: str) -> dict:
    # Read data from son format
    dict = json.load(open(name, "r"))
    # Transform data into a text document
    frag_doc = ""
    frag_doc += f"{dict['name']} is from the {dict['brand']} brand.\n"
    frag_doc += f"{dict['name']} is for {dict['gender']}.\n"
    frag_doc += dict['about']
    frag_doc += dict['faq'].split('\n')[1:]
    for k, v in dict['prices'].items():
        frag_doc += f"You can buy {dict['name']} {k} for {v}.\n"
    frag_doc += f"The rating for {dict['name']} is {dict['rating']} by {dict['rating_count']} many people.\n"
    for review in dict['reviews'].items():
        if int(review['helpful']) - int(review['unhelpful']) >= 0:
            frag_doc += f"{review['comment']}\n"
    
    return frag_doc

