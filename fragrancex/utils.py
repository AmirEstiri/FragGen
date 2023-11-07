import re

def clean_white_spaces(s):
    return re.sub('\s+', ' ', s)

def clean_html(s):
    return re.sub('<[^<]+?>', '', s)