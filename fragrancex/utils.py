import re

def clean_white_spaces(s):
    return re.sub('\s+', ' ', s)