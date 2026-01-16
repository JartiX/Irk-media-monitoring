import re
import config
from filters.keywords import KeywordFilter

while True:
    text = input("COMM: ")

    kw = KeywordFilter()

    print(kw.check_comment_usefulness(text))