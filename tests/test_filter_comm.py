import re
import config
from filters.keywords import KeywordFilter
from database import Comment

while True:
    text = input("COMM: ")
    comment = Comment(
        "0",
        "0",
        text
    )

    kw = KeywordFilter()

    kw.filter_comments([comment])