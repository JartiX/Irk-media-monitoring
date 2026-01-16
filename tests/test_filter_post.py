from filters.keywords import KeywordFilter
from filters.ml_classifier import initialize_classifier
from database.models import Post

kw_f = KeywordFilter()
ml_f = initialize_classifier()

while True:
    text = input("TEXT: ")

    post = Post(
        "0", 
        "2", 
        text,
        "NONE",
        None)
    posts = [post]

    posts = kw_f.filter_posts(posts)

    print(posts[0].is_relevant, posts[0].relevance_score)

    posts = ml_f.classify_posts(posts)

    print(posts[0].is_relevant, posts[0].relevance_score)

