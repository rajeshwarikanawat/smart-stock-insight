from newspaper import Article
from textblob import TextBlob

def analyze_sentiment(news_urls):
    sentiments = []
    for url in news_urls:
        try:
            article = Article(url)
            article.download()
            article.parse()

            analysis = TextBlob(article.text)
            sentiments.append(analysis.sentiment.polarity)
        except:
            continue
    if len(sentiments)==0:
        return 0
    return sum(sentiments)/len(sentiments)