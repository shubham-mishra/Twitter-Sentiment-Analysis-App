
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re
import nltk
import bs4
import os
import json
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from collections import Counter
from io import BytesIO
import base64
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
from sklearn.feature_extraction.text import TfidfVectorizer


class Analysis():
    def __init__(self,):

        #getting keys from configuration file
        with open(os.path.dirname(os.path.abspath(__file__))+'/../files/config.json') as config_file:
            self.keys = json.load(config_file)
        
        
        #getting stopwords
        self.STOP_WORDS = stopwords.words('english');
        self.STOP_WORDS.remove('not');

        #defining external stopwords
        self.words_to_remove = []
        f = open(os.path.dirname(__file__)+'/../files/remove_words.txt')
        for word in f.read().split():
            self.words_to_remove.append(word)

        #defining list to hold tweets and noun phrases
        self.positive_tweets = []
        self.negative_tweets = []
        self.positive_noun_tweets = []
        self.negative_noun_tweets = []

        #defining list holding no of unique words present only in list before lemmatization
        self.unique_only_before_lemmatization = []
        

    def decontraction(self, text):
        patterns = [
        ('won\'t', 'will not'), ('\'d', ' would'), ('\'s', ' is'), ('can\'t', 'can not'),
        ('don\'t', 'do not'), ('\'ll', ' will'), ('\'ve', ' have'), ('\'t', ' not'),
        ('\'re', ' are'), ('\'m', ' am')
        ]
        
        for (pattern, replacer) in patterns:
            regex = re.compile(pattern)
            text = regex.sub(replacer, text)
        return text

    def get_wordnet_pos(self, tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN
            
    def lemmatize(self, text):
        """
        Function to lemmatize words in provided sentence
        """
        words_before_lemmatization = set(text.split())
        pos = nltk.pos_tag(text.split())
        lemmatize_words = [];
        lemmatizer = WordNetLemmatizer()
        #return ' '.join([lemmatizer.lemmatize(word_pos[0], get_wordnet_pos(word_pos[1])) for word_pos in pos])
        for word_pos in pos:
            lemmatize_words.append(lemmatizer.lemmatize(word_pos[0], self.get_wordnet_pos(word_pos[1]))) 
            
        words_after_lemmatization = set(lemmatize_words)
        words_only_in_before_lemmatization = words_before_lemmatization - words_after_lemmatization
        self.unique_only_before_lemmatization.append(len(words_only_in_before_lemmatization))
        # print('There are {} less no of unique words present after lemmatization '.format(len(words_only_in_before_lemmatization)))   
        return ' '.join(lemmatize_words)



    def process_text(self, text, type_clean='not_for_noun'):
                
        text = text.lower()

        #performing decontraction
        text = self.decontraction(text)
        
        #removing urls
        text = re.sub(r"http\S+", '', text)

        #removing html tags
        if type(text) == type(''):
            bs = bs4.BeautifulSoup(text, "lxml") #removing html tags
            text = bs.get_text()
            
        #removing @user ids
        text = re.sub(r'@[\w]*', ' ', text)
        
        #removing special characters
        text = re.sub('[^a-z0-9\s]', ' ', text)
        
        #removing words with numbers
        text = re.sub('\S*\d\S*', ' ', text)
        
        #removing \n \r \t
        regex = re.compile(r'[\n\r\t]')
        text = regex.sub('', text)
        
        #removing roman numbers
        ProgRoman = re.compile(u'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$')
        text = ProgRoman.sub(' ', text)
        
        #lemmatizing words
        text = self.lemmatize(text)
        if type_clean == 'not_for_noun':
            #removing stop words except not, because not helps in finding negative sentiments
            text = ' '.join([word for word in text.split(' ') if word.lower() not in self.STOP_WORDS])

            #removing words having lenght <=2
            text = ' '.join([word for word in text.split(' ') if len(word) > 2])
    
        return text.strip();

    def generate_clean_tweets(self,):
        """
        Function to process tweets and generates clean tweets and drop duplicate tweets
        """
        print("preprocessing obtained tweets")
        self.data['clean_tweets'] = self.data['tweets'].fillna('').apply(self.process_text)

        #emptying unique_only_before_lemmatization so that we can obtain unique words only for current tweets
        self.unique_only_before_lemmatization = []
        self.data['partial_clean_tweets'] = self.data['tweets'].fillna('').apply(lambda x: self.process_text(x, 'for_noun'))
        print('There are {} no of less unique words present after lemmatization'.format(sum(self.unique_only_before_lemmatization)))

        self.data.drop_duplicates(subset ="clean_tweets", keep = 'first', inplace = True)


    def plot_word_cloud(self,pos_tweets, neg_tweets, keyword1, keyword2, type_of_wc):
        """
        Function to plot wordCloud for given tweets
        """
        print('generating word cloud for analysed tweets')
        if type_of_wc == 'noun_phrases':
            #collecting words for each tweet in tweets_word list
            pos_tweets_word = pos_tweets
            neg_tweets_word = neg_tweets
        else:
            #collecting words for each tweet in tweets_word list
            pos_tweets_word = [word for tweet in pos_tweets for word in tweet.split(' ')] 
            neg_tweets_word = [word for tweet in neg_tweets for word in tweet.split(' ')] 
            
        
        images = dict()
        if pos_tweets != None and len(pos_tweets_word) > 0:
            #configuring wordcloud
            wc = WordCloud(background_color="white", max_words=len(pos_tweets_word), collocations=False)

            #combining tweet_word
            combined_tweets = ' '.join(pos_tweets_word)

            #generating wordcloud
            plt.figure(figsize=(20,10))

            wc.generate(combined_tweets)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            
            img_name = keyword1+keyword2+'positive.png'
            
            # plt.savefig(os.path.dirname(__file__)+"/../images/"+img_name)
            figfile = BytesIO()
            plt.savefig(figfile, format='png')
            # wc.to_file(os.path.dirname(__file__)+"/../images/"+img_name).save(figfile, 'png')
            figfile.seek(0)
            figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
            images['positive'] = figdata_png

        
        
        if neg_tweets != None and len(neg_tweets_word) > 0:
            #configuring wordcloud
            wc = WordCloud(background_color="white", max_words=len(neg_tweets_word), collocations=False)
            #combining tweet_word
            combined_tweets = ' '.join(neg_tweets_word)
            #generating wordcloud
            wc.generate(combined_tweets)
            plt.figure(figsize=(20,10))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            img_name = keyword1+keyword2+'negative.png'

            # plt.savefig(os.path.dirname(__file__)+"/../images/"+img_name)
            figfile2 = BytesIO()
            plt.savefig(figfile2, format='png')
            figfile2.seek(0)
            figdata_png = base64.b64encode(figfile2.getvalue()).decode('ascii')
            images['negative'] = figdata_png
        return images

    def get_noun_phrase(self, tweets):
        # noun_phrase = [ '_'.join(noun_phrase.split(' ')) for tweet in tweets for noun_phrase in TextBlob(tweet).noun_phrases]
        noun_phrases = [ noun_phrase for tweet in tweets for noun_phrase in TextBlob(tweet).noun_phrases]
        
        clean_phrases = []
        for noun_phrase in noun_phrases:
            text = ''
            for word in noun_phrase.split():
                if len(word) > 2:
                    text = text + word + '_'
            if len(text) > 2 and text[-1] == '_':
                text = text[:-1]
            if len(text) > 2:
                clean_phrases.append(text)
        return clean_phrases


    def find_sentiments(self, keyword1, keyword2):
        """
        Function to find sentiments for each player and plot wordCloud for each sentiments
        """
        print("performing sentiment analysis")

        self.positive_tweets = []
        self.negative_tweets = []
        self.positive_noun_tweets = []
        self.negative_noun_tweets = []
        
        positive_partial_clean_tweets = []
        negative_partial_clean_tweets = []

        #creating variable which contains counts for each type of sentiments
        positive_tweets_counts = negative_tweets_counts = neutral_tweets_counts = 0
        
        #fetching tweets for given kewords
        mask = self.data['tweets'].str.contains(keyword1, case=False) & self.data['tweets'].str.contains(keyword2, case=False)
        
        tweets = self.data[mask]['clean_tweets']
        partial_clean_tweets = self.data[mask]['partial_clean_tweets']
        
        itr = 0
        
        #finding sentiment score for tweets fetched for particular player
        for tweet in tweets:
            sentiment_score = TextBlob(tweet).sentiment.polarity;

            if sentiment_score >= 0.02: #if compound score is >= 0.02 then that tweet is considered to be positive tweet
                self.positive_tweets.append(tweet)
                positive_partial_clean_tweets.append(partial_clean_tweets.iloc[itr])
                positive_tweets_counts += 1
            elif sentiment_score <= -0.02: #if compound score <= -0.02 then that tweet is considered to be negative tweet#if compound score is > -0.02 and <0.02 then that tweet is considered to be neutral tweet
                self.negative_tweets.append(tweet)
                negative_partial_clean_tweets.append(partial_clean_tweets.iloc[itr])
                negative_tweets_counts += 1
            else: #if compound score is > -0.02 and <0.02 then that tweet is considered to be neutral tweet
                neutral_tweets_counts += 1
            
            itr += 1
            
            if(positive_tweets_counts > 0):
                positive_noun_phrases = self.get_noun_phrase(positive_partial_clean_tweets)
                self.positive_noun_tweets.extend(positive_noun_phrases)
            if(negative_tweets_counts > 0):
                negative_noun_phrases = self.get_noun_phrase(negative_partial_clean_tweets)
                self.negative_noun_tweets.extend(negative_noun_phrases)

    # stopwords_counter = Counter()
    def remove_manual_stopwords(self, tweets):
        tweet_list = [tweet for tweet in tweets if tweet.lower() not in self.words_to_remove]
        # stopwords_counter.update([tweet for tweet in tweets if tweet.lower() in words_to_remove])    
        return tweet_list

    def fetch_twitter_data(self, keyword1, keyword2):
        print('fetching tweets')
        keyword = keyword1 + ' ' + keyword2
        auth = tweepy.OAuthHandler(self.keys['CONSUMER_KEY'], self.keys['CONSUMER_SECRET'])
        auth.set_access_token(self.keys['ACCESS_TOKEN'], self.keys['ACCESS_TOKEN_SECRET'])
        api = tweepy.API(auth)
        tweets_data = tweepy.Cursor(api.search, q=keyword, lang = "en").items(1000)
        tweets = []

        #fetching sentences from tweets and appending it to tweets list
        for tweet in tweets_data:
            blob = TextBlob(tweet.text)
            for sent in blob.sentences:
                tweets.append(str(sent))

        print('fetched {} tweets'.format(len(tweets)))
        self.data = pd.DataFrame(tweets, columns=['tweets'])

        self.generate_clean_tweets()

        return len(tweets)

    def remove_most_common_words(self,):
        #finding most common words in positive_noun_tweets
        common_words = pd.Series(self.positive_noun_tweets).value_counts()[:1]
        common_words_list = list(common_words.index)
        if len(common_words_list) > 0:
            common_word = common_words_list[0]
            print('Common Word Present in Positive Tweets: ',common_word)
            self.positive_noun_tweets = [noun for noun in self.positive_noun_tweets if common_word not in noun]
        
        #finding most common words in negative_noun_tweets
        common_words = pd.Series(self.negative_noun_tweets).value_counts()[:1]
        common_words_list = list(common_words.index)
        if len(common_words_list) > 0:
            common_word = common_words_list[0]
            print('Common Word Present in Negative Tweets: ',common_word)
            self.negative_noun_tweets = [noun for noun in self.negative_noun_tweets if common_word not in noun]


    def analyse_tweets(self, keyword1, keyword2):

        for_players = False
        if for_players:
            tweets_obtained = 1
            self.data = pd.read_csv(os.path.dirname(__file__)+"/../files/twitter_data.csv", encoding = "ISO-8859-1")
            #renaming column Text to tweets
            self.data.rename(columns={'Text':'tweets'}, inplace=True)
            #preprocessing tweets
            self.generate_clean_tweets()

        else:
            tweets_obtained = self.fetch_twitter_data(keyword1, keyword2)
        
        obj = dict()
        if tweets_obtained == 0:
            obj['no_tweets'] = True
        else:
            self.find_sentiments(keyword1, keyword2)

            positive_tweets_counts = len(self.positive_tweets)
            negative_tweets_counts = len(self.negative_tweets)
            positive_noun_tweets_counts = len(self.positive_noun_tweets)
            negative_noun_tweets_counts = len(self.negative_noun_tweets)

            if for_players:
                #obtaining noun phrases after removing words which are not relevant
                clean_positive_noun = self.remove_manual_stopwords(self.positive_noun_tweets)
                clean_negative_noun = self.remove_manual_stopwords(self.negative_noun_tweets)
            else:
                self.remove_most_common_words()
                clean_positive_noun = self.positive_noun_tweets
                clean_negative_noun = self.negative_noun_tweets

            images_dict = self.plot_word_cloud(clean_positive_noun, clean_negative_noun, keyword1, keyword2, 'noun_phrases')

            obj = {
            'positive_tweets_counts': positive_tweets_counts,
            'negative_tweets_counts': negative_tweets_counts,
            'positive_noun_tweets_counts': positive_noun_tweets_counts,
            'negative_noun_tweets_counts': negative_noun_tweets_counts,
            'no_tweets': False
            }

            if images_dict.get('positive') != None:
                obj['positive_image'] = images_dict.get('positive')

            if images_dict.get('negative') != None:
                obj['negative_image'] = images_dict.get('negative')

        return obj