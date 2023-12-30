from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages
# Importing essential libraries and functions

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array
from keras.preprocessing.text import one_hot, Tokenizer
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import io
import urllib, base64


import nltk
nltk.download('stopwords')

import nltk
nltk.download('punkt')


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''

    return TAG_RE.sub('', text)


def preprocess_text(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''
    
    sentence = sen.lower()

    # Remove html tags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

# Embedding layer expects the words to be in numeric form 
# Using Tokenizer function from keras.preprocessing.text library
# Method fit_on_text trains the tokenizer 
# Method texts_to_sequences converts sentences to their numeric form

word_tokenizer = Tokenizer()

# Load the tokenizer from file
with open('D:\\myproject using LSTM\\myapp\\tokenizer.pkl', 'rb') as f:
    word_tokenizer = pickle.load(f)


# Padding all reviews to fixed length 100
maxlen = 100


# Define API key and service name
DEVELOPER_KEY = "AIzaSyB911GXE84hCHzX19pTye7e5bJ71R39tfg"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Define the video ID for which you want to analyze the comments
# video_id = "gTj2OWd5YnM"
# video_id = "BHYtlORK508"

# Initialize the YouTube API client
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

# Define a function to extract the comments from a video
def extract_comments(video_id):
    comments = []
    nextPage_token = None
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=nextPage_token
        ).execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        if "nextPageToken" in response:
            nextPage_token = response["nextPageToken"]
        else:
            break
    return comments


def classify_sentiment(score):
  if score < 0.5:
        return "negative"
  elif score > 0.5:
        return "positive"
  else:
      return "neutral"



# Create your views here.
def index(request):
    if request.method == 'POST':
        video_id=request.POST.get('name')
        # video_id="BHYtlORK508"

        if "youtube.com/watch?v=" in video_id:
            video_id = video_id.split("youtube.com/watch?v=")[1].split("&")[0]
        elif "youtube.com/shorts/" in video_id:
            video_id = video_id.split("youtube.com/shorts/")[1]
        elif "youtu.be/" in video_id:
            video_id = video_id.split("youtu.be/")[1].split("?")[0]
        else:
            # Show a warning message for invalid YouTube link
            messages.warning(request, "Invalid YouTube link. Please enter a valid link.")
            return redirect('index')
        print(video_id)
        

        # Call the extract_comments function to get the comments
        comments = extract_comments(video_id)

        # Convert the comments into a Pandas dataframe
        df = pd.DataFrame(comments, columns=["comment"])

        print(df)


        # Clean the comments using the clean_comments function
        df["comment"] = df["comment"].apply(lambda x: preprocess_text(x))

        # Tokenising instance with earlier trained tokeniser
        unseen_tokenized = word_tokenizer.texts_to_sequences(df["comment"])

        # Pooling instance to have maxlength of 100 tokens
        unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=maxlen)

        from tensorflow import keras
        lstm = keras.models.load_model('D:\\myproject using LSTM\\myapp\\c1_lstm_model_acc_0.868.h5')

        # Passing tokenised instance to the LSTM model for predictions
        unseen_sentiments = lstm.predict(unseen_padded)


        sentiment_labels = [classify_sentiment(score) for score in unseen_sentiments]
        # Create a new DataFrame with sentiment_label and comment columns
        df = pd.DataFrame({"sentiment_label": sentiment_labels, "comment": df["comment"]})


        # Calculate the percentage of positive, negative, and neutral comments
        if "positive" in df["sentiment_label"].unique():
            positive_pct = df["sentiment_label"].value_counts(normalize=True)["positive"] * 100
        else:
            positive_pct = 0

        if "negative" in df["sentiment_label"].unique():
            negative_pct = df["sentiment_label"].value_counts(normalize=True)["negative"] * 100
        else:
            negative_pct = 0


        # Print positive comments
        positive_comments = df[df["sentiment_label"] == "positive"]["comment"].tolist()
        print("Positive comments:")
        for comment in positive_comments:
            print(comment)

        # Print negative comments
        negative_comments = df[df["sentiment_label"] == "negative"]["comment"].tolist()
        print("Negative comments:")
        for comment in negative_comments:
            print(comment)


        # Print the results
        print("Positive comments: {:.2f}%".format(positive_pct))
        print("Negative comments: {:.2f}%".format(negative_pct))

        plt.figure()
        # plot results in a pie chart
        labels = ['Positive', 'Negative']
        sizes = [positive_pct, negative_pct]
        colors = ['green', 'red']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Sentiment Analysis Results for YouTube Video Comments')


        # # Save the plot to a file
        # graph_image = 'D:\\myproject\\static\\plot.png'
        # plt.savefig(graph_image)

        # Save the graph image to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode the graph image to base64
        graph_image = base64.b64encode(buffer.getvalue()).decode()

        # Close the plot to free up memory
        plt.close()

        positive_pct=round(positive_pct, 2)
        negative_pct=round(negative_pct, 2)

        context={
            'processing_done': True,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct, 
            'positive_comments': positive_comments,
            'negative_comments': negative_comments,
            'graph_image': graph_image,  # Adding the graph image to the context
            }
        
        return render(request, 'index.html', context)
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')

def services(request):
    return render(request, 'services.html')

def dynamic_url(request, url_end):
    if url_end == 'about':
        return about(request)
    elif url_end == 'services':
        return services(request)
    elif url_end == 'index':
        return index(request)
    else:
        # Handle invalid URL endings
        return render(request, '404.html')