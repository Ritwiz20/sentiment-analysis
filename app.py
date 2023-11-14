from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np 
import pandas as pd 
import os
from dotenv import load_dotenv
from email.message import EmailMessage
import ssl
import smtplib
from datetime import datetime


load_dotenv()


""" Intialization """

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:5500"
]

""" Set up CORS for FastAPI """

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

""" Helpers """

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def map_sentiment_score(review):

    """ Given a review generates the sentiment rating, ranging from 1-5, with 5 being positive """

    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


def worker_function(url):

    """ Accesing the web to get reviews """

    try:
        r = requests.get(url)
        r.raise_for_status()  
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error accessing {url}: {str(e)}")

    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*body-medium-overflow-wrap.*')
    # results = soup.find_all('p', {'class':regex})
    results = soup.find_all('p')
    reviews = [result.text for result in results]

    if not reviews:
        raise HTTPException(status_code=500, detail=f"No reviews found on {url}")


    """ Converting the reviews to a dataframe for easier processing """

    df = pd.DataFrame(np.array(reviews), columns=['reviews'])
    print(df)

    """ Getting the review for each review in our dataframe """

    df['score'] = df['reviews'].apply(lambda x: map_sentiment_score(x))
    print(df)

    """ Returning the final predicted score by taking the mean and scaling it to a 10 point scheme by multiplying 2"""
    
    return df['score'].mean() * 2



"""  Mail Setup """

sender_email = 'rishi.gnit2025@gmail.com'
sender_password = os.getenv('EMAIL_PASSWORD')

def broadcast(keyword, ans):

    email_list = [
    'ritwiz736.hitcse2020@gmail.com',
    'rk04011@outlook.com',
    'dummy@gmail.com'
    ]

    for email in email_list:
        send_mail(email, keyword, ans)


def send_mail(receiver_email, keyword, data=None):

    date = datetime.now().strftime("Date: %d/%m/%Y Time: %H:%M:%S")
    subject = "Check your result for the sentiment analysis!"
    body = f"The sentiment score for {keyword.title()} is {data:.2f}. Review generated on {date}"

    em = EmailMessage()
    em['From'] = sender_email
    em['To'] = receiver_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context= context) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.sendmail(sender_email, receiver_email, em.as_string())


""" Endpoints """

@app.get("/")
async def root():
    return {"Server is running successfully!"}


@app.post("/get_score")
async def get_score(request : Request):
    try : 
        req_body = await request.json()
        keyword = req_body.get("keyword").lower()

        if not keyword:
                raise HTTPException(status_code=400, detail="Missing 'keyword' in the request body")
        url = f"https://en.wikipedia.org/wiki/{keyword}"
        
        ans = worker_function(url)
        broadcast(keyword, ans)

        return {'data': "{:.2f}".format(ans)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid Keyword Provided.")


