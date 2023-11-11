from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np 
import pandas as pd 


""" Intialization """

app = FastAPI()

origins = [
    "http://localhost:3000",
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
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class':regex})
    reviews = [result.text for result in results]

    if not reviews:
        raise HTTPException(status_code=500, detail=f"No reviews found on {url}")


    """ Converting the reviews to a dataframe for easier processing """

    df = pd.DataFrame(np.array(reviews), columns=['reviews'])
    print(df)

    """ Getting the review for each review in our dataframe """

    df['score'] = df['reviews'].apply(lambda x: map_sentiment_score(x))

    """ Returning the final predicted score by taking the mean and scaling it to a 10 point scheme by multiplying 2"""
    
    return df['score'].mean() * 2



""" Endpoints """

@app.get("/")
async def root():
    return {"Server is running successfully!"}


@app.get("/get_score")
async def get_score(request : Request):
    try : 
        req_body = await request.json()
        keyword = req_body.get("keyword").lower()

        if not keyword:
                raise HTTPException(status_code=400, detail="Missing 'keyword' in the request body")

        url = f"https://www.yelp.com/biz/{keyword}"
        
        ans = worker_function(url)

        return {'data': ans}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid Keyword Provided.")


