---
layout: page
title: Chronicling America API Assignment
description: Searching the Chronicling America API for the term "free market"
---

# Chronicling America API Assignment
In this assignment, you are tasked with:
* searching Chronicling America's API for a key word of your choice
* parsing your results from a dictionary to a `DataFrame` with headings "title", "city", 'date", and "raw_text"
* processing the raw text by removing "\n" characters, stopwords, and then lemmatizing the raw text before adding it to a new column called "lemmas."
* saving your `DataFrame` to a csv file

If you need any help with this assignment please email micah.saxton@tufts.edu



```python
# imports
import requests
import json
import math
import pandas as pd
import spacy
```


```python
# Search parameters
start_date = '1900'
end_date = '2022'
search_term = 'free market'
```


```python
# initial search
url = (f'https://chroniclingamerica.loc.gov/search/pages/results/?state=&date1={start_date}'
       f'&date2={end_date}&proxtext={search_term}&x=16&y=8&dateFilterType=yearRange&rows=20'
       f'&searchType=basic&format=json') 
response = requests.get(url)  # returns a 'object' representing the webpage
raw = response.text  # the text attribute contains the text from the web page as a string
results = json.loads(raw)  # the loads method from the json library transforms the string into a dict
```


```python
results.keys()
```




    dict_keys(['totalItems', 'endIndex', 'startIndex', 'itemsPerPage', 'items'])




```python
# find total amount of pages
total_pages = math.ceil(results['totalItems'] / results['itemsPerPage'])
print(total_pages)
```

    3113



```python
# query the api and save to dict 
data = []
for i in range(1, 11):  # for sake of time I'm doing only 10, you will want to put total_pages+1
    url = (f'https://chroniclingamerica.loc.gov/search/pages/results/?state=&date1={start_date}'
           f'&date2={end_date}&proxtext={search_term}&x=16&y=8&dateFilterType=yearRange&rows=20'
           f'&searchType=basic&format=json&page={i}')  # f-string
    response = requests.get(url)
    raw = response.text
    print(response.status_code)  # checking for errors
    results = json.loads(raw)
    items_ = results['items']
    for item_ in items_:
        temp_dict = {}
        temp_dict['title'] = item_['title_normal']
        temp_dict['city'] = item_['city']
        temp_dict['date'] = item_['date']
        temp_dict['raw_text'] = item_['ocr_eng']
        data.append(temp_dict)
```

    200
    200



```python
# convert dict to dataframe
df = pd.DataFrame.from_dict(data)
```


```python
# convert date column from string to date-time object
df['date'] = pd.to_datetime(df['date'])
```


```python
# sort by date
sorted_df = df.sort_values(by='date')
```


```python
# write fuction to process text
# load nlp model
nlp = spacy.load("en_core_web_sm")
nlp.disable_pipes('ner', 'parser')  # these are unnecessary for the task at hand

junk_words = ['hesse']  # you can add any words you want removed here

def process_text(text):
    """Remove new line characters and lemmatize text. Returns string of lemmas"""
    text = text.replace('\n', ' ')
    doc = nlp(text)
    tokens = [token for token in doc]
    no_stops = [token for token in tokens if not token.is_stop]
    no_punct = [token for token in no_stops if token.is_alpha]
    lemmas = [token.lemma_ for token in no_punct]
    lemmas_lower = [lemma.lower() for lemma in lemmas]
    lemmas_string = ' '.join(lemmas_lower)
    return lemmas_string
    
```


```python
# apply process_text function to raw_text column
sorted_df['lemmas'] = sorted_df['raw_text'].apply(process_text)
```


```python
# save to csv
sorted_df.to_csv(f'../data/{search_term}{start_date}-{end_date}.csv', index=False)
```


```python

```
