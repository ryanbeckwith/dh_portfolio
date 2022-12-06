---
layout: page
title: Named Entity Recognition Assignment
description: Extracting and georeferencing place names in Chapter 2 of Edward Gibbon's "Decline and Fall of the Roman Empire" using the Pleiades gazatteer and spaCy.
---

# NER Assignment
Georeferencing automatically detected place names in Edward Gibbon's *Decline and Fall of the Roman Empire* using the Pleiades gazatteer and `spaCy`.

Your assignment this week is to output a `CSV` file of place names, frequency and coordinates, as we did in class for a chapter of Gibbon of your choice. Try to find a chapter with a lot of places as we will be turning this data into an online map next week. The steps are:

* Choose a chapter from the text
* Begin a function by parsing input text
* Create a `spaCy` dictionary of entities and frequency
* Use the Pleiaes data from class to find coordinates for each place name
* Run your function on your chosen chapter
* Save your final `CSV`
* Come to class on Monday, ready to use it


```python
# import and download any relevant data
import pandas as pd
import spacy
import collections
nlp = spacy.load('en_core_web_sm') # good idea to initialize here
```


```python
gibbon_by_chapter = pd.read_csv('gibbon_text.csv').rename(columns={'Unnamed: 0':'chapter'})
gibbon_by_chapter
second_chapter = gibbon_by_chapter["StringText"][1]
# pass the second chapter into spaCy parser
second_chapter_doc = nlp(second_chapter)
```


```python
places = pd.read_csv('places.csv')
names = pd.read_csv('names.csv')
```


```python
# any helper functions you may need
def make_place_freq_df(doc):
    # putting all of the data into a dictionary
    place_freq = collections.defaultdict(int)
    for entity in doc.ents:
        if (entity.label_ == 'GPE') or (entity.label_ == 'LOC'):
            place_freq[entity.text] += 1 # the utility of defaultdict!
    place_freq = dict(place_freq)
    return pd.DataFrame.from_dict(place_freq, orient='index').reset_index().rename(columns={'index':'place_name',0:'frequency'})

def get_pleiades_id(term):
    """
    Iterates through all of the possible names in the names.csv file
    Returns None if no matched names
    """
    name_row = names.loc[names['attested_form'] == term]
    if len(name_row) == 1:
        return int(name_row.place_id.iloc[0])
    else:
        name_row = names.loc[names['romanized_form_1'] == term]
        if len(name_row) == 1:
            return int(name_row.place_id.iloc[0])
        else:
            name_row = names.loc[names['romanized_form_2'] == term]
            if len(name_row) == 1:
                return int(name_row.place_id.iloc[0])
            else:
                name_row = names.loc[names['romanized_form_3'] == term]
                if len(name_row) == 1:
                    return int(name_row.place_id.iloc[0])
                else:
                    return None
                
# could've just one function here, but not too much trouble to do two
def get_lat(pl_id):
    places_row = places.loc[places['id'] == pl_id]
    if len(places_row) == 1:
        return places_row.representative_latitude.iloc[0]
    
def get_long(pl_id):
    places_row = places.loc[places['id'] == pl_id]
    if len(places_row) == 1:
        return places_row.representative_longitude.iloc[0]
```


```python
# final functions
def make_ner_df(doc):
    df = make_place_freq_df(doc)
    df['pleiades_id'] = df['place_name'].apply(get_pleiades_id)
    df = df.dropna().reset_index(drop=True)
    df['lat'] = df['pleiades_id'].apply(get_lat)
    df['long'] = df['pleiades_id'].apply(get_long)
    return df
```


```python
# try your function out
second_chapter_ner_df = make_ner_df(second_chapter_doc)
second_chapter_ner_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>place_name</th>
      <th>frequency</th>
      <th>pleiades_id</th>
      <th>lat</th>
      <th>long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rome</td>
      <td>32</td>
      <td>423025.0</td>
      <td>41.891775</td>
      <td>12.486137</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nile</td>
      <td>4</td>
      <td>727172.0</td>
      <td>19.211409</td>
      <td>30.567330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Athens</td>
      <td>11</td>
      <td>579885.0</td>
      <td>37.972634</td>
      <td>23.722746</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sparta</td>
      <td>1</td>
      <td>570685.0</td>
      <td>37.077905</td>
      <td>22.427298</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Padua</td>
      <td>2</td>
      <td>393473.0</td>
      <td>45.409561</td>
      <td>11.876975</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Arpinum</td>
      <td>1</td>
      <td>432700.0</td>
      <td>41.648422</td>
      <td>13.609876</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Etruria</td>
      <td>1</td>
      <td>413122.0</td>
      <td>42.758127</td>
      <td>11.546721</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Latium</td>
      <td>2</td>
      <td>432900.0</td>
      <td>41.590543</td>
      <td>13.192265</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Alesia</td>
      <td>1</td>
      <td>177434.0</td>
      <td>47.536622</td>
      <td>4.503884</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Africa</td>
      <td>4</td>
      <td>775.0</td>
      <td>32.500000</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Euphrates</td>
      <td>1</td>
      <td>912849.0</td>
      <td>35.543310</td>
      <td>39.606018</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Capua</td>
      <td>1</td>
      <td>432754.0</td>
      <td>41.086092</td>
      <td>14.250207</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Verona</td>
      <td>2</td>
      <td>383816.0</td>
      <td>45.442130</td>
      <td>10.995736</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Troas</td>
      <td>1</td>
      <td>550944.0</td>
      <td>39.837052</td>
      <td>26.336944</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Milan</td>
      <td>1</td>
      <td>383706.0</td>
      <td>45.463746</td>
      <td>9.188060</td>
    </tr>
    <tr>
      <th>15</th>
      <td>London</td>
      <td>1</td>
      <td>79574.0</td>
      <td>51.513335</td>
      <td>-0.088949</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Paris</td>
      <td>1</td>
      <td>109126.0</td>
      <td>48.854405</td>
      <td>2.346168</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Arles</td>
      <td>1</td>
      <td>148217.0</td>
      <td>43.677616</td>
      <td>4.630799</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Bordeaux</td>
      <td>1</td>
      <td>138248.0</td>
      <td>44.837205</td>
      <td>-0.576533</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Autun</td>
      <td>1</td>
      <td>177460.0</td>
      <td>46.947240</td>
      <td>4.299177</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Vienne</td>
      <td>1</td>
      <td>167719.0</td>
      <td>45.524510</td>
      <td>4.875982</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Hercules</td>
      <td>1</td>
      <td>266032.0</td>
      <td>37.500000</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Sicily</td>
      <td>1</td>
      <td>462492.0</td>
      <td>37.643297</td>
      <td>13.932018</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Media</td>
      <td>1</td>
      <td>903080.0</td>
      <td>34.500000</td>
      <td>46.500000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Arabia</td>
      <td>2</td>
      <td>981506.0</td>
      <td>27.500000</td>
      <td>32.500000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>India</td>
      <td>2</td>
      <td>50004.0</td>
      <td>22.500000</td>
      <td>77.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# save result as CSV
second_chapter_ner_df.to_csv('ch2gibbon_places.csv')
```
