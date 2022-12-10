---
layout: page
title: Parsing a TEI Document for Gibbon Footnotes
description: This notebook parses a TEI XML file representing Gibbon's "Decline and Fall" for all footnote instances, then saves those entries in a CSV file.
---

This notebook parses a TEI XML file representing Gibbon's *Decline and Fall* for all footnote instances, then saves those entries in a `.csv` file.


```python
# imports
from bs4 import BeautifulSoup
import re
import pandas as pd
```


```python
# load xml file
with open('gibbon.xml', encoding='utf8', mode='r') as f:
    tei_string = f.read()
```


```python
# use BeatifulSoup to parse the TEI XML
tei_parsed = BeautifulSoup(tei_string)
```


```python
# find all foot notes
bottom_notes = tei_parsed.find_all('note', attrs={'place':'bottom'})
```


```python
# clean bottom notes and add to a list
clean_bottom_notes = []
for bottom_note in bottom_notes:
    bottom_note_text = re.sub(r'[\ \n]{2,}' , '',  bottom_note.text)
    clean_bottom_notes.append(bottom_note_text)
```


```python
# convert list to dataframe
df = pd.DataFrame(clean_bottom_notes)
```


```python
# check dataframe
df.head()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>\nPons Aureoli,thirteen miles from Bergamo, an...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>On the death of Gallienus, ſee Trebellius Poll...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Some ſuppoſed him, oddly enough, to be a baſta...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>\nNotoria,a periodical and official diſpatch w...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hiſt. Auguſt. p. 208. Gallienus deſcribes the ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# save datafram as csv
df.to_csv('foot_notes.csv')
```
