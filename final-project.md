---
layout: page
title: Final Project
permalink: /final-project/
---

## End Rhyme Analysis in Contemporary Poetry

### Introduction

Poetry is a unique medium of written work that often relies heavily on formal elements, such as rhyme and meter. Interestingly, however, the analysis of end rhymes in poetry is not a trivial task: how, for example, might one distinguish whether two words actually rhyme? While this may seem like a simple question for a human to answer, a machine is not so easily positioned to identify the sonic inconsistencies in English pronounciation. How would you ensure that a computer recognizes rhyming words that end with different letters, such as "stay" and "ballet"?

In this project, I aim to answer two pertinent questions regarding end rhymes in poems:

1) Which (thematic) types of poems contain the highest frequency of end rhymes?
2) How commonly do common end rhyme schemes like "abba", "abab", "aabb", and "abcb" occur in contemporary poetry?

The answers to these two questions will reveal interesting takeaways regarding poetry in general. For example, there is a common stereotype that "love" poems utilize rhyme schemes more frequently than other types of poems. Furthermore, the frequency of common end-rhyme schemes like will help reveal whether rhyming occurs less commonly in contemporary poetry than in the past, where formal rhyme and meter were almost required elements of a successful poem (see [Corn, chapter 1](https://www.goodreads.com/en/book/show/881744.The_Poem_s_Heartbeat)).

### Data Gathering

The data for this project was sourced from a [Kaggle dataset](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems) containing the complete collection of poems from the Poetry
Foundation (as of 2019). The Poetry Foundation is a 501(c)(3) organization that provides a monthly poetry magazine, live poetry events, and a robust virtual collection of poems from numerous authors, as per [their website](https://www.poetryfoundation.org/foundation/about). Other datasets were considered, but this one was chosen for its inclusion of semantic tags for each poem, which would allow for efficient characterization of the thematic types of each poem. This was necessary for classifying which kinds of poems have the highest freqency of end rhymes, a key goal of this project.

Poems in this collection include originals written by modern authors, as well as classic poems published on behalf of famous authors. However, the overwhelming distribution of poems include contemporary submissions from relatively recent authors, thus allowing for this analysis to be somewhat representative of contemporary poetry as a whole.

### Dataset Preparation

Unfortunately, this dataset was highly irregular in many regards. In its raw form, there were several issues that needed to be addressed, including:
1) The presence of extraneous carriage returns (`"\r"`) in the poems themselves
2) Accented versions of standard alphabetical characters
3) Lines ending with pure numbers (which must be converted to "spelled out" string versions of themselves for rhyme analysis to occur)
4) Lines ending in non-alphanumeric characters
5) Lines ending in words that are not real words
6) Empty lines (lines containing only a newline character)
7) Poems that utilized spaces/tabs to distinguish line breaks instead of newline characters

Each of these issues, with the exception of the final one, were appropriately addressed using Python packages, regular expressions, and other data preparation/cleaning techniques.

### Methods

Following the data preparation process, two separate data analysis efforts were made to answer the two separate questions posed in the introduction.

To answer question 1, the following approach was taken:
1) Obtain the last word of each line in every poem in the dataset
2) Query the [DataMuse API](https://www.datamuse.com/api/) for rhymes for every unique last word from the dataset.
    - The API returns a list of every word that rhymes with the target word, as well as a scalar score that quantifies how "well" each particular word rhymes with the target word.
    - Store each API call in a `.json` file (in case of crashes, as it took nearly 1.5 hours to fully query the API).
3) Calculate a "rhyme score" scalar metric as follows:
    - For each list of end words, check every possible pair of end words for rhymes.
    - If there is a rhyme (according to DataMuse), then weigh the rhyme score from DataMuse by how close the two rhymes were in the poem (closer is better).
    - Sum the weighed rhyme scores of every pair of end words to obtain the final rhyme score.
4) Loop over all poems of a particular tag, and average the rhyme scores for each tag.
5) Plot the results in a bar chart (for the top 20 and bottom 20 tags), only considering tags that had at least 50 poems.

To answer question 2, the following approach was taken:
1) For every poem, assign a number to each end word of the poem, starting from the beginning, where the same number is reused for rhyming words.
2) For a string of the form "abab", check over all sliding windows of length 4 in the list of end words, and see if the number pattern matches the string. In this case, "abab" would match to something like 0, 1, 0, 1. As such, a poem is determined to be of the rhyme scheme "abab" if it has any consecutive 4-line sequence that is of the form "abab".
3) Compute whether every poem is of the form "abba", "abab", "aabb", or "abcb".
4) Plot the relative frequency results in a bar chart.

### Code Issues

Unfortunately, while the rhyme detection from DataMuse is certainly robust, and is able to account successfully for edge cases like "stay" and "ballet", other aspects of the code rely heavily on programmatical heuristics which are not necessarily the most accurate. For example, the rhyme score metric over-prioritizes poems of the form "aaaa", as there are more pairs of end rhymes in a poem where every line rhymes with every other rhyme. An alternative approach could have been to establish a target threshold for the rhyme score, and simply classify each poem as either "rhyming" or "not rhyming" based on where it falls with respect to the threshold rhyme score.

Other minor issues come from the fact that words which were not recognized as real words could not be handled properly by the DataMuse API. As such, even though a made-up word might indeed rhyme with another word, there was no way for the computer to recognize that. Furthermore, another related issue was the idea that the "sliding window" approach to rhyme scheme detection overlooks the natural stanza breaks present in many poems. However, since the dataset did not preserve these stanza breaks consistently for all poems, there was no choice but to fully ignore stanzas and focus instead on all possible consecutive line sequences.

### Visualization Explanations

The plots intended to answer Question 1 are shown below:

#### Plot 1: Top 20 Thematic Tags with the Highest Average Rhyme Scores

![png](./top20.png)

This plot shows the top 20 thematic tags which had the highest average rhyme scores, in decreasing order from top to bottom. Furthermore, each bar is colored according to how many poems were contained in the analysis of that tag type, with red being on the low side (~50) and blue being on the high side (~2000).

Immediately, the first observation I made about this plot was the staggeringly high rhyme scores of thematic tags related to love and romantic relationships. In fact, I would categorize 11 of the top 20 tags to be directly related to love in some capacity. These results provide convincing evidence that love poems do indeed contain higher frequencies of end rhymes, thus confirming (at least for this dataset) that love poems are the most susceptible to including rhymes, even in contemporary poetry.

One thing to keep in mind, however, is that many of the tags with very high rhyme scores did seem to have relatively low numbers of poems, which may potentially point to them being statistical outliers that are skewing the distribution substantially in favor of love poems. However, without further statistical analysis, it is difficult to say conclusively if this is the case.

#### Plot 2: Bottom 20 Thematic Tags with the Lowest Average Rhyme Scores

![png](./bottom20.png)

This plot is colored identically to Plot 1, and simply represents the bottom portion of tags with respect to average rhyme score. We can see that these themes are certainly not as aligned with "love" as in Plot 1: in fact, I would argue that none of these themes are directly related to romantic love, especially not compared to the tags in Plot 1. As such, this is further reinforcement that love poems are especially susceptible to containing end rhymes.

Furthermore, we can also informally analyze the kinds of themes we see present in this plot to perhaps characterize the types of poems that are least likely to contain end rhymes. In my opinion, many of these themes are related to concrete scientific phenomena (planets, stars, weather, sciences, the body, health and illness), perhaps indicating that the "quaint" and "carefree" tones that often arises as a result of utilizing end rhymes are not as easily integrated into poems exploring the sciences, which are often thought of as "formal" and "rigid".

Lastly, we must note that while these plots do indeed give a glimpse into the distribution of end rhymes across thematic tags, there are a few pitfalls that we should be careful to avoid. Namely, this analysis does not tell us precisely *why* love poems tend to have higher frequencies of end rhymes, it just tells us that they *do* tend to have higher frequencies. Furthermore, we cannot be certain that these user-submitted tags perfectly align to the true themes of the poems that they tag, so this data could potentially be made inaccurate by poorly labelled poems.

Next, the plot intended to answer Question 2 is shown below:

#### Plot 3: Frequencies of Common Rhyme Schemes

![png](./freqs4.png)

This visualization plots the frequencies of poems containing particular rhyme schemes that are often found in traditional poems: "abba", "abab", "aabb", and "abcb". In this context, frequency refers to the decimal percentage of poems that contained one or more sequences of 4 lines that matched one of these rhyme schemes.

Interestingly, we can see that the "abcb" rhyme scheme is significantly more common than the other rhyme schemes, which is likely just a result of the fact that "abcb" stanzas are typically easier to write than stanzas with two pairs of end rhymes. However, we can see that the "abba" scheme is much less popular than the other two types that require two pairs of end rhymes. This could potentially be evidence that the sonic preferences of readers is more aligned with the "abab" and "aabb" rhyme schemes than the "abba" rhyme scheme, but without further evidence we cannot imply causality.

Lastly, another key takeaway can be made by simply observing the magnitudes of the frequencies in this plot. Even the most popular rhyme scheme of "abcb" barely scratched above 2.5% of all poems in the dataset, which is nowhere near a majority. As such, we may reason that formal rhyme schemes are substantially less frequent in contemporary poetry as compared to past eras in which formal rhyme and meter were effectively required for a poem to be considered a poem.

### Conclusion

In conclusion, we can see that common ideas about rhyme frequencies are upheld by this analysis. Indeed, love poems appear to be the category of poems that are most likely to contain end rhymes, a result that is largely unsurprising. Furthermore, we can see that the use of formal rhyme schemes has certainly declined in contemporary poetry, with the most popular rhyme scheme only comprising just over 2.5% of all poems in this dataset.

### Citations

1. Corn, A. (1997) *The Poem's Heartbeat: A Manual of Prosody*. 
2. Bramhecha, D. (2019). *Poetry Foundation Poems.* Kaggle Datasets. Retrieved November
4, 2022, from https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
3. Datamuse. (2022). Datamuse API. Retrieved November 4, 2022, from
https://www.datamuse.com/api/
4. Jack, U. (2017). *Poetry Analysis using AI & Machine learning.* Kaggle Code. Retrieved
November 4, 2022, from https://www.kaggle.com/code/ultrajack/poetry-analysis-usingai-machine-learning
5. Poetry Foundation. (2022). *About Us.* Poetry Foundation. Retrieved November 4, 2022,
from https://www.poetryfoundation.org/foundation/about

### Code: Data Cleaning


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import requests
from requests.exceptions import HTTPError
import string
import re
from num2words import num2words
from unidecode import unidecode
import os.path
import json
from tqdm import tqdm
sns.set(rc={'figure.figsize':(12,8)})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style("white")
```


```python
df = pd.read_csv("./PoetryFoundationData.csv")
df
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
      <th>Unnamed: 0</th>
      <th>Title</th>
      <th>Poem</th>
      <th>Poet</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>\r\r\n                    Objects Used to Prop...</td>
      <td>\r\r\nDog bone, stapler,\r\r\ncribbage board, ...</td>
      <td>Michelle Menting</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>\r\r\n                    The New Church\r\r\n...</td>
      <td>\r\r\nThe old cupola glinted above the clouds,...</td>
      <td>Lucia Cherciu</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>\r\r\n                    Look for Me\r\r\n   ...</td>
      <td>\r\r\nLook for me under the hood\r\r\nof that ...</td>
      <td>Ted Kooser</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>\r\r\n                    Wild Life\r\r\n     ...</td>
      <td>\r\r\nBehind the silo, the Mother Rabbit\r\r\n...</td>
      <td>Grace Cavalieri</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>\r\r\n                    Umbrella\r\r\n      ...</td>
      <td>\r\r\nWhen I push your button\r\r\nyou fly off...</td>
      <td>Connie Wanek</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13849</th>
      <td>13</td>
      <td>\r\r\n                    1-800-FEAR\r\r\n    ...</td>
      <td>\r\r\nWe'd  like  to  talk  with  you  about  ...</td>
      <td>Jody Gladding</td>
      <td>Living,Social Commentaries,Popular Culture</td>
    </tr>
    <tr>
      <th>13850</th>
      <td>14</td>
      <td>\r\r\n                    The Death of Atahual...</td>
      <td>\r\r\n\r\r\n</td>
      <td>William Jay Smith</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13851</th>
      <td>15</td>
      <td>\r\r\n                    Poet's Wish\r\r\n   ...</td>
      <td>\r\r\n\r\r\n</td>
      <td>William Jay Smith</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13852</th>
      <td>0</td>
      <td>\r\r\n                    0\r\r\n</td>
      <td>\r\r\n          Philosophic\r\r\nin its comple...</td>
      <td>Hailey Leithauser</td>
      <td>Arts &amp; Sciences,Philosophy</td>
    </tr>
    <tr>
      <th>13853</th>
      <td>1</td>
      <td>\r\r\n                    !\r\r\n</td>
      <td>\r\r\nDear Writers, I’m compiling the first in...</td>
      <td>Wendy Videlock</td>
      <td>Relationships,Gay, Lesbian, Queer,Arts &amp; Scien...</td>
    </tr>
  </tbody>
</table>
<p>13854 rows × 5 columns</p>
</div>




```python
df["Poem"].iloc[0]
```




    "\r\r\nDog bone, stapler,\r\r\ncribbage board, garlic press\r\r\n     because this window is loose—lacks\r\r\nsuction, lacks grip.\r\r\nBungee cord, bootstrap,\r\r\ndog leash, leather belt\r\r\n     because this window had sash cords.\r\r\nThey frayed. They broke.\r\r\nFeather duster, thatch of straw, empty\r\r\nbottle of Elmer's glue\r\r\n     because this window is loud—its hinges clack\r\r\nopen, clack shut.\r\r\nStuffed bear, baby blanket,\r\r\nsingle crib newel\r\r\n     because this window is split. It's dividing\r\r\nin two.\r\r\nVelvet moss, sagebrush,\r\r\nwillow branch, robin's wing\r\r\n     because this window, it's pane-less. It's only\r\r\na frame of air.\r\r\n"




```python
df["Title"].iloc[0]
```




    '\r\r\n                    Objects Used to Prop Open a Window\r\r\n                '




```python
df.drop("Unnamed: 0", axis=1, inplace=True)
df["id"] = df.index
```


```python
df["Title"] = df["Title"].str.strip()
df["Poem"] = df["Poem"].str.replace(r'[\t\r]', "")
df["Poem"] = df["Poem"].str.strip()
df
```

    /tmp/ipykernel_1816/3718095344.py:2: FutureWarning: The default value of regex will change from True to False in a future version.
      df["Poem"] = df["Poem"].str.replace(r'[\t\r]', "")





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
      <th>Title</th>
      <th>Poem</th>
      <th>Poet</th>
      <th>Tags</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Objects Used to Prop Open a Window</td>
      <td>Dog bone, stapler,\ncribbage board, garlic pre...</td>
      <td>Michelle Menting</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Church</td>
      <td>The old cupola glinted above the clouds, shone...</td>
      <td>Lucia Cherciu</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Look for Me</td>
      <td>Look for me under the hood\nof that old Chevro...</td>
      <td>Ted Kooser</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wild Life</td>
      <td>Behind the silo, the Mother Rabbit\nhunches li...</td>
      <td>Grace Cavalieri</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Umbrella</td>
      <td>When I push your button\nyou fly off the handl...</td>
      <td>Connie Wanek</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13849</th>
      <td>1-800-FEAR</td>
      <td>We'd  like  to  talk  with  you  about  fear t...</td>
      <td>Jody Gladding</td>
      <td>Living,Social Commentaries,Popular Culture</td>
      <td>13849</td>
    </tr>
    <tr>
      <th>13850</th>
      <td>The Death of Atahuallpa</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13850</td>
    </tr>
    <tr>
      <th>13851</th>
      <td>Poet's Wish</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13851</td>
    </tr>
    <tr>
      <th>13852</th>
      <td>0</td>
      <td>Philosophic\nin its complex, ovoid emptiness,\...</td>
      <td>Hailey Leithauser</td>
      <td>Arts &amp; Sciences,Philosophy</td>
      <td>13852</td>
    </tr>
    <tr>
      <th>13853</th>
      <td>!</td>
      <td>Dear Writers, I’m compiling the first in what ...</td>
      <td>Wendy Videlock</td>
      <td>Relationships,Gay, Lesbian, Queer,Arts &amp; Scien...</td>
      <td>13853</td>
    </tr>
  </tbody>
</table>
<p>13854 rows × 5 columns</p>
</div>




```python
df["Poem"].iloc[0]
```




    "Dog bone, stapler,\ncribbage board, garlic press\n     because this window is loose—lacks\nsuction, lacks grip.\nBungee cord, bootstrap,\ndog leash, leather belt\n     because this window had sash cords.\nThey frayed. They broke.\nFeather duster, thatch of straw, empty\nbottle of Elmer's glue\n     because this window is loud—its hinges clack\nopen, clack shut.\nStuffed bear, baby blanket,\nsingle crib newel\n     because this window is split. It's dividing\nin two.\nVelvet moss, sagebrush,\nwillow branch, robin's wing\n     because this window, it's pane-less. It's only\na frame of air."




```python
# Columns in main dataset: Poem ID, Title, Poem, Poet, Tags
# Idea: hash poem IDs in tag buckets
tags = defaultdict(set)

def parse_tags(tagstr):
    return tagstr.split(",")

def build_tags(row):
    tagstr = row["Tags"]
    if isinstance(tagstr, str):
        taglist = parse_tags(tagstr)
        for tag in taglist:
            tags[tag].add(row["id"])
             
for _, row in df.iterrows():
    build_tags(row)
```


```python
len(tags)
```




    129




```python
def get_last_word(line):
    # Substitutes non-standard (accented) characters
    line = unidecode(line)
    # Removes non-alphanumerics from the end of the string
    line = re.sub(r"^[\W_]+|[\W_]+$", "", line)
    last_word = line.split(" ")[-1]
    only_alphanumeric = re.sub(r"[^a-zA-Z0-9]", " ", last_word)
    last_rhyme = only_alphanumeric.split()[-1]
    if last_rhyme.isnumeric():
        numword = num2words(last_rhyme)
        only_alpha = re.sub(r"[^a-zA-Z]", " ", numword)
        return only_alpha.split(" ")[-1]
    return last_rhyme.lower()
```


```python
print(get_last_word(" &Hello, my name is Ryan. . ."))
print(get_last_word("Hello, my name is Ryan."))
print(get_last_word("Hello, my name is 32.\n"))
print(get_last_word("Hello, my name is-32.\n"))
print(get_last_word("Hello, my name is------12342343.\n"))
print(get_last_word("Hello, my name is------12342343. í\n"))
```

    ryan
    ryan
    two
    two
    three
    i



```python
df["Poem"].iloc[0]
```




    "Dog bone, stapler,\ncribbage board, garlic press\n     because this window is loose—lacks\nsuction, lacks grip.\nBungee cord, bootstrap,\ndog leash, leather belt\n     because this window had sash cords.\nThey frayed. They broke.\nFeather duster, thatch of straw, empty\nbottle of Elmer's glue\n     because this window is loud—its hinges clack\nopen, clack shut.\nStuffed bear, baby blanket,\nsingle crib newel\n     because this window is split. It's dividing\nin two.\nVelvet moss, sagebrush,\nwillow branch, robin's wing\n     because this window, it's pane-less. It's only\na frame of air."




```python
last_words = defaultdict(list)

def build_last_words(row):
    poem = row["Poem"]
    id = row["id"]
    lines = poem.splitlines()
    for line in filter(lambda line: any(char.isalpha() or char.isdigit() for char in line), lines):
        last_word = get_last_word(line)
        last_words[id].append(last_word)

for _, row in df.iterrows():
    build_last_words(row)
```

### Code: End Rhyme Frequency by Tags (Question 1)


```python
print(sum(len(l) for l in last_words.values()))
```

    327750



```python
unique_last_words = { word for l in last_words.values() for word in l }
print(len(unique_last_words))
```

    36731



```python
# Using the Datamuse API
def get_rhymes(word):
    try:
        response = requests.get(f"https://api.datamuse.com/words?rel_rhy={word}")
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
        return response.json()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
        return []
    except Exception as err:
        print(f'Other error occurred: {err}')  # Python 3.6
        return []
```


```python
rhymes = defaultdict(list)

if not os.path.isfile("rhymes.json"):
    for word in tqdm(unique_last_words):
        rhyme_list = get_rhymes(word)
        if len(rhyme_list) != 0:
            rhymes[word] = rhyme_list
    with open("rhymes.json", "x") as outfile:
        json.dump(rhymes, outfile, indent=4) 
else:
    with open("rhymes.json") as f:
        rhymes = json.load(f)
```


```python
print(len(rhymes))
```

    27771



```python
df["rhyme_score"] = 0.0
df
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
      <th>Title</th>
      <th>Poem</th>
      <th>Poet</th>
      <th>Tags</th>
      <th>id</th>
      <th>rhyme_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Objects Used to Prop Open a Window</td>
      <td>Dog bone, stapler,\ncribbage board, garlic pre...</td>
      <td>Michelle Menting</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Church</td>
      <td>The old cupola glinted above the clouds, shone...</td>
      <td>Lucia Cherciu</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Look for Me</td>
      <td>Look for me under the hood\nof that old Chevro...</td>
      <td>Ted Kooser</td>
      <td>NaN</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wild Life</td>
      <td>Behind the silo, the Mother Rabbit\nhunches li...</td>
      <td>Grace Cavalieri</td>
      <td>NaN</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Umbrella</td>
      <td>When I push your button\nyou fly off the handl...</td>
      <td>Connie Wanek</td>
      <td>NaN</td>
      <td>4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13849</th>
      <td>1-800-FEAR</td>
      <td>We'd  like  to  talk  with  you  about  fear t...</td>
      <td>Jody Gladding</td>
      <td>Living,Social Commentaries,Popular Culture</td>
      <td>13849</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13850</th>
      <td>The Death of Atahuallpa</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13850</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13851</th>
      <td>Poet's Wish</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13851</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13852</th>
      <td>0</td>
      <td>Philosophic\nin its complex, ovoid emptiness,\...</td>
      <td>Hailey Leithauser</td>
      <td>Arts &amp; Sciences,Philosophy</td>
      <td>13852</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13853</th>
      <td>!</td>
      <td>Dear Writers, I’m compiling the first in what ...</td>
      <td>Wendy Videlock</td>
      <td>Relationships,Gay, Lesbian, Queer,Arts &amp; Scien...</td>
      <td>13853</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>13854 rows × 6 columns</p>
</div>




```python
def index_of_first(lst, pred):
    for i,v in enumerate(lst):
        if pred(v):
            return i
    return -1
```


```python
def compute_rhyme_score(last_word_list):
    DEFAULT_RAW_SCORE = 100
    rhyme_score = 0.0
    list_len = len(last_word_list)
    for i in range(list_len):
        last_word_i = last_word_list[i]
        if last_word_i not in rhymes:
            continue
        for j in range(i + 1, list_len):
            last_word_j = last_word_list[j]
            distance_factor = (10 * (list_len - (j - i)) / list_len) ** 2
            if last_word_i == last_word_j:
                rhyme_score += distance_factor * DEFAULT_RAW_SCORE
                continue
            rhyme_idx = index_of_first(rhymes[last_word_i],
                                       lambda rhyme_dict:
                                           rhyme_dict["word"] == last_word_j)
            if rhyme_idx != -1:
                rhyme_dict = rhymes[last_word_i][rhyme_idx]
                raw_score = rhyme_dict["score"] if "score" in rhyme_dict \
                                                else DEFAULT_RAW_SCORE
                rhyme_score += distance_factor * raw_score
    total_possible_rhymes = (list_len * (list_len - 1)) / 2
    return rhyme_score / total_possible_rhymes
```


```python
print(compute_rhyme_score(["augmented", "unprecedented", "drew", "to"]))
print(compute_rhyme_score(["augmented", "augmented"]))
print(compute_rhyme_score(["augmented", "augmented", "unprecedented"]))
print(last_words[0])
print(compute_rhyme_score(last_words[0]))
print(last_words[3780])
print(compute_rhyme_score(last_words[3780]))
```

    185606.25
    2500.0
    30833.33333333334
    ['stapler', 'press', 'lacks', 'grip', 'bootstrap', 'belt', 'cords', 'broke', 'empty', 'glue', 'clack', 'shut', 'blanket', 'newel', 'dividing', 'two', 'sagebrush', 'wing', 'only', 'air']
    595.478947368421
    ['muteness', 'branches', 'born', 'is', 'tongue', 'grunts', 'tongue', 'it', 'speech', 'internment', 'yard', 'zipper', 'hemisphere', 'him', 'tongue', 'swallows', 'freely', 'arm', 'tarry', 'high', 'within', 'cloud', 'home', 'one']
    58.8768115942029



```python
for id, last_word_list in tqdm(filter(lambda t: len(t[1]) > 1,
                                      last_words.items())):
    rhyme_score = compute_rhyme_score(last_word_list)
    index = np.flatnonzero(df["id"] == id)[0]
    df.at[index, "rhyme_score"] = rhyme_score

df
```

    9524it [03:18, 47.93it/s] 





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
      <th>Title</th>
      <th>Poem</th>
      <th>Poet</th>
      <th>Tags</th>
      <th>id</th>
      <th>rhyme_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Objects Used to Prop Open a Window</td>
      <td>Dog bone, stapler,\ncribbage board, garlic pre...</td>
      <td>Michelle Menting</td>
      <td>NaN</td>
      <td>0</td>
      <td>595.478947</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Church</td>
      <td>The old cupola glinted above the clouds, shone...</td>
      <td>Lucia Cherciu</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Look for Me</td>
      <td>Look for me under the hood\nof that old Chevro...</td>
      <td>Ted Kooser</td>
      <td>NaN</td>
      <td>2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wild Life</td>
      <td>Behind the silo, the Mother Rabbit\nhunches li...</td>
      <td>Grace Cavalieri</td>
      <td>NaN</td>
      <td>3</td>
      <td>2696.474896</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Umbrella</td>
      <td>When I push your button\nyou fly off the handl...</td>
      <td>Connie Wanek</td>
      <td>NaN</td>
      <td>4</td>
      <td>569.898634</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13849</th>
      <td>1-800-FEAR</td>
      <td>We'd  like  to  talk  with  you  about  fear t...</td>
      <td>Jody Gladding</td>
      <td>Living,Social Commentaries,Popular Culture</td>
      <td>13849</td>
      <td>4645.890308</td>
    </tr>
    <tr>
      <th>13850</th>
      <td>The Death of Atahuallpa</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13850</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>13851</th>
      <td>Poet's Wish</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13851</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13852</th>
      <td>0</td>
      <td>Philosophic\nin its complex, ovoid emptiness,\...</td>
      <td>Hailey Leithauser</td>
      <td>Arts &amp; Sciences,Philosophy</td>
      <td>13852</td>
      <td>3388.751715</td>
    </tr>
    <tr>
      <th>13853</th>
      <td>!</td>
      <td>Dear Writers, I’m compiling the first in what ...</td>
      <td>Wendy Videlock</td>
      <td>Relationships,Gay, Lesbian, Queer,Arts &amp; Scien...</td>
      <td>13853</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>13854 rows × 6 columns</p>
</div>




```python
# Look at some of the poems with the highest rhyme scores
topn = df.nlargest(3, 'rhyme_score')
for i, row in topn.iterrows():
    print(f"{i}: \"{row['Title']}\" by {row['Poet']}")
    print(f"Lines: {len(last_words[df.at[i, 'id']])}")
    print(f"Rhyme score: {df.at[i, 'rhyme_score']}")
    print(row.loc['Poem'])
```

    12889: "Villanelle of Change" by Edwin Arlington Robinson
    Lines: 7
    Rhyme score: 211080.46647230317
    Since Persia fell at Marathon,
       The yellow years have gathered fast:Long centuries have come and gone.And yet (they say) the place will don
       A phantom fury of the past,Since Persia fell at Marathon;And as of old, when Helicon
       Trembled and swayed with rapture vast(Long centuries have come and gone),This ancient plain, when night comes on,
       Shakes to a ghostly battle-blast,Since Persia fell at Marathon.But into soundless Acheron
       The glory of Greek shame was cast:Long centuries have come and gone,The suns of Hellas have all shone,
       The first has fallen to the last:—Since Persia fell at Marathon,Long centuries have come and gone.
    9204: "A Moment" by Mary Elizabeth Coleridge
    Lines: 5
    Rhyme score: 206092.8
    The clouds had made a crimson crown
      Above the mountains high.The stormy sun was going down
      In a stormy sky.Why did you let your eyes so rest on me,
      And hold your breath between?In all the ages this can never be
      As if it had not been.
    12882: "The House on the Hill" by Edwin Arlington Robinson
    Lines: 7
    Rhyme score: 160214.86880466467
    They are all gone away,
       The House is shut and still,There is nothing more to say.Through broken walls and gray
       The winds blow bleak and shrill:They are all gone away.Nor is there one to-day
       To speak them good or ill:There is nothing more to say.Why is it then we stray
       Around the sunken sill?They are all gone away,And our poor fancy-play
       For them is wasted skill:There is nothing more to say.There is ruin and decay
       In the House on the Hill:They are all gone away,There is nothing more to say.



```python
# obtain average rhyme score for each tag
avg_score = defaultdict(dict)

for tag, idset in tags.items():
    curr_score_sum = 0.0
    for id in idset:
        index = np.flatnonzero(df["id"] == id)[0]
        curr_score_sum += df.at[index, "rhyme_score"]
    avg_score[tag]["rhyme_score"] = curr_score_sum / len(idset)
    avg_score[tag]["n_poems"] = len(idset)
```


```python
avg_score_df = pd.DataFrame.from_records([(tag, d["rhyme_score"], d["n_poems"]) for tag, d in avg_score.items()],
                                         columns=["tag", "avg_rhyme_score", "n_poems"])
avg_score_df
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
      <th>tag</th>
      <th>avg_rhyme_score</th>
      <th>n_poems</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Living</td>
      <td>1716.931374</td>
      <td>6243</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time &amp; Brevity</td>
      <td>2148.513441</td>
      <td>1468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Relationships</td>
      <td>1707.557232</td>
      <td>3856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Family &amp; Ancestors</td>
      <td>1140.750802</td>
      <td>1345</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nature</td>
      <td>1594.534573</td>
      <td>3613</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Birth</td>
      <td>3110.680493</td>
      <td>16</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Father's Day</td>
      <td>980.821727</td>
      <td>43</td>
    </tr>
    <tr>
      <th>126</th>
      <td>Kwanzaa</td>
      <td>2953.144725</td>
      <td>14</td>
    </tr>
    <tr>
      <th>127</th>
      <td>Thanksgiving</td>
      <td>1824.508874</td>
      <td>18</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Cinco de Mayo</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>129 rows × 3 columns</p>
</div>




```python
# select only tags with at least 50 poems
relevant_avg_score_df = avg_score_df[avg_score_df["n_poems"] >= 50]
print(relevant_avg_score_df.shape)
relevant_avg_score_df
```

    (99, 3)





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
      <th>tag</th>
      <th>avg_rhyme_score</th>
      <th>n_poems</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Living</td>
      <td>1716.931374</td>
      <td>6243</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time &amp; Brevity</td>
      <td>2148.513441</td>
      <td>1468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Relationships</td>
      <td>1707.557232</td>
      <td>3856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Family &amp; Ancestors</td>
      <td>1140.750802</td>
      <td>1345</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nature</td>
      <td>1594.534573</td>
      <td>3613</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Farewells &amp; Good Luck</td>
      <td>3271.599365</td>
      <td>50</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Architecture &amp; Design</td>
      <td>1050.531582</td>
      <td>85</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Funerals</td>
      <td>5091.782541</td>
      <td>85</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Classic Love</td>
      <td>4751.504513</td>
      <td>123</td>
    </tr>
    <tr>
      <th>118</th>
      <td>Anniversary</td>
      <td>4202.597493</td>
      <td>57</td>
    </tr>
  </tbody>
</table>
<p>99 rows × 3 columns</p>
</div>




```python
relevant_n_df = relevant_avg_score_df.nlargest(20, "avg_rhyme_score")

n_poems = relevant_n_df["n_poems"]

ax = sns.barplot(relevant_n_df,
                 x="avg_rhyme_score",
                 y="tag",
                 hue="n_poems",
                 dodge=False,
                 palette="RdBu",)

ax.set_xlim(0, 7700)
ax.set_xlabel("Average rhyme score")
ax.set_ylabel("Tag")
ax.legend(title='# of poems')
```




    <matplotlib.legend.Legend at 0x7f0c2988d160>




    
![png](poetry_files/poetry_29_1.png)
    



```python
relevant_n_df = relevant_avg_score_df.nsmallest(20, "avg_rhyme_score")
relevant_n_df = relevant_n_df.loc[::-1].reset_index(drop=True)
n_poems = relevant_n_df["n_poems"]

ax = sns.barplot(relevant_n_df,
                 x="avg_rhyme_score",
                 y="tag",
                 hue="n_poems",
                 dodge=False,
                 palette="RdBu",)

ax.set_xlim(0, 7700)
ax.set_xlabel("Average rhyme score")
ax.set_ylabel("Tag")
ax.legend(title='# of poems')
```




    <matplotlib.legend.Legend at 0x7f0c432cc1c0>




    
![png](poetry_files/poetry_30_1.png)
    


### Code: Rhyme Schemes (Question 2)


```python
def next_rhyme_int(prev_list):
    return max(prev_rhyme_int for _, prev_rhyme_int in prev_list) + 1

def get_rhyme_int(word, prev_list):
    for prev_word, prev_rhyme_int in prev_list:
        if word == prev_word:
            return prev_rhyme_int
        rhyme_idx = index_of_first(rhymes[word],
                                   lambda rhyme_dict:
                                       rhyme_dict["word"] == prev_word)
        if rhyme_idx != -1: # word rhymes with prev_word
            return prev_rhyme_int
        else: # search again the other way around
            if prev_word not in rhymes:
                continue
            rhyme_idx_rev = index_of_first(rhymes[prev_word],
                                           lambda rhyme_dict:
                                               rhyme_dict["word"] == word)
            if rhyme_idx_rev != -1:
                return prev_rhyme_int
    return next_rhyme_int(prev_list)

def compute_rhyme_scheme(last_word_list):
    if len(last_word_list) <= 1:
        return []
    rhyme_scheme = [(last_word_list[0], 0)]
    curr_int = 0
    for i in range(1, len(last_word_list)):
        last_word_i = last_word_list[i]
        if last_word_i not in rhymes:
            rhyme_scheme.append((last_word_i, next_rhyme_int(rhyme_scheme)))
            continue
        rhyme_int = get_rhyme_int(last_word_i, rhyme_scheme)
        rhyme_scheme.append((last_word_i, rhyme_int))
    return rhyme_scheme         
```


```python
# Testing
print(compute_rhyme_scheme(["augmented", "unprecedented", "drew", "to"]))
print(compute_rhyme_scheme(["augmented", "augmented"]))
print(compute_rhyme_scheme(["augmented", "augmented", "unprecedented"]))
print(last_words[0])
print(compute_rhyme_scheme(last_words[0]))
print(last_words[12889])
print(compute_rhyme_scheme(last_words[12889]))
print(last_words[12882])
print(compute_rhyme_scheme(last_words[12882]))
```

    [('augmented', 0), ('unprecedented', 0), ('drew', 1), ('to', 1)]
    [('augmented', 0), ('augmented', 0)]
    [('augmented', 0), ('augmented', 0), ('unprecedented', 0)]
    ['stapler', 'press', 'lacks', 'grip', 'bootstrap', 'belt', 'cords', 'broke', 'empty', 'glue', 'clack', 'shut', 'blanket', 'newel', 'dividing', 'two', 'sagebrush', 'wing', 'only', 'air']
    [('stapler', 0), ('press', 1), ('lacks', 2), ('grip', 3), ('bootstrap', 4), ('belt', 5), ('cords', 6), ('broke', 7), ('empty', 8), ('glue', 9), ('clack', 10), ('shut', 11), ('blanket', 12), ('newel', 13), ('dividing', 14), ('two', 9), ('sagebrush', 15), ('wing', 16), ('only', 17), ('air', 18)]
    ['marathon', 'don', 'helicon', 'on', 'acheron', 'shone', 'gone']
    [('marathon', 0), ('don', 0), ('helicon', 0), ('on', 0), ('acheron', 1), ('shone', 2), ('gone', 0)]
    ['away', 'gray', 'day', 'stray', 'play', 'decay', 'say']
    [('away', 0), ('gray', 0), ('day', 0), ('stray', 0), ('play', 0), ('decay', 0), ('say', 0)]



```python
rhyme_schemes = { id: compute_rhyme_scheme(l) for id, l in tqdm(last_words.items()) }
```

    100%|██████████| 13753/13753 [02:59<00:00, 76.72it/s] 



```python
def matches_rhyme_scheme(rhyme_scheme_list, rhyme_scheme_str):
    def str_to_list():
        l = []
        seen_chars = {}
        curr_counter = 0
        for c in rhyme_scheme_str:
            if c in seen_chars:
                l.append(seen_chars[c])
            else:
                l.append(curr_counter)
                seen_chars[c] = curr_counter
                curr_counter += 1
        return l
    
    def get_window(i, j):
        return [num for _, num in rhyme_scheme_list[i:j]]
    
    derived_list = str_to_list()
    for i in range(len(rhyme_scheme_list) - len(derived_list) + 1):
        for j in range(i + (len(derived_list) - 1), len(rhyme_scheme_list)):
            window = get_window(i, j + 1)
            if window == derived_list:
                return True
    return False
```


```python
# Testing
print(matches_rhyme_scheme(compute_rhyme_scheme(["augmented", "unprecedented", "drew", "to"]), "abab"))
print(matches_rhyme_scheme(compute_rhyme_scheme(["augmented", "unprecedented", "drew", "to"]), "aabb"))
print(matches_rhyme_scheme(compute_rhyme_scheme(["augmented", "unprecedented", "drew", "to", "hello", "goodbye"]), "aabb"))
print(matches_rhyme_scheme(compute_rhyme_scheme(["augmented", "hello", "unprecedented", "drew", "to", "hello", "goodbye"]), "aabb"))
```

    False
    True
    True
    False



```python
schemes = ["abba", "abab", "aabb", "abcb"]
for scheme in schemes:
    df[scheme] = False
df
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
      <th>Title</th>
      <th>Poem</th>
      <th>Poet</th>
      <th>Tags</th>
      <th>id</th>
      <th>rhyme_score</th>
      <th>abab</th>
      <th>aabb</th>
      <th>abcb</th>
      <th>abba</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Objects Used to Prop Open a Window</td>
      <td>Dog bone, stapler,\ncribbage board, garlic pre...</td>
      <td>Michelle Menting</td>
      <td>NaN</td>
      <td>0</td>
      <td>595.478947</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Church</td>
      <td>The old cupola glinted above the clouds, shone...</td>
      <td>Lucia Cherciu</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Look for Me</td>
      <td>Look for me under the hood\nof that old Chevro...</td>
      <td>Ted Kooser</td>
      <td>NaN</td>
      <td>2</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wild Life</td>
      <td>Behind the silo, the Mother Rabbit\nhunches li...</td>
      <td>Grace Cavalieri</td>
      <td>NaN</td>
      <td>3</td>
      <td>2696.474896</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Umbrella</td>
      <td>When I push your button\nyou fly off the handl...</td>
      <td>Connie Wanek</td>
      <td>NaN</td>
      <td>4</td>
      <td>569.898634</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13849</th>
      <td>1-800-FEAR</td>
      <td>We'd  like  to  talk  with  you  about  fear t...</td>
      <td>Jody Gladding</td>
      <td>Living,Social Commentaries,Popular Culture</td>
      <td>13849</td>
      <td>4645.890308</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13850</th>
      <td>The Death of Atahuallpa</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13850</td>
      <td>30.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13851</th>
      <td>Poet's Wish</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13851</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13852</th>
      <td>0</td>
      <td>Philosophic\nin its complex, ovoid emptiness,\...</td>
      <td>Hailey Leithauser</td>
      <td>Arts &amp; Sciences,Philosophy</td>
      <td>13852</td>
      <td>3388.751715</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13853</th>
      <td>!</td>
      <td>Dear Writers, I’m compiling the first in what ...</td>
      <td>Wendy Videlock</td>
      <td>Relationships,Gay, Lesbian, Queer,Arts &amp; Scien...</td>
      <td>13853</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>13854 rows × 10 columns</p>
</div>




```python
for id, rhyme_scheme in tqdm(rhyme_schemes.items()):
    index = np.flatnonzero(df["id"] == id)[0]
    for scheme in schemes:
        df.at[index, scheme] = matches_rhyme_scheme(rhyme_scheme, scheme)
df
```

    100%|██████████| 13753/13753 [04:53<00:00, 46.85it/s] 





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
      <th>Title</th>
      <th>Poem</th>
      <th>Poet</th>
      <th>Tags</th>
      <th>id</th>
      <th>rhyme_score</th>
      <th>abab</th>
      <th>aabb</th>
      <th>abcb</th>
      <th>abba</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Objects Used to Prop Open a Window</td>
      <td>Dog bone, stapler,\ncribbage board, garlic pre...</td>
      <td>Michelle Menting</td>
      <td>NaN</td>
      <td>0</td>
      <td>595.478947</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Church</td>
      <td>The old cupola glinted above the clouds, shone...</td>
      <td>Lucia Cherciu</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Look for Me</td>
      <td>Look for me under the hood\nof that old Chevro...</td>
      <td>Ted Kooser</td>
      <td>NaN</td>
      <td>2</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wild Life</td>
      <td>Behind the silo, the Mother Rabbit\nhunches li...</td>
      <td>Grace Cavalieri</td>
      <td>NaN</td>
      <td>3</td>
      <td>2696.474896</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Umbrella</td>
      <td>When I push your button\nyou fly off the handl...</td>
      <td>Connie Wanek</td>
      <td>NaN</td>
      <td>4</td>
      <td>569.898634</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13849</th>
      <td>1-800-FEAR</td>
      <td>We'd  like  to  talk  with  you  about  fear t...</td>
      <td>Jody Gladding</td>
      <td>Living,Social Commentaries,Popular Culture</td>
      <td>13849</td>
      <td>4645.890308</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13850</th>
      <td>The Death of Atahuallpa</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13850</td>
      <td>30.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13851</th>
      <td>Poet's Wish</td>
      <td></td>
      <td>William Jay Smith</td>
      <td>NaN</td>
      <td>13851</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13852</th>
      <td>0</td>
      <td>Philosophic\nin its complex, ovoid emptiness,\...</td>
      <td>Hailey Leithauser</td>
      <td>Arts &amp; Sciences,Philosophy</td>
      <td>13852</td>
      <td>3388.751715</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13853</th>
      <td>!</td>
      <td>Dear Writers, I’m compiling the first in what ...</td>
      <td>Wendy Videlock</td>
      <td>Relationships,Gay, Lesbian, Queer,Arts &amp; Scien...</td>
      <td>13853</td>
      <td>0.000000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>13854 rows × 10 columns</p>
</div>




```python
# Obtain frequencies of common rhyme schemes
freqs = df.sum(axis=0).iloc[5:9] / df.shape[0]
freqs
```

    /tmp/ipykernel_1816/3619108465.py:2: FutureWarning: The default value of numeric_only in DataFrame.sum is deprecated. In a future version, it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid columns or specify the value of numeric_only to silence this warning.
      freqs = df.sum(axis=0).iloc[5:9] / df.shape[0]





    abab    0.016963
    aabb    0.017396
    abcb     0.02613
    abba    0.004981
    dtype: object




```python
# Plot the frequencies in a bar plot
ax = sns.barplot(x=freqs.index,
                 y=freqs.values)

ax.set_xlabel("Rhyme scheme")
ax.set_ylabel(f"Frequency (out of all {df.shape[0]} poems)")
```




    Text(0, 0.5, 'Frequency (out of all 13854 poems)')




    
![png](poetry_files/poetry_40_1.png)
    

