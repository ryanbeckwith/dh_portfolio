---
layout: page
title: Basics V Coding Exercises
description: Select exercises from Python Crash Course, chapter 8
---

# Exercises

8-1, 8-2, 8-3, 8-5, 8-6, 8-7

## 8-1


```python
def display_message():
    print("We're learning about functions!")
    
display_message()
```

    We're learning about functions!


## 8-2


```python
def favorite_book(title):
    print(f"One of my favorite books is {title}.")
    
favorite_book("Alice in Wonderland")
```

    One of my favorite books is Alice in Wonderland.


## 8-3


```python
def make_shirt(size, text):
    print(f"Size: {size}, says \"{text}\".")
    
make_shirt("Large", "Hello, World!")
```

    Size: Large, says "Hello, World!".


## 8-5


```python
def describe_city(city, country):
    print(f"{city} is in {country}.")

describe_city("Reykjavik", "Iceland")
```

    Reykjavik is in Iceland.


## 8-6


```python
def city_country(city, country):
    return f"{city}, {country}"

print(city_country("Santiago", "Chile"))
print(city_country("Reykjavik", "Iceland"))
print(city_country("Rome", "Italy"))
```

    Santiago, Chile
    Reykjavik, Iceland
    Rome, Italy


## 8-7


```python
def make_album(artist, title, num_tracks=None):
    album = { "artist": artist, "title": title }
    if num_tracks is not None:
        album["num_tracks"] = num_tracks
    return album

print(make_album("Frank Ocean", "Blond"))
print(make_album("J. Cole", "KOD"))
print(make_album("DROELOE", "A Matter of Perspective"))
print(make_album("Bruno Major", "To Let A Good Thing Die", num_tracks=10))
```

    {'artist': 'Frank Ocean', 'title': 'Blond'}
    {'artist': 'J. Cole', 'title': 'KOD'}
    {'artist': 'DROELOE', 'title': 'A Matter of Perspective'}
    {'artist': 'Bruno Major', 'title': 'To Let A Good Thing Die', 'num_tracks': 10}

