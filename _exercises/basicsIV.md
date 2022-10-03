---
layout: page
title: Basics IV Coding Exercises
description: Select exercises from Python Crash Course, chapter 6
---

# Exercises

6-1, 6-2, 6-3, 6-4, 6-5, 6-7, 6-8, 6-9, 6-11

## 6-1


```python
person = {
    "first_name": "Finn",
    "last_name": "Tekverk",
    "age": 21,
    "city": "King's Park"
}
print(person["first_name"])
print(person["last_name"])
print(person["age"])
print(person["city"])
```

    Finn
    Tekverk
    21
    King's Park


## 6-2


```python
fav_nums = {
    "Adam": 10,
    "Finn": 15,
    "Josh": 3,
    "Joe": 23,
    "Ryan": 55
}
for name, fav_num in fav_nums.items():
    print(f"{name}'s favorite number is {fav_num}")
```

    Adam's favorite number is 10
    Finn's favorite number is 15
    Josh's favorite number is 3
    Joe's favorite number is 23
    Ryan's favorite number is 55


## 6-3


```python
glossary = {
    "list": "An ordered, indexed collection of Python objects.",
    "list comprehension": "A terse method for constructing Python lists.",
    "conditional statement": "A logical expression which evaluates to either true or false.",
    "dictionary": "A data structure which maps keys to values.",
    "string": "An ordered collection of characters."
}
for term, defn in glossary.items():
    print(f"{term}: {defn}\n")
```

    list: An ordered, indexed collection of Python objects.
    
    list comprehension: A terse method for constructing Python lists.
    
    conditional statement: A logical expression which evaluates to either true or false.
    
    dictionary: A data structure which maps keys to values.
    
    string: An ordered collection of characters.
    


## 6-4


```python
glossary = {
    "list": "An ordered, indexed collection of Python objects.",
    "list comprehension": "A terse method for constructing Python lists.",
    "conditional statement": "A logical expression which evaluates to either true or false.",
    "dictionary": "A data structure which maps keys to values.",
    "string": "An ordered collection of characters.",
    "iterable": "An object capable of returning its members one at a time.",
    "generator": "A function which returns a generator iterator.",
    "lambda": "An anonymous inline function consisting of a single expression which is evaluated when the function is called.",
    "boolean": "A variable which is either True or False.",
    "method": "A function which is defined inside a class body."
}
for term, defn in glossary.items():
    print(f"{term}: {defn}\n")
```

    list: An ordered, indexed collection of Python objects.
    
    list comprehension: A terse method for constructing Python lists.
    
    conditional statement: A logical expression which evaluates to either true or false.
    
    dictionary: A data structure which maps keys to values.
    
    string: An ordered collection of characters.
    
    iterable: An object capable of returning its members one at a time.
    
    generator: A function which returns a generator iterator.
    
    lambda: An anonymous inline function consisting of a single expression which is evaluated when the function is called.
    
    boolean: A variable which is either True or False.
    
    method: A function which is defined inside a class body.
    


## 6-5


```python
rivers = {
    "Nile": "Egypt",
    "Amazon": "Brazil",
    "Ohio": "The United States of America"
}
for river, place in rivers.items():
    print(f"The {river} river runs through {place}.")
```

    The Nile river runs through Egypt.
    The Amazon river runs through Brazil.
    The Ohio river runs through The United States of America.


## 6-7


```python
people = [
    {
        "first_name": "Finn",
        "last_name": "Tekverk",
        "age": 21,
        "city": "King's Park"
    },
    {
        "first_name": "Adam",
        "last_name": "Peters",
        "age": 22,
        "city": "Andover"
    },
    {
        "first_name": "Josh",
        "last_name": "Kalet",
        "age": 21,
        "city": "Hoboken"
    }
]

for person in people:
    for key, val in person.items():
        print(f"{key}: {val}")
    print()
```

    first_name: Finn
    last_name: Tekverk
    age: 21
    city: King's Park
    
    first_name: Adam
    last_name: Peters
    age: 22
    city: Andover
    
    first_name: Josh
    last_name: Kalet
    age: 21
    city: Hoboken
    


## 6-8


```python
pets = [
    {
        "kind": "dog",
        "owner": "finn"
    },
    {
        "kind": "cat",
        "owner": "josh"
    },
    {
        "kind": "fish",
        "owner": "ryan"
    }
]

for pet in pets:
    for key, val in pet.items():
        print(f"{key}: {val}")
    print()
```

    kind: dog
    owner: finn
    
    kind: cat
    owner: josh
    
    kind: fish
    owner: ryan
    


## 6-9


```python
favorite_places = {
    "finn": ["Medford"],
    "josh": ["Medford", "NYC", "Boston"],
    "joe": ["Medford", "Philadelphia"],
}
for name, places in favorite_places.items():
    print(f"{name}: {', '.join(places)}")
```

    finn: Medford
    josh: Medford, NYC, Boston
    joe: Medford, Philadelphia


## 6-11


```python
cities = {
    "Boston": {
        "country": "USA",
        "population": 689326,
        "fact": "The very first chocolate factory in the USA was in Boston."
    },
    "NYC": {
        "country": "USA",
        "population": 8380000,
        "fact": "New York City became the first capital of the United States in 1789."
    },
    "Philadelphia": {
        "country": "USA",
        "population": 1582000,
        "fact": "The city is home to America's first zoo."
    }
}
for city_name, city_info in cities.items():
    print(f"{city_name}:")
    for key, val in city_info.items():
        print(f"    {key}: {val}")
    print()
```

    Boston:
        country: USA
        population: 689326
        fact: The very first chocolate factory in the USA was in Boston.
    
    NYC:
        country: USA
        population: 8380000
        fact: New York City became the first capital of the United States in 1789.
    
    Philadelphia:
        country: USA
        population: 1582000
        fact: The city is home to America's first zoo.
    

