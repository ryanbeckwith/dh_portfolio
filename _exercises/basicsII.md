---
layout: page
title: Basics II Coding Exercises
description: Select exercises from Python Crash Course, chapters 3 and 4
---

# Exercises

3-1, 3-2, 3-3, 3-4, 3-5, 3-6, 3-7, 3-8, 4-1, 4-2, 4-3, 4-6, 4-10, 4-11

## 3-1


```python
names = ["Adam", "Finn", "Josh", "Joe"]
print(names[0])
print(names[1])
print(names[2])
print(names[3])
```

    Adam
    Finn
    Josh
    Joe


## 3-2


```python
message = "Hello, "
for name in names:
    print(message + name)
```

    Hello, Adam
    Hello, Finn
    Hello, Josh
    Hello, Joe


## 3-3


```python
cars = ["Honda Civic", "Honda Accord", "Honda CR-V"]
for car in cars:
    print(f"I would like to own a {car}")
```

    I would like to own a Honda Civic
    I would like to own a Honda Accord
    I would like to own a Honda CR-V


## 3-4


```python
people = ["Me", "Myself", "I"]
for person in people:
    print(f"You're invited to dinner, {person}!")
```

    You're invited to dinner, Me!
    You're invited to dinner, Myself!
    You're invited to dinner, I!


## 3-5


```python
people = ["Me", "Myself", "I"]
for person in people:
    print(f"You're invited to dinner, {person}!")

print(f"{people[0]} can't make it")
people[0] = "Han Solo"
for person in people:
    print(f"You're invited to dinner, {person}!")
```

    You're invited to dinner, Me!
    You're invited to dinner, Myself!
    You're invited to dinner, I!
    Me can't make it
    You're invited to dinner, Han Solo!
    You're invited to dinner, Myself!
    You're invited to dinner, I!


## 3-6


```python
people = ["Me", "Myself", "I"]
for person in people:
    print(f"You're invited to dinner, {person}!")

print(f"{people[0]} can't make it")
people[0] = "Han Solo"
for person in people:
    print(f"You're invited to dinner, {person}!")

people.insert(0, "Jabba the Hutt")
people.insert(2, "Chewy")
people.append("Boba Fett")
for person in people:
    print(f"You're invited to dinner, {person}!")
```

    You're invited to dinner, Me!
    You're invited to dinner, Myself!
    You're invited to dinner, I!
    Me can't make it
    You're invited to dinner, Han Solo!
    You're invited to dinner, Myself!
    You're invited to dinner, I!
    You're invited to dinner, Jabba the Hutt!
    You're invited to dinner, Han Solo!
    You're invited to dinner, Chewy!
    You're invited to dinner, Myself!
    You're invited to dinner, I!
    You're invited to dinner, Boba Fett!


## 3-7


```python
people = ["Me", "Myself", "I"]
for person in people:
    print(f"You're invited to dinner, {person}!")

print(f"{people[0]} can't make it")
people[0] = "Han Solo"
for person in people:
    print(f"You're invited to dinner, {person}!")

people.insert(0, "Jabba the Hutt")
people.insert(2, "Chewy")
people.append("Boba Fett")
for person in people:
    print(f"You're invited to dinner, {person}!")

while len(people) > 2:
    print(f"{people.pop()} is no longer invited to dinner.")
for person in people:
    print(f"You're still invited to dinner, {person}!")
del people[0]
del people[0]
print(people)
```

    You're invited to dinner, Me!
    You're invited to dinner, Myself!
    You're invited to dinner, I!
    Me can't make it
    You're invited to dinner, Han Solo!
    You're invited to dinner, Myself!
    You're invited to dinner, I!
    You're invited to dinner, Jabba the Hutt!
    You're invited to dinner, Han Solo!
    You're invited to dinner, Chewy!
    You're invited to dinner, Myself!
    You're invited to dinner, I!
    You're invited to dinner, Boba Fett!
    Boba Fett is no longer invited to dinner.
    I is no longer invited to dinner.
    Myself is no longer invited to dinner.
    Chewy is no longer invited to dinner.
    You're still invited to dinner, Jabba the Hutt!
    You're still invited to dinner, Han Solo!
    []


## 3-8


```python
places = ["Rome", "Athens", "New Zealand", "Japan", "Iceland"]
print(places)
print(sorted(places))
print(places)
places.reverse()
print(places)
places.reverse()
print(places)
places.sort()
print(places)
places.sort(reverse=True)
print(places)
```

    ['Rome', 'Athens', 'New Zealand', 'Japan', 'Iceland']
    ['Athens', 'Iceland', 'Japan', 'New Zealand', 'Rome']
    ['Rome', 'Athens', 'New Zealand', 'Japan', 'Iceland']
    ['Iceland', 'Japan', 'New Zealand', 'Athens', 'Rome']
    ['Rome', 'Athens', 'New Zealand', 'Japan', 'Iceland']
    ['Athens', 'Iceland', 'Japan', 'New Zealand', 'Rome']
    ['Rome', 'New Zealand', 'Japan', 'Iceland', 'Athens']


## 4-1


```python
pizzas = ["Mushroom and onion", "Sausage", "Pepperoni"]
for pizza in pizzas:
    print(f"{pizza} pizza is quite good.")
print("I love pizza! I love The Republic!")
```

    Mushroom and onion pizza is quite good.
    Sausage pizza is quite good.
    Pepperoni pizza is quite good.
    I love pizza! I love The Republic!


## 4-2


```python
animals = ["dog", "cat", "fish"]
for animal in animals:
    print(f"A {animal} would make a great pet.")
print("Any of these animals would make a great pet!")
```

    A dog would make a great pet.
    A cat would make a great pet.
    A fish would make a great pet.
    Any of these animals would make a great pet!


## 4-3


```python
for i in range(1, 21):
    print(i)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20


## 4-6


```python
for i in range(1, 21, 2):
    print(i)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19


## 4-10


```python
more_animals = [*animals, "parrot", "cheetah", "lion", "walrus"]
print(f"The first three items in the list are: {more_animals[:3]}")
print(f"Three items from the middle of the list are: {more_animals[2:5]}")
print(f"The last three items in the list are: {more_animals[-3:]}")
```

    The first three items in the list are: ['dog', 'cat', 'fish']
    Three items from the middle of the list are: ['fish', 'parrot', 'cheetah']
    The last three items in the list are: ['cheetah', 'lion', 'walrus']


## 4-11


```python
pizzas = ["Mushroom and onion", "Sausage", "Pepperoni"]
friend_pizzas = pizzas[:]
pizzas.append("Pesto chicken")
friend_pizzas.append("Anchovy")
for pizza in pizzas:
    print(f"My favorite pizzas are: {pizza}")
for pizza in friend_pizzas:
    print(f"My friend's favorite pizzas are: {pizza}")
print("I love pizza! I love The Republic!")
```

    My favorite pizzas are: Mushroom and onion
    My favorite pizzas are: Sausage
    My favorite pizzas are: Pepperoni
    My favorite pizzas are: Pesto chicken
    My friend's favorite pizzas are: Mushroom and onion
    My friend's favorite pizzas are: Sausage
    My friend's favorite pizzas are: Pepperoni
    My friend's favorite pizzas are: Anchovy
    I love pizza! I love The Republic!



```python

```
