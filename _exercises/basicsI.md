---
layout: page
title: Basics I Coding Exercises
description: Select exercises from Python Crash Course, chapter 2
---

# Select Exercises from Chapter 2
2-1, 2-2, 2-3, 2-4, 2-5, 2-6, 2-7, 2-10

# 2-1
```python
# Store a string message in a variable, then print it
message = "Hello, world!"
print(message)
```

    Hello, world!


# 2-2
```python
# Store a string message in a variable, print it, change the variable,
# and print it again
message = "Hello, world!"
print(message)
message = "Goodbye, world!"
print(message)
```

    Hello, world!
    Goodbye, world!


# 2-3
```python
name = "Ryan"
print(f'Hello {name}, would you like to learn some Python today?')
```

    Hello Ryan, would you like to learn some Python today?


# 2-4
```python
name = "Ryan"
print(name.lower())
print(name.upper())
print(name.title())
```

    ryan
    RYAN
    Ryan


# 2-5
```python
print('Albert Einstein once said, "A person who never made a mistake never tried anything new."')
```

    Albert Einstein once said, "A person who never made a mistake never tried anything new."


# 2-6
```python
famous_person = "Albert Einstein"
message = f'{famous_person} once said, "A person who never made a mistake never tried anything new."'
print(message)
```

    Albert Einstein once said, "A person who never made a mistake never tried anything new."


# 2-7
```python
famous_person = "\tAlbert Einstein\n"
print(famous_person)
print(famous_person.lstrip())
print(famous_person.rstrip())
print(famous_person.strip())
```

    	Albert Einstein
    
    Albert Einstein
    
    	Albert Einstein
    Albert Einstein


# 2-10
```python
# Comments are shown in the above code blocks for 2-1 and 2-2
```
