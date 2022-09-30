---
layout: page
title: Basics III Coding Exercises
description: Select exercises from Python Crash Course, chapter 5
---

# Exercises

5-1, 5-2, 5-6, 5-7, 5-8, 5-9, 5-10

## 5-1


```python
car = 'subaru'
print("Is car == 'subaru'? I predict True.")
print(car == 'subaru')

print("\nIs car == 'audi'? I predict False.")
print(car == 'audi')

print("\nIs True == False? I predict False.")
print(True == False)

print("\nIs None == None? I predict True.")
print(None == None)

print("\nIs print == print? I predict True.")
print(print == print)

print("\nIs 0 == False? I predict True.")
print(0 == False)

z = 0
print("\nIs 0 == z? I predict True.")
print(0 == z)

print("\nIs 0 == z := z + 1? I predict False.")
print(0 == (z := z + 1))

print("\nIs \"\" == False? I predict False.")
print("" == False)

print("\nIs \"\" == 0? I predict False.")
print("" == 0)
```

    Is car == 'subaru'? I predict True.
    True
    
    Is car == 'audi'? I predict False.
    False
    
    Is True == False? I predict False.
    False
    
    Is None == None? I predict True.
    True
    
    Is print == print? I predict True.
    True
    
    Is 0 == False? I predict True.
    True
    
    Is 0 == z? I predict True.
    True
    
    Is 0 == z := z + 1? I predict False.
    False
    
    Is "" == False? I predict False.
    False
    
    Is "" == 0? I predict False.
    False


## 5-2


```python
print("\nIs \"\" == ''? I predict True.")
print("" == '')

print("\nIs \"\" == '\"'? I predict False.")
print("" == '"')

print("\nIs \"\" != ''? I predict False.")
print("" != '')

print("\nIs \"a\" != \"A\"? I predict True.")
print("a" != "A")

print("\nIs \"0\" == \"0\".lower()? I predict True.")
print("" == "".lower())

print("\nIs \"A\" == \"A\".lower()? I predict False.")
print("A" == "A".lower())

print("\nIs 1 == (not 0)? I predict True.")
print(1 == (not 0))

print("\nIs 0 == 1 - 2? I predict False.")
print(0 == 1 - 2)

print("\nIs 0 != 1? I predict True.")
print(0 != 1)

print("\nIs 0 != -0? I predict False.")
print(0 != -0)

print("\nIs 0 > -1? I predict True.")
print(0 > -1)

print("\nIs 0 > -0? I predict False.")
print(0 > -0)

print("\nIs -1 < 1? I predict True.")
print(-1 < 1)

print("\nIs 1 < -1? I predict False.")
print(1 < -1)

print("\nIs 0 >= -0? I predict True.")
print(0 >= -0)

print("\nIs 0 >= (not 0)? I predict False.")
print(0 >= (not 0))

print("\nIs (not 0) <= (not 0)? I predict True.")
print((not 0) <= (not 0))

print("\nIs 0 <= -10? I predict False.")
print(0 <= -10)

print("\nIs True and 1? I predict True.")
print(1 and True)

print("\nIs False and 1? I predict False.")
print(False and 1)

print("\nIs \"\" or True? I predict True.")
print("" or True)

print("\nIs \"\" or False? I predict False.")
print("" or False)

print("\nIs [] in [[]]? I predict True.")
print([] in [[]])

print("\nIs [] in []? I predict False.")
print([] in [])

print("\nIs [] not in [[1, []]]? I predict True.")
print([] not in [[1, []]])

print("\nIs [] not in [[[]], []]? I predict False.")
print([] not in [[[]], []])

```

    
    Is "" == ''? I predict True.
    True
    
    Is "" == '"'? I predict False.
    False
    
    Is "" != ''? I predict False.
    False
    
    Is "a" != "A"? I predict True.
    True
    
    Is "0" == "0".lower()? I predict True.
    True
    
    Is "A" == "A".lower()? I predict False.
    False
    
    Is 1 == (not 0)? I predict True.
    True
    
    Is 0 == 1 - 2? I predict False.
    False
    
    Is 0 != 1? I predict True.
    True
    
    Is 0 != -0? I predict False.
    False
    
    Is 0 > -1? I predict True.
    True
    
    Is 0 > -0? I predict False.
    False
    
    Is -1 < 1? I predict True.
    True
    
    Is 1 < -1? I predict False.
    False
    
    Is 0 >= -0? I predict True.
    True
    
    Is 0 >= (not 0)? I predict False.
    False
    
    Is (not 0) <= (not 0)? I predict True.
    True
    
    Is 0 <= -10? I predict False.
    False
    
    Is True and 1? I predict True.
    True
    
    Is False and 1? I predict False.
    False
    
    Is "" or True? I predict True.
    True
    
    Is "" or False? I predict False.
    False
    
    Is [] in [[]]? I predict True.
    True
    
    Is [] in []? I predict False.
    False
    
    Is [] not in [[1, []]]? I predict True.
    True
    
    Is [] not in [[[]], []]? I predict False.
    False


## 5-6


```python
age = 10
if age < 2:
    print("You're a baby.")
elif age >= 2 and age < 4:
    print("You're a toddler.")
elif age >= 4 and age < 13:
    print("You're a kid.")
elif age >= 13 and age < 20:
    print("You're a teenager.")
elif age >= 20 and age < 65:
    print("You're an adult.")
else:
    print("You're an elder.")
```

    You're a kid.


## 5-7


```python
favorite_fruits = ["apple", "raspberry", "pineapple"]
in_statement = 'You really like '
if "apple" in favorite_fruits:
    print(in_statement + "apples!")
if "blueberry" in favorite_fruits:
    print(in_statement + "blueberries!")
if "pineapple" in favorite_fruits:
    print(in_statement + "pineapples!")
if "raspberry" in favorite_fruits:
    print(in_statement + "raspberries!")
if "kiwi" in favorite_fruits:
    print(in_statement + "kiwis!")
```

    You really like apples!
    You really like pineapples!
    You really like raspberries!


## 5-8


```python
usernames = ["admin", "adminfake", "ADMIN", "adm1n", "aadminn"]
for name in usernames:
    if name == "admin":
        print(f"Hello {name}, would you like to see a status report?")
    else:
        print(f"Hello {name}, thank you for logging in again.")
```

    Hello admin, would you like to see a status report?
    Hello adminfake, thank you for logging in again.
    Hello ADMIN, thank you for logging in again.
    Hello adm1n, thank you for logging in again.
    Hello aadminn, thank you for logging in again.


## 5-9


```python
usernames = []
if usernames:
    for name in usernames:
        if name == "admin":
            print(f"Hello {name}, would you like to see a status report?")
        else:
            print(f"Hello {name}, thank you for logging in again.")
else:
    print("We need to find some users!")
```

    We need to find some users!


## 5-10


```python
current_users = ["admin", "adminfake", "ADMIN", "adm1n", "aadminn"]
lower_users = [user.lower() for user in current_users]
new_users = ["ryan", "adam", "admin", "josh", "joe"]
for new_user in new_users:
    if new_user.lower() in lower_users:
        print(f"You need a new username: {new_user} is already in use.")
    else:
        print(f"{new_user} is an available username!")
```

    ryan is an available username!
    adam is an available username!
    You need a new username: admin is already in use.
    josh is an available username!
    joe is an available username!

