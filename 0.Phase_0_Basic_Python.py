# USER INPUT/OUTPUT

# Single input
name = input("Enter your name: ")
print("Hello,", name)

# Multiple inputs in one line
x, y = map(int, input("Enter two numbers: ").split())
print("Sum:", x + y)

# Reading N lines
n = int(input("How many lines? "))
lines = [input(f"Line {i+1}: ") for i in range(n)]
print("Collected lines:", lines)

# Output formatting
print(f"My name is {name}, I am {x} years old")
print("Pi approx = {:.2f}".format(3.14159))

# DATA TYPES

# Primitive types
i = 42
f = 3.14
b = True
s = "Python"
n = None

# Collections
lst = [1, 2, 3]          # list
tpl = (1, 2, 3)          # tuple
st = {1, 2, 2, 3}        # set
d = {"a": 1, "b": 2}     # dict

# Type conversion
print(int("10"), float("2.5"), str(100))

# Checking types
print(type(lst), isinstance(s, str))

# IF-ELSE STATEMENTS

x = int(input("Enter a number: "))

if x < 0:
    print("Negative")
elif x == 0:
    print("Zero")
else:
    print("Positive")

# Inline if (ternary)
parity = "even" if x % 2 == 0 else "odd"
print("Number is", parity)

# SWITCH STATEMENT

cmd = input("Enter command (start/stop/pause): ")

match cmd:
    case "start":
        print("System starting...")
    case "stop":
        print("System stopping...")
    case "pause":
        print("System paused")
    case _:
        print("Unknown command")
        
# ARRAYS, STRINGS

# Arrays (lists in Python)
arr = [5, 2, 9, 1]
arr.append(7)
arr.insert(0, 42)
arr.pop()
arr.sort()
print(arr)

# Strings
text = "Hello Arjun"
print(text.lower(), text.upper())
print(text.replace("Arjun", "World"))
print("Arjun" in text)
print(text[::-1])  # reverse

# FOR LOOPS

# Range loop
for i in range(5):
    print(i)

# Enumerate
for idx, val in enumerate(["a", "b", "c"]):
    print(idx, val)

# Dict iteration
d = {"x": 1, "y": 2}
for k, v in d.items():
    print(k, v)
    
# WHILE LOOPS

n = 5
while n > 0:
    print("n =", n)
    n -= 1

# Sentinel loop
while True:
    s = input("Enter text (or quit): ")
    if s == "quit":
        break
    print("You typed:", s)
    
# FUNCTIONS

# Immutable (acts like pass by value)
def increment(x):
    x += 1
    return x

num = 10
print(increment(num), num)  # 11, 10

# Mutable (acts like pass by reference)
def mutate_list(lst):
    lst.append("new")
    lst[0] = "changed"

A = ["orig"]
mutate_list(A)
print(A)  # ["changed", "orig", "new"]

# LAMBDA FUNCTIONS

from typing import List, Tuple, Optional, Callable
from collections import defaultdict, OrderedDict, deque

# lambda format
# lambda input : expression
x = lambda x: x ** 2
print(x(9))

x = lambda x, y: x + y
print(x(2, 3))

x = lambda x: "even" if x % 2 == 0 else "odd"
print(x(3))
print(x(2))

# higer order functions -> functions which need some parameters as well as another function as its parameter
# example -> print the sum of even, odd and elements div by 3 from the following List

list1 = [11, 14, 21, 23, 56, 78, 45, 29, 28]
xx = lambda x: x % 2 == 0  # even
xy = lambda x: x % 2 != 0  # odd
xz = lambda x: x % 3 == 0  # divisible by 3


def eg_of_hof(fun: Callable, vec: List):
    res = 0
    for item in vec:
        if fun(item):
            res += item
    return res


print(eg_of_hof(xx, list1))
print(eg_of_hof(xy, list1))
print(eg_of_hof(xz, list1))

# map reduce and filter

# map -> format = map(function, iterable)
list2 = [2, 4, 5, 1, 4]
list2 = list(map(lambda x: x * 2, list2))
print(list2)

# Example : suppose we have some data from database and we want to fetch only names from these data -> then we can do this using map
Student = [
    {
        'name': 'arjun',
        'age': 20
    },
    {
        'name': 'gokul',
        'age': 21
    },
    {
        'name': 'pritha',
        'age': 34
    }
]
print(Student)

names = list(map(lambda studentData: studentData['name'], Student))
print(names)

# filter -> format  = filter (function, iterable)
list3 = [1, 2, 3, 4, 5, 6, 7]
temp = list(filter(lambda x: x > 4, list3))
print(temp)

# reduce -> format = functools.reduce(function, iterable)
import functools

list4 = [1, 2, 3, 4, 5]

# greatest element
print(functools.reduce(lambda x, y: x if x > y else y, list4))

# smallest element
print(functools.reduce(lambda x, y: x if x < y else y, list4))

# sum of all elements
print(functools.reduce(lambda x, y: x + y, list4))

# list comprehension
list5 = [1, 2, 3, 4, 5]
list6 = [i * 2 for i in list5]
print(list6)

list7 = [i for i in list5 if i % 2 == 0]
print(list7)

# similarly dictionary comprehension can be done

# Safe copy
import copy
def safe_process(lst):
    local = copy.deepcopy(lst)
    local.append("temp")
    return local

B = [1, 2, 3]
print(safe_process(B), B)

# TIME COMPLEXITY

# O(1) → constant
def get_first(arr):
    return arr[0]

# O(n) → linear
def print_all(arr):
    for x in arr:
        print(x)

# O(n^2) → quadratic
def print_pairs(arr):
    for i in arr:
        for j in arr:
            print(i, j)

# O(log n) → binary search
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1