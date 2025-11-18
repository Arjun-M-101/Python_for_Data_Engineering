# Part 1 — STRINGS (Complete, DE-focused)
import re
import csv, io
import unicodedata

print("\n---STRINGS---")
# Core cleaning & trimming
s = "  Hello World  \n"
print("strip:", s.strip())
print("lstrip:", s.lstrip())
print("rstrip:", s.rstrip())

# Case & formatting
s = "hello arjun"
print("lower:", s.lower())
print("upper:", s.upper())
print("title:", s.title())
print("capitalized:", s.capitalize())
print("f-string:", f"Name: {s.title()}, Age: {25}")
print("format:", "Name: {} Age: {}".format("Arjun", 25))

# Replace / normalize / remove characters
s = "HeLLo---World!!"
s_clean = s.replace("-", " ").replace("!!", "!").strip()
print("replace/clean:", s_clean)
s_no_special = re.sub(r"[^0-9a-zA-Z\s]", "", s)
print("no special:", s_no_special)

# Splitting and joining
line = "a,b,c,,d"
parts = line.split(",")
print("split:", parts)
parts_r = line.rsplit(",", maxsplit=1)
print("rsplit:", parts_r)
joined = "|".join([p for p in parts if p])
print("joined:", joined)

# Partition / rpartition
a, sep, b = "user@example.com".partition("@")
print("partition:", a, sep, b)

# Starts/ends/find/count
s = "prefix_data.csv"
print("startswith:", s.startswith("pre"))
print("endswith:", s.endswith(".csv"))
print("find:", s.find("needle"))    # returns -1 if not found
try:
    idx = s.index("needle")
except ValueError:
    idx = -1
print("index (safe):", idx)
print("count 'a':", s.count("a"))

# Padding / alignment / zero-fill
print("zfill:", "42".zfill(5))
print("ljust:", "hi".ljust(10))
print("rjust:", "hi".rjust(10))
print("center:", "hi".center(10))

# Character checks
print("isalpha:", "abc".isalpha())
print("isdigit:", "123".isdigit())
print("isalnum:", "abc123".isalnum())
print("isspace:", "   ".isspace())

# Strip sets
print("strip '-':", "---abc---".strip("-"))
print("strip 'xyz':", "xyabczz".strip("xyz"))

# String to bytes / encode / decode
b = "µ".encode("utf-8")
s = b.decode("utf-8")
print("encode/decode:", s)

# Safe trimming and null handling (DE pattern)
def safe_str(x):
    if x is None: return ""
    return str(x).strip()

print("safe_str None:", safe_str(None))
print("safe_str with spaces:", safe_str("   hi   "))

# Split & extract numeric tokens
s = "price: 1,234 INR"
num = re.search(r"(\d[\d,]*)", s)
if num:
    val = int(num.group(1).replace(",", ""))
    print("numeric token:", val)

# CSV-safe splitting (handles quoted commas)
row = 'a,"b,c",d'
reader = csv.reader(io.StringIO(row))
print("csv.reader:", next(reader))

# Padding/truncating fields for fixed-width
def fixed_width(s, width): return (s[:width]).ljust(width)
print("fixed_width:", fixed_width("hello", 10))

# Normalization examples: lower + strip + collapse spaces
def norm_text(s):
    return re.sub(r'\s+', ' ', (s or "").strip().lower())
print("norm_text:", norm_text("   Hello   world   from Arjun   "))

# Tokenization / n-grams (quick)
tokens = norm_text("Hello world from Arjun").split()
bigrams = list(zip(tokens, tokens[1:]))
print("tokens:", tokens)
print("bigrams:", bigrams)

# Part 2 — NUMBERS & MATH (Casting, rounding, formats)
import math
print("\n---NUMBERS & MATH---")

## Casts & safe casts
def safe_int(x, default=None):
    try: return int(x)
    except Exception: return default

def safe_float(x, default=None):
    try: return float(x)
    except Exception: return default

print("safe_int '123a':", safe_int("123a", -1))
print("safe_float '12.3a':", safe_float("12.3a", -1))

## Rounding & formatting
print("round:", round(3.14159, 2))
print("format number:", format(1234567.89, ","))
print("format 2f:", "{:.2f}".format(1.2345))
print("f-string 2f:", f"{1.2345:.2f}")

## Math helpers
print("math floor:", math.floor(2.9))
print("math ceil:", math.ceil(2.1))
print("math sqrt:", math.sqrt(16))
print("math log base 10:", math.log(100, 10))
print("abs(-5):", abs(-5))
print("min, max:", min([1,2,3]), max([1,2,3]))

## Decimal for exactness (money)
from decimal import Decimal, getcontext
getcontext().prec = 10
print("Decimal sum:", Decimal("10.23") + Decimal("0.07"))

## Numeric normalization patterns (DE)
# Remove commas:
print("int with commas:", int("1,234".replace(",", "")))
# Convert booleans to ints:
print("int(True):", int(True))
# Percent strings:
print("percent string to float:", float("12.5%".strip("%"))/100)

## Vectorized operations (numpy)
import numpy as np
arr = np.array([1.2, 3.4])
print("numpy mean:", arr.mean())
print("numpy sum:", arr.sum())
print("numpy round:", arr.round(2))

# Part 3 — COLLECTIONS (Lists, Tuples, Sets)
print("\n---COLLECTIONS---")

## Lists — basics & transformations
lst = [1,2,3]
lst.append(4)
lst.extend([5,6])
lst.insert(0, 0)
lst.pop()    # removes last
lst.pop(0)   # removes first
lst.remove(2) if 2 in lst else None
lst.clear()
lst = [1,2,3,4,2]

## Comprehensions and idioms (DE patterns)
print("squares:", [x*x for x in lst])
print("filtered:", [x for x in lst if x>2])
print("flattened:", [y for x in [[1,2],[3,4]] for y in x])
print("unique:", list(dict.fromkeys(lst)))

## Map / filter / reduce
print("map:", list(map(lambda x: x*2, lst)))
print("filter:", list(filter(lambda x: x%2==0, lst)))
from functools import reduce
print("reduce:", reduce(lambda a,b: a+b, lst, 0))

## Sorting & keys
people = [{"name":"a","age":30},{"name":"b","age":25}]
people_sorted = sorted(people, key=lambda x: x["age"])
print("sorted people:", people_sorted)
people.sort(key=lambda x: x['age'], reverse=True)
print("reverse sorted people:", people)

## Slicing & stepping
a = [1,2,3,4,5]
print("slicing:", a[1:5:2])
print("reverse:", a[::-1])

## Tuples
t = (1,2,3)
t = tuple(list(t) + [4])
a,b,*rest = [1,2,3,4]
print("tuple unpacking:", a,b,rest)

## Sets — operations & DE uses
s = set([1,2,2,3])
s.add(4); s.discard(2)
print("set ops:", s.union({5}), s.intersection({2,3}), s.symmetric_difference({3,4}))

## Common DE patterns

# Remove duplicates while keeping order:
seq = [1,2,2,3]
print("dedup:", list(dict.fromkeys(seq)))

# Chunk list into pieces (for batch writes):
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]
print("chunks:", list(chunks([1,2,3,4,5],2)))

# Flatten nested lists (recursive safe):
def flatten(l):
    for x in l:
        if isinstance(x, (list, tuple)):
            yield from flatten(x)
        else:
            yield x
print("flattened:", list(flatten([[1,2],[3,[4,5]]])))

# Part 4 — DICTS & JSON (Flatten/merge/transform)

# Dicts are JSON-like; often used to transform nested JSON from APIs.

## Basic dict ops
print("\n---DICT & JSON---")
d = {"a":1}
v = d.get("b", 0)
d["c"] = 3
d.update({"d":4})
d.pop("a", None)

## Iteration & comprehension
print("dict comp:", {k: v*2 for k,v in d.items() if isinstance(v, int) and v>1})
for k,v in d.items():
    pass  # iteration demo; do nothing

## Nested get / safe extraction
def nested_get(d, path, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict): return default
        cur = cur.get(p)
        if cur is None: return default
    return cur
test = {"x":{"y":{"z":1}}}
print("nested_get:", nested_get(test,"x.y.z"))

## Flatten nested JSON -> single-level dict
def flatten_json(y):
    out = {}
    def rec(x, name=""):
        if isinstance(x, dict):
            for k,v in x.items():
                rec(v, f"{name}{k}." if name else f"{k}.")
        elif isinstance(x, list):
            for i,v in enumerate(x):
                rec(v, f"{name}{i}.")
        else:
            out[name[:-1]] = x
    rec(y)
    return out
nested = {"a":{"b":2},"c":[3,4]}
print("flatten_json:", flatten_json(nested))

## Merge dicts carefully (DE pattern)
old = {"a":1,"b":2}
new = {"b":10,"c":30}
merged = {**old, **new}

# or deep merge
def deep_merge(a,b):
    for k,v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            deep_merge(a[k], v)
        else:
            a[k]=v
    return a
print("merged:", merged)

## JSON load/dump
import json
py = json.loads('{"a":1}')
s = json.dumps(py, indent=2, ensure_ascii=False)
print("json dump:", s)

## Handling nested arrays of dicts
data = {"items":[{"x":1},{"x":2}]}
rows = []
for item in data["items"]:
    flat = flatten_json(item)
    rows.append(flat)
print("rows:", rows)

## Convert dicts to rows for Pandas
import pandas as pd
nested_json = {"user":{"name":"arjun"}}
df = pd.json_normalize(nested_json, sep="_")
print("pd.json_normalize:", df)

## Key normalization patterns
def normalize_keys(d):
    return {k.strip().lower().replace(" ", "_"): v for k,v in d.items()}
print("normalize_keys:", normalize_keys({" User Name ":"a"}))

# Part 5 — DATES, TIMES, REGEX, ENCODING (DE heavy)

print("\n---DATES, TIMES, REGEX, ENCODING (DE heavy)---")
from datetime import datetime, date, timedelta

## datetime basics
dt = datetime.utcnow()
dt_iso = dt.isoformat()
dt2 = datetime.strptime("2024-06-12", "%Y-%m-%d")
print("dt, dt_iso, dt2:", dt, dt_iso, dt2)
print("add day:", (dt + timedelta(days=1)).date())

## Pandas datetime convenience
s = pd.to_datetime(["2024-06-12","2024-06-13"])
df = pd.DataFrame({'date_str':["2024-06-12","2024-06-13"]})
df['date'] = pd.to_datetime(df['date_str'], errors='coerce')
df['year'] = df['date'].dt.year
print("datetime years:", df['year'])

## Timezone handling (pytz or zoneinfo)
from datetime import timezone
now_utc = datetime.now(timezone.utc)
try:
    from zoneinfo import ZoneInfo
    dtz = datetime.now(ZoneInfo("Asia/Kolkata"))
    print("zoneinfo:", dtz)
except ImportError:
    pass  # if python <3.9, skip

## Regex — essential DE patterns
p = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
m = p.search("Date:2024-06-12")
if m:
    year,month,day = m.groups()
    print("regex extract:", year,month,day)
print("replace spaces:", re.sub(r"\s+"," ", "a   b   c")) # replace:
print("findall:", re.findall(r"\d+", "abc123def45")) # findall

## Encoding quirks & normalization
print("encoded:", "µ".encode('utf-8', errors='replace'))
print("unicode normalized:", unicodedata.normalize("NFKC", "äöü"))

# Use csv for safe parsing
row = 'a,"b,c",d'
reader = csv.reader(io.StringIO(row), quoting=csv.QUOTE_MINIMAL)
print("safe csv read:", next(reader))

from dateutil import parser
dt = parser.parse("12 Jun 2024 10:00 AM")
print("parser dt:", dt)

print("fromtimestamp:", datetime.fromtimestamp(1690000000))

# Part 6 — PANDAS, PYSPARK TRANSFORMS & UTILITIES
print("\n---PANDAS, PYSPARK, UTILITIES---")

## Pandas: common DE ops
df = pd.DataFrame({'col':[None,1],'id':["1",None],'qty':[1,2],'price':[10,20],'user_id':[1,1],'order_id':[1,2],'json_col':[{"foo":"bar"},None]})
df['col'].fillna(0, inplace=True)
df.dropna(subset=['id','col'], inplace=True)
df['id'] = df['id'].astype('Int64')
df['total'] = df['qty'] * df['price']
agg = df.groupby('user_id').agg(total=('total','sum'), cnt=('order_id','count')).reset_index()
df2 = df.copy()
df = df.merge(df2, on='id', how='left', suffixes=('', '_right'))

df_json = pd.json_normalize(df['json_col'].dropna().tolist()).add_prefix('json_')
df = df.join(df_json)

# Chunked reading
def process(chunk): pass  # stub for demo
for chunk in pd.read_csv(io.StringIO("id\n1\n2"), chunksize=1):
    process(chunk)

# PySpark: demo (will not run without Spark but syntax shown)

## PySpark: core transformations (snippets)
'''
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, regexp_extract, explode, to_date, to_timestamp
spark = SparkSession.builder.appName("app").getOrCreate()
df = spark.read.csv("file.csv", header=True, inferSchema=True)

df = df.withColumn("price", col("price").cast("double"))
df = df.withColumn("total", col("qty") * col("price"))

df = df.withColumn("zipcode", regexp_extract(col("address"), r"(\\d{5})", 1))
df = df.withColumn("item", explode(col("items")))
df = df.join(df2, "id", "left")
df.write.mode("overwrite").partitionBy("region").parquet("/out/path")
df.write.jdbc(url="jdbc:postgresql://host:5432/db", table="tbl", mode="append", properties={"user":"u","password":"p","driver":"org.postgresql.Driver"})
'''

## Utilities — safe cast, parse, mask, hash
import hashlib, base64

def sha256_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def mask_string(s, keep_right=4):
    s = str(s)
    return "*"*(len(s)-keep_right) + s[-keep_right:]

def to_int_or_none(x):
    try: return int(x)
    except: return None

def safe_div(a,b):
    try: return a/b
    except Exception: return None

# Batch write, connection patterns, and Transformer class skeleton omitted for brevity; see previous message for patterns.

print("sha256 hash of 'test':", sha256_hash('test'))
print("masked string 'arjun1234':", mask_string('arjun1234'))
print("to_int_or_none '123':", to_int_or_none('123'))
print("safe_div 10/2:", safe_div(10,2))

## Transformer class skeleton (use in your ETL projects)
class Transformer:
    def __init__(self, cfg):
        self.cfg = cfg

    def clean_text(self, s):
        if s is None: return ""
        s = str(s)
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r'\s+',' ', s).strip()
        return s

    def to_int(self, x, default=None):
        try: return int(x)
        except: return default

    def parse_date(self, s, fmt=None):
        if pd.isna(s): return None
        if fmt:
            try: return datetime.strptime(s, fmt)
            except: return None
        try:
            return parser.parse(s)
        except:
            return None

    def normalize_row(self, row):
        row['name'] = self.clean_text(row.get('name'))
        row['age'] = self.to_int(row.get('age'))
        row['created_at'] = self.parse_date(row.get('created_at'))
        return row

print("Transformer demo:", Transformer({}).normalize_row({'name':' arjun ','age':'32','created_at':'2024-01-01'}))

# Quick checklist & DE best-practices — reference only, not executed
'''
* Always validate inputs (`None`, `""`, `NaN`) early.
* Use `try/except` for safe casts and log failures.
* Use vectorized Pandas ops or Spark transformations — avoid Python loops on large datasets.
* Use `csv`, `json`, `pd.json_normalize`, `spark.read.*` for robust reading — avoid naive `split(",")`.
* For DB writes: batch inserts, or DB-native bulk APIs; prefer Spark JDBC with partitioning for large datasets.
* Normalize keys & types early (canonical names, lowercased keys, typed columns).
* Log counts: rows in/out, null counts, error rows — crucial for debugging.
* Checkpoint intermediate data (bronze/silver/gold) when pipeline is critical.
'''
