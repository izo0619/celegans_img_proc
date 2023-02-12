import numpy as np

data = range(0,70,2)
chunks = [data[x:x+10] for x in range(0, len(data), 10)]
print(chunks)