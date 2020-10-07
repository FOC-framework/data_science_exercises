import pandas as pd
import numpy as np
import string

def w(n):
    w = 0;
    for i in range(n):
        print('i='+str(i))
        w = w * w + 1
    return w

print(w(2))