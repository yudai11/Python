
import sys
import numpy as np
import pandas as pd

if __name__ == '__main__':
    predata = sys.stdin.read()
    predata = predata.split()
    m = int(predata[0])
    data = []
    for i in range(m):
        x = float(predata[2*i + 1])
        y = float(predata[2*i + 2])
        data.append([x,y])

    data = pd.DataFrame(data)
    
    print(data)
    print(m)
    print(data.loc[1,1])
