a = 5
b = 3
if a>b:
        a=a*2+3
        b=b-6
        c=a/b
        print (c)

print ("next")
c=a+b+\
    10*a-b/4-\
    5+a*3
print (c)

print ("next")

a=5
b=3
if a>b:
        print ("True")
        print (a)
else:
    print("False")
    print(b)

print ("next")

a=1
b=10
while a<b:
        a+=1
        print (a)

print ("next")


for i in range (1, 10):
    print(i)

print ("next")


def binhphuong(number):
    return number*number

print(binhphuong(5))

print ("next")


import numpy as np

a=np.array([0,1,2,3,4,5])
# Hien thi mang a
print(a)
# Hiển thị số chieu cua a
print(a.ndim)
#hien thi hinh dang cua a
print(a.shape)
#hien thi phan tu co gia tri lớn hơn 3
print(a[a>3])
#thay doi gia tri
a[a>3]=10
# Thay doi hinh dang
b=a.reshape((3,2))
# Hien thi mang b
print(b)
# gan gia tri phan tu b
b[2][0]=50
# Nhan cac gia tri cua b voi 2
c=b*2
#hien thi mang c
print (c)

print("next")

import pandas as pd

dt = pd.read_csv("play_tennis.csv", delimiter=',')
dt.head()
dt.tail()

dt.loc[3:8]
dt.iloc[:, 3:6]
dt.iloc[5:10, 3:4]
dt.Outlook

