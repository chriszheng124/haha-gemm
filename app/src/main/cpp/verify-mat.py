#!/usr/local/bin/python3

from numpy import *

#  a = random.randint(10, size = 9).tolist()
#  b = random.randint(10, size = 9).tolist()
#  print("a:")
#  print(a)
#  print("b:")
#  print(b)

#  ma = reshape(a, (3, 3))
#  mb = reshape(b, (3, 3))

ma = arange(16).reshape(4, 4)
a = ma.ravel('F').tolist()
print("a: ")
print(a)
mb = arange(16).reshape(4, 4)
b = mb.ravel('F').tolist()
print("b: ")
print(b)
mc = dot(ma, mb)
print("mc: ")
print(mc)

#  if(ma == mb).all():
    #  print("ma == mb")

