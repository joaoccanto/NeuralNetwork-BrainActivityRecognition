from activationFunctions import sigmoid
from gradientFunctions import sigmoidGradient
import numpy as np
from scipy import io
import leather

mat = io.loadmat('011-2015-1.mat')

data = mat['datainmicrovolts']
chart = leather.Chart('Dots')
chart.add_dots(data)
chart.to_svg('file1.svg')

print ("this is the value")
print (sigmoidGradient(2))
