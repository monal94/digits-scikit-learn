# Import
from sklearn.preprocessing import scale
from sklearn import datasets

# Apply 'scale()' to the 'digits' data
digits = datasets.load_digits()
data = scale(digits.data)
