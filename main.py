# Import 'datasets' from 'sklearn'
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA, RandomizedPCA
import matplotlib.pyplot as plt

# Load in the 'digits' data
digits = datasets.load_digits()

# Print the 'digits' data
print(digits)

# Print the keys
print(digits.keys)

# Print out the data
print(digits.data)

# Print out the target
print(digits.target)

# Print out the description of 'digits' data
print(digits.DESCR)

# Get the digits data
digits_data = digits.data

# Print the digits data
print(digits_data.shape)

# Get the target digits
digits_target = digits.target

# Print target data shape
print(digits_target.shape)

# Get the number of unique labels
number_digits = len(np.unique(digits_target))

# Print unique values
print(number_digits)

# Isolate the 'images'
digits_images = digits.images

# Inspect the shape
print(digits_images.shape)

# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

# Show the plot
plt.show()

# Create a Randomized PCA model that takes two components
randomized_pca = RandomizedPCA(n_components=2)

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Create a regular PCA model
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Inspect the shape
print(reduced_data_pca.shape)

# Print out the data
print(reduced_data_rpca)
print(reduced_data_pca)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][digits.target == i]
    y = reduced_data_rpca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()
