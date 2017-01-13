# Import
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale

digits = datasets.load_digits()
data = scale(digits.data)

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images,
                                                                               test_size=0.25, random_state=42)

n_samples, n_features = X_train.shape

print(n_samples)
print(n_features)
n_digits = len(np.unique(y_train))

clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
clf.fit(X_train)

#  Visualization code
fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

plt.show()
