from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import train_test_split

digits = datasets.load_digits()
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target,
                                                                               digits.images, test_size=0.25,
                                                                               random_state=42)

svc_model = svm.SVC(gamma=0.001, C=100, kernel='linear')
svc_model.fit(X_train, y_train)
