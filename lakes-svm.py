from sklearn.svm import SVC
import numpy as np
import csv
from matplotlib import pyplot as plt

# Input file
train_file = 'data/train_1.csv'
test_file = 'data/test_1.csv'
headers = ['Name','CDOM','B1','B2','B3','B4','B5','B6','B7']

# Parse CSV
def parse_csv(filename):
    X = []
    Y = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                print row
                CDOM = round(float(row[1]))
                B1 = float(row[2])
                B2 = float(row[3])
                B3 = float(row[4])
                B4 = float(row[5])
                B5 = float(row[6])
                B6 = float(row[7])
                B7 = float(row[8])
                x = [B1, B2, B3, B4, B5, B6, B7]
                y = CDOM
                X.append(x)
                Y.append(y)
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))
    return (X,Y)
    
# Convert to NP
(X1, Y1) = parse_csv(train_file)
(X2, Y2) = parse_csv(test_file)

"""
SVC Class
    C=1.0
    cache_size=200
    class_weight=None
    coef0=0.0
    degree=3,
    gamma=0.0
    kernel='rbf'
    max_iter=-1
    probability=False,
    random_state=None
    shrinking=True
    tol=0.001
    verbose=False
"""

clf = SVC(kernel='rbf')
clf.fit(X1, Y1)

for i in range(len(Y2)):
    x = X2[i]
    y = Y2[i]
    p = clf.predict(x)
    print "Actual: %f\tPredicted: %f" % (y, p)
