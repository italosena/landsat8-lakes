from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import csv
from matplotlib import pyplot as plt

# Input file
train_file = 'data/train_1.csv'
test_file = 'data/test_1.csv'
params = ['B1','B2','B3','B4','B5','B6','B7']
subset = 'CDOM'

# Find
def find(lst, p):
    return min([i for i, x in enumerate(lst) if x==p])
    
# Parse CSV
# args: (1) path to dataset file, and (2) list of parameters of interest
# returns: (1) parameters 2d np.array, and (2) class, 1d np.array 
def parse_csv(filename, params, subset):
    with open(filename, 'rb') as f:
        csvreader = csv.reader(f)
        headers = csvreader.next()
        subset_lookup = find(headers, subset)
        param_lookup = {}
        for p in params:
            param_lookup[p] = find(headers, p)
        try:
            X = []
            y = []
            for row in csvreader:
                s = float(row[subset_lookup])
                y.append(s)
                sample = []
                for p in param_lookup:
                    i = param_lookup[p]
                    sample.append(float(row[i]))
                X.append(sample)
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))
        except Exception as e:
            print str(e)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Test an SVC
def test_svc(X1, y1, X2, y2, kernel='rbf'):
    """
    Uses the SVC class
    """
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X1, y1)
    err_lst = []
    for i in range(len(y2)):
        x = X2[i]
        y = round(y2[i])
        p = clf.predict(x)
        rmse = np.sqrt(mean_squared_error([y], [p]))
        err_lst.append(rmse)
    return err_lst

# Test an SVR
def test_svr(X1, y1, X2, y2, kernel='rbf'):
    """
    Uses the SVR class
    """
    clf = SVR(kernel='rbf', probability=True)
    clf.fit(X1, y1)
    err_lst = []
    for i in range(len(y2)):
        x = X2[i]
        y = round(y2[i])
        p = clf.predict(x)
        rmse = np.sqrt(mean_squared_error([y], [p]))
        err_lst.append(rmse)
    return err_lst
    
if __name__ == '__main__':
    (X1, y1) = parse_csv(train_file, params, subset)
    (X2, y2) = parse_csv(test_file, params, subset)
    c_res = test_svc(X1, y1, X2, y2)
    r_res = test_svr(X1, y1, X2, y2)
    print "x:%f, s:%f" % (np.mean(r_res), np.std(r_res))
    print "x:%f, s:%f" % (np.mean(c_res), np.std(c_res))
