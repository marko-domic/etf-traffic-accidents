from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from Util import readValuesFromFileLeft, readValuesFromFileRight, numOfLastRowsForTest, testSizePercent, \
    calculateAverageDistance

import numpy as np

print('Data import.')

# Import data from dataset
importedPre = readValuesFromFileLeft()
importedPost = readValuesFromFileRight()

expectedPre = importedPre[-numOfLastRowsForTest:]
expectedPost = importedPost[-numOfLastRowsForTest:]

importedPre = importedPre[:-numOfLastRowsForTest]
importedPost = importedPost[:-numOfLastRowsForTest]
print('Data successfully imported.')

X_train, X_test, y_train, y_test = train_test_split(np.array(importedPre), np.array(importedPost),
                                                                     test_size=testSizePercent)

print("-----------------------")
print("Random Forest Regressor")

start = timer()

clf = RandomForestRegressor()
clf.fit(X_train, y_train)
prediction = clf.predict(np.array(expectedPre))

end = timer()

print('{:16s}: {:5.3f}{:2s}'.format("AVERAGE_DISTANCE", calculateAverageDistance(prediction, np.array(expectedPost)),
                                    "km"))
print('{:13s}: {:5.3f}{:1s}'.format("Finished in: ", end - start, "s"))
