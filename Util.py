from math import sin, cos, sqrt, atan2, radians

import pandas as pd

datasetFileName = "../dataset/generated_accidents_dataset.csv"

monthsArray = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december"]

numberOfScannedRows = 55824
numOfLastRowsForTest = 1000
testSizePercent = 0.1
R = 6373.0

def prepareColumnsLeft():
    columnListLeft = []

    columnListLeft.append("temperature")
    columnListLeft.append("barometer")
    columnListLeft.append("wind")
    columnListLeft.append("humidity")

    for x in range(0, 11):
        columnListLeft.append(monthsArray[x])

    for x in range(1, 31):
        columnListLeft.append(str(x) + "_day")

    for x in range(1, 4):
        columnListLeft.append(str(x) + "_quarter_of_day")

    columnListLeft.append("with_material_damage")
    columnListLeft.append("with_the_dead")
    columnListLeft.append("with_the_injured")
    columnListLeft.append("with_one_vehicle")
    columnListLeft.append("with_two_vehicles_without_turning")
    columnListLeft.append("with_two_vehicles_turning_or_crossing")
    columnListLeft.append("with_a_parked_vehicle")
    columnListLeft.append("with_walker")
    columnListLeft.append("undefined_accident_type")

    return columnListLeft

def prepareColumnsRight():
    columnListRight = []

    columnListRight.append("longitude")
    columnListRight.append("latitude")

    return columnListRight

def readValuesFromFileLeft():
    return pd.read_csv(datasetFileName, usecols=prepareColumnsLeft(), nrows=numberOfScannedRows)

def readValuesFromFileRight():
    return pd.read_csv(datasetFileName, usecols=prepareColumnsRight(), nrows=numberOfScannedRows)

def calculateDistanceBetweenCoordinates(lonP, latP, lonE, latE):
    lon1 = radians(lonP)
    lat1 = radians(latP)

    lon2 = radians(lonE)
    lat2 = radians(latE)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def calculateAverageDistance(predictionArray, expectedArray):
    sumOfDistances = 0
    numOfCoordinates = 0

    for x in range(0, expectedArray.__len__() - 1):
        distance = calculateDistanceBetweenCoordinates(predictionArray[x][0], predictionArray[x][1],
                                                       expectedArray[x][0],
                                                       expectedArray[x][1])

        sumOfDistances += distance
        numOfCoordinates += 1

    return sumOfDistances / numOfCoordinates