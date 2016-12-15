'''

Dinesh Singh

dinesh.singh@nyu.edu

k-Means Unsupervised Learning Algorithm

The K-Means algorithm works by first partitioning the objects into "k" subsets. The algorithm then finds seed points at the centroids (mean point) of each cluster in the current subset.
Each object is then assigned to the nearest cluster with the nearest seed point. This process is
repeated for each subset. In my implementation of K-Means, I created Point and Cluster classes,
that were used to track each cluster and its points. I used Euclidean distance as a heuristic, and also
had a function for file I/O. The cutoff is .5 for K-Means, and obviously, k = 2 clusters were used for this dataset
(to represent Republicans and Democrats).

'''

import math
import random
import csv


# Calculates the Euclidean distance
def euclideanDistance(a, b):
    distance = 0
    for x in range(a.n):
        distance += pow((a.coords[x] - b.coords[x]), 2)
    return math.sqrt(distance)


# Loads the data from a file into a dataset
def loadDataset(filename):
    dataSet = list()
    with open(filename, 'r') as csvfile:
        next(csvfile)
        lines = csv.reader(csvfile)
        lineSet = list(lines)
        for x in range(len(lineSet)):
            for y in range(len(lineSet[x])):
                lineSet[x][y] = float(lineSet[x][y])
            dataSet.append(lineSet[x])
    return dataSet


# Class representing each data point
class Point:
    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)


# Class representing each cluster of data points
class Cluster:
    def __init__(self, points):
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        self.points = points
        self.n = points[0].n
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        return str(self.points)

    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = euclideanDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        numPoints = len(self.points)
        coords = [p.coords for p in self.points]
        unzipped = zip(*coords)
        centroid_coords = [math.fsum(dList) / numPoints for dList in unzipped]
        return Point(centroid_coords)


# Definition of the main k-Means algorithm
def kMeans(points, numClusters, cutoff):
    init = random.sample(points, numClusters)
    clusters = [Cluster([p]) for p in init]
    numLoops = 0

    while True:
        lists = [[] for c in clusters]
        numClusters = len(clusters)

        numLoops += 1
        for p in points:
            smallestDist = euclideanDistance(p, clusters[0].centroid)
            clusterIndex = 0

            for i in range(1, numClusters):
                distance = euclideanDistance(p, clusters[i].centroid)

                if distance < smallestDist:
                    smallestDist = distance
                    clusterIndex = i
            lists[clusterIndex].append(p)

        largestShift = 0.0

        for i in range(numClusters):
            shift = clusters[i].update(lists[i])
            largestShift = max(largestShift, shift)

        if largestShift < cutoff:
            print("Converged after %s iterations" % numLoops)
            break
    return clusters


def main():

    numClusters = 2
    cutoff = 0.5

    testSet = loadDataset('votes-test.csv')
    points = list()
    for row in range(len(testSet)):
        p = Point(testSet[row])
        points.append(p)

    clusters = kMeans(points, numClusters, cutoff)

    for i, c in enumerate(clusters):
        for p in c.points:
            print(" Cluster: ", i, "\t Point :", p)


main()