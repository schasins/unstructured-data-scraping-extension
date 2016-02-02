#!/usr/bin/python

from operator import attrgetter
from fann2 import libfann
import re
import copy
import sys
from bitarray import bitarray
import array
import csv
import os
import itertools
import time
import random
import math


# **********************************************************************
# Data structure for documents, for dealing with single-node
# features that depend on other textboxes in the same document
# **********************************************************************

def values(list, attrName):
	return map(attrgetter(attrName),list)

def highest(list, attrName):
	attrVals = values(list, attrName)
	return max(attrVals)

def lowest(list, attrName):
	attrVals = values(list, attrName)
	return min(attrVals)

def isNumber(x):
	return (isinstance(x, (int, long, float, complex)) and not isinstance(x,bool))

class Document:
	def __init__(self, boxList, name):
		self.boxList = boxList
		self.name = name

	def addSingleNodeFeaturesOneDocument(self):
		# for numerical features, compare each value to the range of values in the document
		self.addSmallestLargestRanksForNumerical()

		# for some features, compare to the docHeight, docWidth
		self.addPercentagesForWidthAndHeightRelated()

	def addSmallestLargestRanksForNumerical(self):
		# for numerical features, compare each value to the range of values in the document
		# first collect the range of values in the document
		# TODO: the below relies on the assumption that all numeric values are shared by all boxes
		# if that ever changes, update this
		ranges = {}
		firstBox = self.boxList[0]
		firstBoxFeatures = firstBox.getFeatures()
		for feature in firstBoxFeatures:
			if (isNumber(firstBox.getFeature(feature))):
				rangeSet = set()
				for box in self.boxList:
					rangeSet.add(box.getFeature(feature))
				ranges[feature] = sorted(list(rangeSet))

		# then add a feature that gives the ranking
		for feature in firstBoxFeatures:
			if (isNumber(firstBox.getFeature(feature))):
				rangeLs = ranges[feature]
				rangeLsLen = len(rangeLs)
				for box in self.boxList:
					index = rangeLs.index(box.getFeature(feature))
					box.addFeature(feature+"-smallest-rank", index + 1)
					box.addFeature(feature+"-largest-rank", rangeLsLen - index)

	def addPercentagesForWidthAndHeightRelated(self):
		# first figure out some whole-document stuff
		docTop = lowest(self.boxList, "top")
		docLeft = lowest(self.boxList, "left")
		docHeight = highest(self.boxList, "bottom") - docTop
		docWidth = highest(self.boxList, "right") - docLeft

		# for some features, compare to the docHeight, docWidth
		for box in self.boxList:
			for feature in ["right", "left"]:
				box.addFeature(feature+"-relative", float(box.getFeature(feature))-docLeft)
			for feature in ["right-relative", "left-relative", "width"]:
				box.addFeature(feature+"-percent", float(box.getFeature(feature))/docWidth)

			for feature in ["top", "bottom"]:
				box.addFeature(feature+"-relative", float(box.getFeature(feature))-docTop)
			for feature in ["top-relative", "bottom-relative", "height"]:
				box.addFeature(feature+"-percent", float(box.getFeature(feature))/docHeight)

	def allBoxesFeatures(self):
		return reduce(lambda acc, box : acc.union(box.getFeatures()), self.boxList, set())


# **********************************************************************
# Data structure for textboxes, tracking single-node features
# **********************************************************************

class Box:
	def __init__(self, left, top, right, bottom, text, label, otherFeaturesDict, name="dontcare"):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom
		self.text = text
		self.label = label
		self.otherFeaturesDict = otherFeaturesDict
		self.features = {}
		self.boolFeatureVector = bitarray()
		self.numFeatureVector = array.array('f')
		self.name = name

		self.addFeatures()

	def addFeatures(self):
		for coord in ["left","top","right","bottom"]:
			self.addFeature(coord, attrgetter(coord)(self))
		self.addFeature("width", self.right-self.left)
		self.addFeature("height", self.bottom-self.top)

		self.addWordFeatures()

		for feature in self.otherFeaturesDict:
			self.addFeature(feature, self.otherFeaturesDict[feature])

	def __str__(self):
		return self.name

	def addFeature(self, featureName, value):
		self.features[featureName] = value

	def hasFeature(self, featureName):
		return featureName in self.features

	def getFeature(self, featureName):
		return self.features[featureName]

	def getFeatureSafe(self, featureName):
		if self.hasFeature(featureName):
			return self.getFeature(featureName)
		else:
			return 0

	def getFeatures(self):
		return self.features.keys()

	def addWordFeatures(self):
		wordsStr = self.text.strip().lower()
		words = re.split("[\s\.,\-\/\#\!\$%\^&\*\;\:\{\}=\-\_\`\~\(\)]*", wordsStr)
		numWords = len(words)
		self.addFeature("numwords", numWords)
		uniqueWords = set(words)
		numUniqueWords = len(uniqueWords)
		self.addFeature("numuniquewords", numUniqueWords)
		for word in uniqueWords:
			self.addFeature("hasword-"+word, True);

	def setBoolFeatureVector(self, booleanFeatureList):
		a = bitarray()
		for feature in booleanFeatureList:
			if self.hasFeature(feature):
				a.append(1)
			else:
				a.append(0)
		self.boolFeatureVector = a

	def setNumFeatureVector(self, numFeatureList):
		a = array.array('f')
		for feature in numFeatureList:
			if not self.hasFeature(feature):
				print "Freak out!  One of our boxes doesn't have a numeric feature so we don't know what value to put in.  Feature:", feature
				exit(1)
			else:
				a.append(self.getFeature(feature))
		self.numFeatureVector = a

	def wholeSingleBoxFeatureVector(self):
		vec = map(int, list(self.boolFeatureVector)) + list(self.numFeatureVector)
		return vec


# **********************************************************************
# CSV details
# **********************************************************************

class CSVHandling():
	@staticmethod
	def canInterpretAsFloat(s):
		try:
			float(s)
			return True
		except ValueError:
			return False

	@staticmethod
	def csvToBoxlists(csvname):
		csvfile = open(csvname, "rb")
		reader = csv.reader(csvfile, delimiter=",", quotechar="\"")

		documents = {}
		boxIdCounter = 0
		firstRow = True
		columnTitles = []
		numColumns = 0
		specialElements = ["doc", "left", "top", "right", "bottom", "text", "label"]

		for row in reader:
			if firstRow:
				firstRow = False
				columnTitles = row
				numColumns = len(columnTitles)
				for specialElement in specialElements:
					if specialElement not in columnTitles:
						print "Freak out!  One of the column titles we really need isn't present:", specialElement
			else:
				sVals = {}
				oVals = {}
				for i in range(numColumns):
					valType = columnTitles[i]
					if valType in ["font-family","font-style","font-weight","color","background-color"]:
						# for now we don't have a good way of turning these into booleans or numeric features
						# todo: decide how to actually deal with categorical things like this
						continue
					targetDict = oVals
					if valType in specialElements:
						targetDict = sVals
					val = row[i]
					if valType != "text" and CSVHandling.canInterpretAsFloat(val):
						val = float(val)
					targetDict[valType] = val

				if sVals["left"] < 0 or sVals["top"] < 0:
					# for now, filter out boxes that appear offscreen here.  might want to filter these earlier
					continue
				box = Box(sVals["left"], sVals["top"], sVals["right"], sVals["bottom"], sVals["text"], sVals["label"], oVals, str(boxIdCounter))

				boxIdCounter += 1

				boxList = documents.get(sVals["doc"], [])
				boxList.append(box)
				documents[sVals["doc"]] = boxList

		documentList = []
		for feature in documents:
			newDocument = Document(documents[feature], feature)
			newDocument.addSingleNodeFeaturesOneDocument() # there are some features that depend on the whole doc, so let's add those now
			documentList.append(newDocument)

		return documentList

# **********************************************************************
# Everything we need to do to make the dataset ready for Rosette
# **********************************************************************

def processDataForRosette(datasetRaw, ranges = None):
	if ranges == None:

		names = datasetRaw[0]

		# todo: later this should maybe be set by user, to make it adjustable
		# but we definitely need the bitwidth to be at least large enough to handle the number of columns - 1
		numColumns = len(names)
		bitwidth = int(math.ceil(math.log((numColumns - 1), 2)) + 1)
		# so Rosette's signed ints can handle from -2**(bitwidth - 1) through 2**(bitwdith - 1 ) - 1
		maxValueAllowed = (2**(bitwidth - 1)) - 1
		minValueAllowed = (2**(bitwidth - 1)) * -1
		rangeAllowed = maxValueAllowed - minValueAllowed


		types = ["numeric"]*len(names)
		oldMins = ["",""]
		oldMaxes = ["",""]
		newMins = [minValueAllowed] * len(datasetRaw[0])
		newMaxes = [maxValueAllowed] * len(datasetRaw[0])

		dataset = datasetRaw[1:]

		for i in range(2, len(dataset[0])): # start at 2 because we don't do this for labels or document names
			values = map(lambda row: row[i], dataset)
			oldMax = max(values)
			oldMin = min(values)
			oldMins.append(oldMin)
			oldMaxes.append(oldMax)
			oldRange = (oldMax - oldMin)
			if oldRange == 0:
				print "Freak out.  We should have already filtered out any columns that have the same value for all rows."
				exit(1)
			for j in range(len(dataset)):
				dataset[j][i] =  (float((dataset[j][i] - oldMin) * rangeAllowed) / oldRange) + minValueAllowed

		ranges = [oldMins, oldMaxes, newMins, newMaxes]
		return [[bitwidth], names, types] + ranges + dataset, ranges

	else:
		# we should be scaling not according to a bitwidth but according to given ranges

		names = datasetRaw[0]
                types = ["numeric"]*len(names)
		oldMins = ranges[0]
		oldMaxes = ranges[1]
		newMins = ranges[2]
		newMaxes = ranges[3]

		dataset = datasetRaw[1:]

		for i in range(2, len(dataset[0])): # start at 2 because we don't do this for labels or document names
			oldMax = oldMaxes[i]
			oldMin = oldMins[i]
			minValueAllowed = newMins[i]
			maxValueAllowed = newMaxes[i]
			oldRange = (oldMax - oldMin)
			rangeAllowed = maxValueAllowed - minValueAllowed
			for j in range(len(dataset)):
				val = dataset[j][i]
				if val > oldMax:
					val = oldMax # recall that the range (max and min) came from training data, and this is testing data, so may exceed previously observed range
				if val < oldMin:
					val = oldMin
				dataset[j][i] =  (float((val - oldMin) * rangeAllowed) / oldRange) + minValueAllowed

		ranges = [oldMins, oldMaxes, newMins, newMaxes]
		return [names, types] + ranges + dataset, ranges


# **********************************************************************
# Helpers
# **********************************************************************

def splitDocumentsIntoTrainingAndTestingSets(docList, trainingPortion):
	numDocuments = len(docList)
	splitPoint = int(trainingPortion*numDocuments)
	trainingDocuments = docList[:splitPoint]
	testingDocuments = docList[splitPoint:]
	return trainingDocuments, testingDocuments

# saves the feature vector dataset into filename
def saveDataset(docList, filename, boolFeatures, numFeatures, ranges = None):
	data = []

	firstRow = ["label", "docName"] + boolFeatures + numFeatures

	data.append(firstRow)

	i = 0
	for doc in docList:
		i += 1
		for box in doc.boxList:
			box.setBoolFeatureVector(boolFeatures)
			box.setNumFeatureVector(numFeatures)
			row = [box.label, doc.name]
			featureVec = box.wholeSingleBoxFeatureVector()
			row = row + featureVec
			data.append(row)

	processedData, ranges = processDataForRosette(data, ranges)

	outputFile = open(filename, "w")
	for row in processedData:
		outputFile.write(",".join(map(lambda cell: str(cell), row))+"\n")

	return ranges

def divideIntoBooleanAndNumericFeatures(features, box):
	# right now this puts things like font-weight into bool features
	# TODO: see about getting rid of this
	# should really just make each feature with its type
	boolFeatures = []
	numFeatures = []
	for feature in features:
		if not box.hasFeature(feature):
			boolFeatures.append(feature)
		elif isNumber(box.getFeature(feature)):
			numFeatures.append(feature)
		else:
			boolFeatures.append(feature)
	return boolFeatures, numFeatures

def popularSingleBoxFeatures(docList, targetPercentDocuments):
	# first go through each document and figure out the single-node features for the document
	featureLists = []
	for doc in docList:
		featureLists.append(doc.allBoxesFeatures())

	# decide on the filtered set of single-node features that is interesting to us, based on how many
	# different document use each single-node feature
	featureScores = {}
	for featureList in featureLists:
		for feature in featureList:
			featureScores[feature] = featureScores.get(feature, 0) + 1

	numberOfDocumentsThreshold = int(len(docList)*targetPercentDocuments)
	popularFeatures = [k for k, v in featureScores.items() if v >= numberOfDocumentsThreshold]
	boolFeatures, numFeatures = divideIntoBooleanAndNumericFeatures(popularFeatures, docList[0].boxList[0])

	print "decided on a feature set with", len(boolFeatures), "bool features and", len(numFeatures), "numerical features"

	# now let's filter out any features that produce only a single value on the training set
	featuresToKeep = []
	for feature in popularFeatures:
		firstVal = docList[0].boxList[0].getFeatureSafe(feature)
		keepFeature = False
		for doc in docList:
			for box in doc.boxList:
				currVal = box.getFeatureSafe(feature) # todo: this is the other problem with the font-weight.  this doesn't filter it out because not treated exactly as bool
				if currVal != firstVal:
					keepFeature = True
					break
			if keepFeature:
				break
		if keepFeature:
			featuresToKeep.append(feature)

	boolFeatures, numFeatures = divideIntoBooleanAndNumericFeatures(featuresToKeep, docList[0].boxList[0])

	print "removed constant features.  decided on a feature set with", len(boolFeatures), "bool features and", len(numFeatures), "numerical features"

	return boolFeatures, numFeatures

# **********************************************************************
# High level structure
# **********************************************************************

def makeSingleNodeNumericFeatureVectors(filename, trainingsetFilename, testingsetFilename):
	docList = CSVHandling.csvToBoxlists(filename) # each boxList corresponds to a document

	trainingDocuments, testingDocuments = splitDocumentsIntoTrainingAndTestingSets(docList, .8)

	# get everything we need to make feature vectors from both training and testing data
	boolFeatures, numFeatures = popularSingleBoxFeatures(trainingDocuments, .4)

	# todo: let's also remove columns where val is always the same
	# todo: we have to scale the values in each column, prepare for rosette

	ranges = saveDataset(trainingDocuments, trainingsetFilename, boolFeatures, numFeatures)
	saveDataset(testingDocuments, testingsetFilename, boolFeatures, numFeatures, ranges)

def main():
	makeSingleNodeNumericFeatureVectors("webDatasetFullCleaned.csv", "trainingSetSingeNodeFeatureVectors.csv",  "testSetSingeNodeFeatureVectors.csv")
main()

