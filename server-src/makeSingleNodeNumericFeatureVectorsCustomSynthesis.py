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
# Custom filter synthesis
# **********************************************************************

class FilterComponent():
	def __init__(self, colIndex, lessEq, threshold, numFiltered):
		self.colIndex = colIndex
		self.lessEq = lessEq
		self.threshold = threshold
		self.numFiltered = numFiltered

	def __str__(self):
		op = ">="
		if self.lessEq:
			op = "<="
		return "row["+str(self.colIndex)+"] "+op+" "+str(self.threshold)

	def stringWithHeadings(self, headings):
		op = ">="
		if self.lessEq:
			op = "<="
		return "row["+headings[self.colIndex]+"] "+op+" "+str(self.threshold)

	def accepts(self, row):
		if self.lessEq:
			return row[self.colIndex] <= self.threshold
		else:
			return row[self.colIndex] >= self.threshold

class Filter():
	def __init__(self, filterComponentList):
		self.filterComponentList = filterComponentList

	def __str__(self):
		return " or ".join(map(str, self.filterComponentList))

	def stringWithHeadings(self, headings):
		return " or ".join(map(lambda x: x.stringWithHeadings(headings), self.filterComponentList))

	def numFiltered(self, dataset):
		numFilteredCounter = 0
		for row in dataset:
			for filterComponent in self.filterComponentList:
				if filterComponent.accepts(row):
					numFilteredCounter += 1
					break
		return numFilteredCounter

	def test(self, dataset):
		numDatasetRows = len(dataset)
		numFilteredCounter = 0
		numFilteredThatHaveLabel = 0
		for row in dataset:
			for filterComponent in self.filterComponentList:
				if filterComponent.accepts(row):
					numFilteredCounter += 1
					if row[0] != "nolabel":
						numFilteredThatHaveLabel += 1
					break

		print "num rows in test set", numDatasetRows
		print "num rows in test set that are filtered", numFilteredCounter
		print "num rows in test set that are filtered but shouldn't be (have labels)", numFilteredThatHaveLabel

def synthesizeFilter(dataset, numericalColIndexes):

	# first let's decide how many "nolabel" items we want to filter
	labelCount = 0
	nolabelCount = 0
	for row in dataset:
		if row[0] != "nolabel":
			labelCount += 1
		else:
			nolabelCount += 1
	targetNumFiltered = nolabelCount - (4 * labelCount) # there should be at most 4 nolabels per label in the output dataset
	print "number of rows in dataset:", len(dataset)
	print "number of rows with labels:", labelCount
	print "target number of rows to filter:", targetNumFiltered

	possibleFilters = []
	bestFilterSoFar = None
	bestFilterScore = 0
	for currColIndex in numericalColIndexes:

		# loop for finding the lowest col val associated with a label, highest col val associated with label
		lowestLabel = sys.maxint
		highestLabel = - sys.maxint
		for row in dataset:
			label = row[0]
			if label != "nolabel":
				currVal = row[currColIndex]
				if currVal < lowestLabel:
					lowestLabel = currVal
				elif currVal > highestLabel:
					highestLabel = currVal

		# loop for counting rows with col vals below lowestLabel, finding highest val below lowestLabel, counting rows with col vals above highestLabel, finding lowest val above highestLabel
		startNum = 0 # the number of nolabel values at the start of the sorted col, before the first labeled value
		endNum = 0 # the number of nolabel values at the end of the sorted col, after the last labeled value
		startThreshold = - sys.maxint # don't actually want to use the labeled val as the threshold.  better to be cautious, use the highest val associated with a nolabel
		endThreshold = sys.maxint # don't actually want to use the labeled val as the threshold.  better to be cautious, use the lowest val associated with a nolabel
		for row in dataset:
			currVal = row[currColIndex]
			if currVal < lowestLabel:
				startNum += 1
				if currVal > startThreshold:
					startThreshold = currVal
			elif currVal > highestLabel:
				endNum += 1
				if currVal < endThreshold:
					endThreshold = currVal

		if startNum > 0:
			newFilter = FilterComponent(currColIndex, True, startThreshold, startNum)
			possibleFilters.append(newFilter)
			if startNum > bestFilterScore:
				bestFilterSoFar = newFilter
				bestFilterScore = startNum

		if endNum > 0:
			newFilter = FilterComponent(currColIndex, False, endThreshold, endNum)
			possibleFilters.append(newFilter)
			if endNum > bestFilterScore:
				bestFilterSoFar = newFilter
				bestFilterScore = endNum

	print "best single filter score:", bestFilterScore

  # if a single filter is sufficient, let's go for that
	if bestFilterSoFar.numFiltered > targetNumFiltered:
		return Filter([bestFilterSoFar])

	# let's try using more than one
	maxComponents = 3
	for i in range(2, maxComponents + 1):
		filterCombos = itertools.combinations(possibleFilters, i)
		for filterCombo in filterCombos:
			f = Filter(filterCombo)
			numFiltered = f.numFiltered(dataset)
			if numFiltered > bestFilterScore:
				bestFilterScore = numFiltered
				bestFilterSoFar = f
		print "best filter with no more than", i, "components:", bestFilterScore
		if bestFilterScore > targetNumFiltered:
			return bestFilterSoFar
	return bestFilterSoFar


# **********************************************************************
# Helpers
# **********************************************************************

def splitDocumentsIntoTrainingAndTestingSets(docList, trainingPortion):
	numDocuments = len(docList)
	splitPoint = int(trainingPortion*numDocuments)
	trainingDocuments = docList[:splitPoint]
	testingDocuments = docList[splitPoint:]
	return trainingDocuments, testingDocuments

# converts a set of documents to feature vectors
def datasetToRelation(docList, boolFeatures, numFeatures):
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

	return data

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

def makeSingleNodeNumericFeatureVectors(filename, trainingsetFilename, testingsetFilename, rosetteFilename):
	docList = CSVHandling.csvToBoxlists(filename) # each boxList corresponds to a document

	trainingDocuments, testingDocuments = splitDocumentsIntoTrainingAndTestingSets(docList, .8)

	# get everything we need to make feature vectors from both training and testing data
	boolFeatures, numFeatures = popularSingleBoxFeatures(trainingDocuments, .4)

	trainingFeatureVectors = datasetToRelation(trainingDocuments, boolFeatures, numFeatures)
	testingFeatureVectors = datasetToRelation(testingDocuments, boolFeatures, numFeatures)

	# let's synthesize a filter
	numericalColIndexes = range(2 + len(boolFeatures), 2 + len(boolFeatures) + len(numFeatures)) # recall first two rows are label and doc name.  todo: do this more cleanly in future
	noLabelFilter = synthesizeFilter(trainingFeatureVectors[1:], numericalColIndexes) # cut off that first row, since that's just the headings
	print noLabelFilter
	print noLabelFilter.stringWithHeadings(trainingFeatureVectors[0])
	noLabelFilter.test(testingFeatureVectors[1:])

def main():
	makeSingleNodeNumericFeatureVectors("webDatasetFullCleaned.csv", "trainingSetSingeNodeFeatureVectors.csv",  "testSetSingeNodeFeatureVectors.csv", "synthesizeFilterGenerated.rkt")
main()

