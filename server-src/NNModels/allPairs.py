#!/usr/bin/python

from operator import attrgetter
import libfann
import re
import copy
import sys
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
			if (isNumber(firstBox.getFeature(feature)) and not feature.startswith("wordfreq")): # we don't want to add ranks and so on for wordfreqs
				rangeSet = set()
				for box in self.boxList:
					rangeSet.add(box.getFeature(feature))
				ranges[feature] = sorted(list(rangeSet))

		# then add a feature that gives the ranking
		for feature in firstBoxFeatures:
			if (isNumber(firstBox.getFeature(feature)) and not feature.startswith("wordfreq")):
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
		wordFreqs = {}
		for word in words:
			wordFreqs[word] = wordFreqs.get(word, 0) + 1
		for word in wordFreqs:
			self.addFeature("wordfreq-"+word, wordFreqs[word])
		self.addFeature("numuniquewords", len(wordFreqs.keys()))

	def setNumFeatureVector(self, numFeatureList):
		a = array.array('f')
		for feature in numFeatureList:
			if not self.hasFeature(feature):
                                if feature.startswith("wordfreq"): #special case because for that we want to just set the count to 0
                                        a.append(0)
                                else:
                                        print "Freak out!  One of our boxes doesn't have a numeric feature so we don't know what value to put in.  Feature:", feature
                                        exit(1)
			else:
				try:
					a.append(self.getFeature(feature))
				except:
					print feature
					print self.getFeature(feature)
					exit()
		self.numFeatureVector = a

	def wholeSingleBoxFeatureVector(self):
		vec = list(self.numFeatureVector)
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
		reader = csv.reader(csvfile, delimiter=",", escapechar='\\', quotechar="\"")

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
					if valType in ["font-family","font-style","font-weight","color","background-color","font_family", "column"]:
						# for now we don't have a good way of turning these into booleans or numeric features
						# todo: decide how to actually deal with categorical things like this
						continue
					targetDict = oVals
					if valType in specialElements:
						targetDict = sVals

					val = row[i]
					if (len(row)) != numColumns:
						raise Exception("Malformed dataset file.  Number of cells is not consistent across rows.")

					if valType not in ["text", "doc", "label"] and CSVHandling.canInterpretAsFloat(val):
						val = float(val)
					elif valType not in ["text", "doc", "label"]:
						# for now we need everything to be numbers, so...
						if val == "TRUE" or val == "True":
							val = 1
						elif val == "FALSE" or val == "False":
							val = -1
						else:
							val = 0

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
# Data structures for custom filter synthesis
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
			if self.accepts(row):
				numFilteredCounter += 1
		return numFilteredCounter

	def test(self, dataset):
		numDatasetRows = len(dataset)
		numFilteredCounter = 0
		numFilteredThatHaveLabel = 0
		for row in dataset:
			if self.accepts(row):
				numFilteredCounter += 1
				if row[0] != noLabelString:
					numFilteredThatHaveLabel += 1

		print "num rows in test set", numDatasetRows
		print "num rows in test set that are filtered", numFilteredCounter
		print "num rows in test set that are filtered but shouldn't be (have labels)", numFilteredThatHaveLabel

	def accepts(self, row):
		for filterComponent in self.filterComponentList:
			if filterComponent.accepts(row):
				return True # we're ORing, so return true if any accept
		return False

	# this is a weird filter, because the filtered things are the things we're throwing out -- remember, we made the filter for getting rid of "nolabel" items
	def filterDataset(self, dataset):
		outputDataset = []
		for row in dataset:
			if not self.accepts(row):
				outputDataset.append(row)
		return outputDataset

# **********************************************************************
# Custom filter synthesis
# **********************************************************************

def synthesizeFilter(dataset, numericalColIndexes):

	# first let's decide how many "nolabel" items we want to filter
	labelCount = 0
	nolabelCount = 0
	for row in dataset:
		if row[0] != noLabelString:
			labelCount += 1
		else:
			nolabelCount += 1
	targetNumFiltered = nolabelCount - (2 * labelCount) # there should be at most 2 nolabels per label in the output dataset
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
			if label != noLabelString:
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

	bestFilterSoFar = Filter([bestFilterSoFar])

	# let's try using more than one
	maxComponents = 3 # don't want to go beyond 3 for fear of overfitting
        # we'll try better combinations sooner if we first sort the list of possible filters
        # this is worth it since testing combinations on a large dataset is pretty expensive
        possibleFilters.sort(key=lambda x: x.numFiltered, reverse=True)
	for i in range(2, maxComponents + 1):
		filterCombos = itertools.combinations(possibleFilters, i)
		for filterCombo in filterCombos:
                        if filterCombo[0].numFiltered < bestFilterScore/i:
                                # recall that we sorted the list first, and combinations retains sorting: ABCD -> AB, AC, AD, BC, BD, CD
                                # so if we get a filter where the first component filters less than half of what we need, and only 2
                                # components are allowed, we know we can call off this search
                                break
                        # same idea here -- can't do better than the sum
                        sumFiltered = 0
                        for component in filterCombo:
                                sumFiltered += component.numFiltered
                        if sumFiltered < bestFilterScore:
                                continue

			f = Filter(list(filterCombo))
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
def datasetToRelation(docList, features):
	data = []

	firstRow = ["label", "docName"] + features

	data.append(firstRow)

	i = 0
	for doc in docList:
		i += 1
		for box in doc.boxList:
			box.setNumFeatureVector(features)
			row = [box.label, doc.name]
			featureVec = box.wholeSingleBoxFeatureVector()
			row = row + featureVec
			data.append(row)

	return data

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

	print "decided on a feature set with", len(popularFeatures), "features"
	return popularFeatures

class LabelHandler():
	labelsToLabelIds = {}
	labelIdsToLabels = []
	numLabels = 0
	
	def __init__(self, labelLs):
		self.labelIdsToLabels = labelLs
		for i in range(len(labelLs)):
			self.labelsToLabelIds[labelLs[i]] = i
		self.numLabels = len(labelLs)

	def getOneInNRepForLabel(self, label):
		labelVec = [-1]*self.numLabels
		labelVec[self.labelsToLabelIds[label]] = 1
		return labelVec

	def getXInNRepForLabels(self, labelLs):
		labelVec = [-1]*self.numLabels
		for label in labelLs:
			labelVec[self.labelsToLabelIds[label]] = 1
		return labelVec

	def closestLabel(self, labelVec):
		winningIndex = labelVec.index(max(labelVec))
		return self.labelAtIndex(winningIndex)

	def labelsFromNetAnswer(self, labelVec):
		indices = [i for i, x in enumerate(labelVec) if x > 0]
		labels = map(lambda x: self.labelAtIndex(x), indices)
                return labels

	def getLabelForOneInNRep(self, labelVec):
	  index = labelVec.index(1)
	  return self.labelAtIndex(index)

	def getLabelsForXInNRep(self, labelVec):
		indices = [i for i, x in enumerate(labelVec) if x == 1]
		labels = map(lambda x: self.labelAtIndex(x), indices)
                return labels

	def labelAtIndex(self, index):
		return self.labelIdsToLabels[index]

def getLabelsFromDataset(dataset):
	labelSet = set()
	for row in dataset:
		labelStr = row[0]
		labels = labelStr.split("|")
		for label in labels:
			labelSet.add(label)
	return list(labelSet)

def relationToDocuments(datasetRaw, labelHandler):
	documents = {}
	for row in datasetRaw:
		docName = row[1]
		docRows = documents.get(docName, [])
                try:
                        label = labelHandler.getXInNRepForLabels(row[0].split("|"))
                except:
                        print "Hey!  This is bad.  You saw a label that labelHandler doesn't have.", row[0]
                        continue
		docRows.append((row[2:], label))
		documents[docName] = docRows
	return documents

# convert the purely relational form to (input, output) pairs, convert output to vector form
def saveDatasetMemConscious(datasetRaw, trainingSetFilename, labelHandler):
	# we want to make an input vector from each pair of textboxes in each document
	documentsNums = relationToDocuments(datasetRaw, labelHandler)

        # doing all the string conversions for each pair is silly.  let's avoid that
        documents = {}
        for key in documentsNums:
                strPairs = []
                pairs = documentsNums[key]
                for pair in pairs:
                        inputStr = " ".join(map(lambda x: str(x), pair[0]))
                        outputStr = " ".join(map(lambda x: str(x), pair[1]))
                        strPairs.append((inputStr,outputStr))
                documents[key] = strPairs

        # let's get the data we need for starting the output file

        randomKey = documentsNums.keys()[0]
        inputSize = len(documentsNums[randomKey][0][0]) * 2 # times 2 because we'll be sticking together 2 diff boxes for each row
        outputSize = len(documentsNums[randomKey][0][1]) * 2

	# let's just estimate how large this will actually be...
	numDatapoints = 0
	for document in documents:
		numBoxes = len(documents[document])
		numDatapoints += numBoxes*(numBoxes - 1)
	print "expected number of datapoints in final dataset:", numDatapoints

        fileHandle = NNWrapper.startDatasetFile(numDatapoints, inputSize, outputSize, trainingSetFilename) 

	batchSize = 10000

	outputDataset = []
	docsSoFar = 0
        boxCounter = 0
        counter = 0
	for document in documents:
		boxes = documents[document]
		boxPairs = itertools.permutations(boxes, 2) # is there a nice way to do this with combinations instead of permuations?  or do we want the largest dataset we can get?  probably yes.
		for pair in boxPairs:
			boxCounter += 1
			outputDataset.append(([pair[0][0],pair[1][0]], [pair[0][1],pair[1][1]])) # (input, output)
			if boxCounter == batchSize:
				print "data points so far:", (counter + 1) * batchSize
                                NNWrapper.addToDatasetFile(fileHandle, outputDataset)
				counter += 1
				boxCounter = 0
				outputDataset = [] # clear it out, because this can get huge even for one document
		docsSoFar += 1
		print "documents so far (out of", len(documents.keys()), "):", docsSoFar
	# and let's flush out anything left in here
	NNWrapper.addToDatasetFile(fileHandle, outputDataset)
	print "last addition to dataset file.  total data points:", counter * batchSize + len(outputDataset)
        print "predicted num datapoints:", numDatapoints
        fileHandle.close()

        return inputSize, outputSize

def convertColumnToRange(dataset, colIndex, newMin, newMax):
	rangeAllowed = newMax - newMin
	values = map(lambda row: row[colIndex], dataset)
	oldMax = max(values)
	oldMin = min(values)
	oldRange = (oldMax - oldMin)
	if oldRange == 0:
		# print "new constant col after filtering", colIndex
		raise Exception("trying to convert a constant column to a range")
	for j in range(len(dataset)):
                dataset[j][colIndex] =  (float((dataset[j][colIndex] - oldMin) * rangeAllowed) / oldRange) + newMin
	return (oldMin, oldMax)

# the same as the normal convertColumnToRange, but uses fixed oldrange, so anything out of the target oldrange gets pushed into that range first
def convertColumnToRangeCutoff(dataset, colIndex, newMin, newMax, oldMin, oldMax):
	rangeAllowed = newMax - newMin
	oldRange = (oldMax - oldMin)
        for i in range(len(dataset)):
		if dataset[i][colIndex] > oldMax:
			dataset[i][colIndex] = oldMax
		elif dataset[i][colIndex] < oldMin:
			dataset[i][colIndex] = oldMin
                dataset[i][colIndex] =  (float((dataset[i][colIndex] - oldMin) * rangeAllowed) / oldRange) + newMin

def scaleRelation(datasetRaw, ranges=None):
	dataset = datasetRaw[1:]

	if ranges == None:
		ranges = []
		for i in range(2, len(dataset[0])): # start at 2 because we don't do this for labels or document names
			oldRange = convertColumnToRange(dataset, i, -1, 1)
			ranges.append(oldRange)
	else:
		for i in range(2, len(dataset[0])): # start at 2 because we don't do this for labels or document names
			currColRange = ranges[i-2]
			convertColumnToRangeCutoff(dataset, i, -1, 1, currColRange[0], currColRange[1])

	return dataset, ranges

def removeConstantColumns(dataset):
        constantIndexes = []
        for i in range(len(dataset[0])):
                firstVal = dataset[1][i] # first row is headers
                constantCol = True
                for j in range(2,len(dataset)):
                        if firstVal != dataset[j][i]:
                                constantCol = False
                                break
                if constantCol:
                        constantIndexes.append(i)

        return removeColumns(dataset, constantIndexes), constantIndexes

def removeColumns(dataset, indexes):
        sortedIndexes = sorted(indexes, reverse=True)
        #print sortedIndexes
        for row in dataset:
                for index in sortedIndexes:
                        try:
                                del row[index]
                        except Exception:
                                print "row len", len(row)
                                print "index to remove", index
                                print "last row len", len(dataset[-1])
                                raise Exception("gah")
        return dataset
                

# **********************************************************************
# NN and NN-related functionality
# **********************************************************************

class NNWrapper():
	connection_rate = 1
	learning_rate = 0.05
	iterations_between_reports = 1

	testingSummaryFilename = "testingSummary.csv"
	totalTested = 0
	totalCorrect = 0

	numThatActuallyHaveLabel = None
	numThatActuallyHaveLabelCorrectlyLabeled = None

	@staticmethod
	def saveDatasetToFile(datasetPairs, filename):
		numPairs = len(datasetPairs)
		inputSize = len(datasetPairs[0][0])
		outputSize = len(datasetPairs[0][1])
		fileStrs = [str(numPairs)+" "+str(inputSize)+" "+str(outputSize)] 
                for pair in datasetPairs:
			fileStrs.append(" ".join(map(lambda x: str(x), pair[0])))
			fileStrs.append(" ".join(map(lambda x: str(x), pair[1])))
		f = open(filename, "w")
                f.write("\n".join(fileStrs))
		f.close()

	@staticmethod
	def saveDatasetToFileStringLists(datasetPairs, inputSize, outputSize, filename):
		numPairs = len(datasetPairs)
		fileStrs = [str(numPairs)+" "+str(inputSize)+" "+str(outputSize)] 
                for pair in datasetPairs:
			fileStrs.append(" ".join(pair[0]))
			fileStrs.append(" ".join(pair[1]))
		f = open(filename, "w")
                f.write("\n".join(fileStrs))
		f.close()

	@staticmethod
	def startDatasetFile(numPairs, inputSize, outputSize, filename):
                f = open(filename, "w")
		f.write(str(numPairs)+" "+str(inputSize)+" "+str(outputSize)+"\n")
                return f

        @staticmethod
        def addToDatasetFile(fileHandle, datasetPairs):
                fileStrs = []
                for pair in datasetPairs:
			fileStrs.append(" ".join(pair[0]))
			fileStrs.append(" ".join(pair[1]))
                fileHandle.write("\n".join(fileStrs))

	# todo: actually make this ready
	@staticmethod
	def saveDatasetToFileAlreadyString(numPairs, inputSize, outputSize, string, filename):
		f = open(filename, "w")
		numPairs = len(datasetPairs)
		inputSize = len(datasetPairs[0][0])
		outputSize = len(datasetPairs[0][1])
		f.write(str(numPairs)+" "+str(inputSize)+" "+str(outputSize)+"\n")
		for pair in datasetPairs:
			f.write(" ".join(map(lambda x: str(x), pair[0]))+"\n")
			f.write(" ".join(map(lambda x: str(x), pair[1]))+"\n")
		f.close()

	@staticmethod
	def mergeSingleDocumentTrainingFiles(filenames, finalFilename):
		totalNumExamples = 0
		inputSize = 0
		outputSize = 0
		for filename in filenames:
			with open(filename, 'r') as f:
				firstLine = f.readline()
				items = firstLine.split(" ")
				numInputOutputPairs = int(items[0])
				totalNumExamples += numInputOutputPairs
				inputSize = int(items[1])
				outputSize = int(items[2])

		try:
			os.remove(finalFilename)
		except:
			print "already no such file"
		outputfile = open(finalFilename, "w")
		outputfile.write(str(totalNumExamples)+" "+str(inputSize)+" "+str(outputSize)+"\n")
		for filename in filenames:
                        print filename
			with open(filename) as f:
                                fileStr = f.read()
                                endFirstLine = fileStr.index("\n")
                                remainingStr = fileStr[endFirstLine+1:]
                                outputfile.write(remainingStr+"\n")
                        os.remove(filename) # can take so much space!
		outputfile.close()

	@staticmethod
	def trainNetwork(dataFilename, netFilename, layerSizes, max_iterations, desired_error):
		# layerSizes should look something like this: (numInput, 200, 80, 40, 20, 10, numOutput)
		ann = libfann.neural_net()
		#ann.create_sparse_array(NNWrapper.connection_rate, (numInput, 6, 4, numOutput)) #TODO: is this what we want? # the one that works in 40 seconds 4, 10, 6, 1.  the one that trained in 30 secs was 6,6
		ann.create_standard_array(layerSizes)
		ann.set_learning_rate(NNWrapper.learning_rate) # rprop doesn't use learning rate
		ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
                #ann.set_training_algorithm(libfann.TRAIN_RPROP)
		ann.set_training_algorithm(libfann.TRAIN_QUICKPROP)
                ann.set_bit_fail_limit(.3)
		#ann.randomize_weights(0,0)

		t0 = time.time()
		ann.train_on_file(dataFilename, max_iterations, NNWrapper.iterations_between_reports, desired_error)
		t1 = time.time()
		seconds = t1-t0
		print "Seconds: ", seconds

		m, s = divmod(seconds, 60)
		h, m = divmod(m, 60)
		print "Time to train:"
		print "%d:%02d:%02d" % (h, m, s)

		ann.save(netFilename)

	@staticmethod
	def testNet(datasetRaw, netFilename, labelHandler):
		if NNWrapper.numThatActuallyHaveLabel == None:
			NNWrapper.numThatActuallyHaveLabel = {}
			NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled = {}

		try:
			os.remove(testingSummaryFilename)
		except:
			print "already no such file"

		testingSummaryFile = open(NNWrapper.testingSummaryFilename, "a")

		ann = libfann.neural_net()
		ann.create_from_file(netFilename)
		#ann.print_connections()

		labelLen = labelHandler.numLabels

		stats = {}
		documents = relationToDocuments(datasetRaw, labelHandler)
		for document in documents:
                        print document
			boxes = documents[document]
			boxPairs = itertools.permutations(boxes, 2)

			for pair in boxPairs:

				featureVec = pair[0][0]+pair[1][0]

                                actualLabelsBox1 = labelHandler.getLabelsForXInNRep(pair[0][1])
                                actualLabelsBox2 = labelHandler.getLabelsForXInNRep(pair[1][1])

				result = ann.run(featureVec)

				testingSummaryFile.write(str(pair[0][1])+"\t;"+str(pair[1][1])+"\t;"+str(result))

				guessedLabelsBox1 = labelHandler.labelsFromNetAnswer(result[:labelLen])
				guessedLabelsBox2 = labelHandler.labelsFromNetAnswer(result[labelLen:])

				for actualLabelBox1 in actualLabelsBox1:
					box1Stats = stats.get(actualLabelBox1, {"left": {}, "right": {}})
					for guessedLabelBox1 in guessedLabelsBox1:
						box1Stats["left"][guessedLabelBox1] = box1Stats["left"].get(guessedLabelBox1, 0) + 1
					stats[actualLabelBox1] = box1Stats

				for actualLabelBox2 in actualLabelsBox2:
					box2Stats = stats.get(actualLabelBox2, {"left": {}, "right": {}})
					for guessedLabelBox2 in guessedLabelsBox2:
						box2Stats["right"][guessedLabelBox2] = box2Stats["right"].get(guessedLabelBox2, 0) + 1
					stats[actualLabelBox2] = box2Stats

		for key in stats:
			print key, "left"
			print "*******************"
			for label in labelHandler.labelIdsToLabels:
				count = stats[key]["left"].get(label, 0)
				print label, "\t\t\t", count
			print key, "right"
			print "*******************"
			for label in labelHandler.labelIdsToLabels:
				count = stats[key]["right"].get(label, 0)
				print label, "\t\t\t", count

# **********************************************************************
# High level structure
# **********************************************************************

def makeSingleNodeNumericFeatureVectors(filename, trainingsetFilename, netFilename, testOnly):

	docList = CSVHandling.csvToBoxlists(filename) # each boxList corresponds to a document

	trainingDocuments, testingDocuments = splitDocumentsIntoTrainingAndTestingSets(docList, .8) # go back to .8 once done testing

	# get everything we need to make feature vectors from both training and testing data
	popularFeatures = popularSingleBoxFeatures(trainingDocuments, .1) # go back to .07 once done testing
	trainingFeatureVectors = datasetToRelation(trainingDocuments, popularFeatures)
	testingFeatureVectors = datasetToRelation(testingDocuments, popularFeatures)

	# let's synthesize a filter
	numericalColIndexes = range(2, 2 + len(popularFeatures)) # recall first two rows are label and doc name.  todo: do this more cleanly in future
	noLabelFilter = synthesizeFilter(trainingFeatureVectors[1:], numericalColIndexes) # cut off that first row, since that's just the headings
	print noLabelFilter
	print noLabelFilter.stringWithHeadings(trainingFeatureVectors[0])
	noLabelFilter.test(testingFeatureVectors[1:])

	# now that we have a filter, we're ready to filter both the training set and the test set
	trainingFeatureVectors = [trainingFeatureVectors[0]] + noLabelFilter.filterDataset(trainingFeatureVectors[1:])
	testingFeatureVectors = [testingFeatureVectors[0]] + noLabelFilter.filterDataset(testingFeatureVectors[1:])

	print "len before", len(trainingFeatureVectors[0])
	trainingFeatureVectors, columnsToRemove = removeConstantColumns(trainingFeatureVectors)
	print "identified", len(columnsToRemove), "constant columns"
	print "len after", len(trainingFeatureVectors[0])

	print "len before", len(testingFeatureVectors[0])
	testingFeatureVectors = removeColumns(testingFeatureVectors, columnsToRemove)
	print "removed constant columns from test set"
	print "len after", len(testingFeatureVectors[0])

	# now we need to process the data for the NN -- scale everything to the [-1,1] range
	trainingFeatureVectors, ranges = scaleRelation(trainingFeatureVectors)
	testingFeatureVectors, ranges = scaleRelation(testingFeatureVectors, ranges)
	print "scaled"

        labelHandler = LabelHandler(getLabelsFromDataset(trainingFeatureVectors))
        
        if not testOnly:
                # now let's actually save the training set to file
                numInput, numOutput = saveDatasetMemConscious(trainingFeatureVectors, trainingsetFilename, labelHandler)
                print "saved data"

                # now that we've saved the datasets we need, let's actually run the NN on them
                desired_error = 0.015
                max_iterations = 500
                layerStructure = makeLayerStructure(numInput, numOutput, 20)
                NNWrapper.trainNetwork(trainingsetFilename, netFilename, layerStructure, max_iterations, desired_error)
        
        NNWrapper.testNet(testingFeatureVectors, netFilename, labelHandler)

#noLabelString = "nolabel"
noLabelString = "null"

def makeLayerStructure(numInput, numOutput, numHiddenLayers):
        start = numInput/3 # otherwise it can just get so big...
        end = numOutput*2
        denominator = ((start / end) ** (1.0 / (numHiddenLayers)))
        currLayerSize = start
        layerSizes = [numInput, start]
        for i in range(numHiddenLayers - 1):
                currLayerSize = currLayerSize / denominator
                layerSizes.append(int(math.ceil(currLayerSize)))
        layerSizes.append(numOutput)
        print layerSizes
        return layerSizes

def main():
        testOnly = False
	makeSingleNodeNumericFeatureVectors("cvDataset.csv", "trainingSetCV.data", "netCV.net", testOnly)	
	#makeSingleNodeNumericFeatureVectors("webDatasetFullCleaned.csv", "trainingSet.data", "net.net")	

main()

