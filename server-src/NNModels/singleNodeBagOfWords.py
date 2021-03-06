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
		wordFreqs = {}
		for word in words:
			wordFreqs[word] = wordFreqs.get(word, 0) + 1
		for word in wordFreqs:
			self.addFeature("wordfreq-"+word, wordFreqs[word])
		self.addFeature("numuniquewords", len(wordFreqs.keys()))

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
                                if feature.startswith("wordfreq"): #special case because for that we want to just set the count to 0
                                        a.append(0)
                                else:
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
				if row[0] != "nolabel":
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
	maxComponents = 3 # don't want to go beyond 3 for fear of overfitting
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
def datasetToRelation(docList, features):
	data = []

	firstRow = ["label", "docName"] + features

	data.append(firstRow)

	i = 0
	for doc in docList:
		i += 1
		for box in doc.boxList:
			box.setBoolFeatureVector([])
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

	print "decided on a feature set with", len(popularFeatures)
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
		labelVec = [0]*self.numLabels
		labelVec[self.labelsToLabelIds[label]] = 1
		return labelVec

	def closestLabel(self, labelVec):
		winningIndex = labelVec.index(max(labelVec))
		return self.labelAtIndex(winningIndex)

	def getLabelForOneInNRep(self, labelVec):
	  index = labelVec.index(1)
	  return self.labelAtIndex(index)

	def labelAtIndex(self, index):
		return self.labelIdsToLabels[index]

def getLabelsFromDataset(dataset):
	labelSet = set()
	for row in dataset:
		labelSet.add(row[0])
	return list(labelSet)

# convert the purely relational form to (input, output) pairs, convert output to vector form
def rowToInputOutputPairs(datasetRaw, labelHandler):
	outputDataset = []
	for row in datasetRaw:
		inp = row[2:]
		outp = labelHandler.getOneInNRepForLabel(row[0])
		outputDataset.append((inp, outp))
	return outputDataset

constantColumnsSoFar = 0

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

def relationsToNNPairs(datasetRaw, labelHandler, ranges=None):
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

	pairs = rowToInputOutputPairs(dataset, labelHandler)

	return pairs, ranges

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
	learning_rate = 0.5
	iterations_between_reports = 1

	testingSummaryFilename = "testingSummary.csv"
	totalTested = 0
	totalCorrect = 0

	numThatActuallyHaveLabel = None
	numThatActuallyHaveLabelCorrectlyLabeled = None

	@staticmethod
	def saveDatasetToFile(datasetPairs, filename):
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
	def trainNetwork(dataFilename, netFilename, layerSizes, max_iterations, desired_error):
		# layerSizes should look something like this: (numInput, 200, 80, 40, 20, 10, numOutput)
		ann = libfann.neural_net()
		#ann.create_sparse_array(NNWrapper.connection_rate, (numInput, 6, 4, numOutput)) #TODO: is this what we want? # the one that works in 40 seconds 4, 10, 6, 1.  the one that trained in 30 secs was 6,6
		ann.create_sparse_array(NNWrapper.connection_rate, layerSizes)
		ann.set_learning_rate(NNWrapper.learning_rate)
		ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
		ann.set_bit_fail_limit(.2)
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
	def testNet(testSet, netFilename, labelHandler):
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

		numTested = 0
		numLabeledCorrectly = 0
		for pair in testSet:
			featureVec = pair[0]
			actualLabel = labelHandler.getLabelForOneInNRep(pair[1])

			result = ann.run(featureVec)
			#print result, actualLabel
			numTested += 1
			NNWrapper.numThatActuallyHaveLabel[actualLabel] = NNWrapper.numThatActuallyHaveLabel.get(actualLabel, 0) + 1
			guessedLabel = labelHandler.closestLabel(result)
                        testingSummaryFile.write(guessedLabel+","+actualLabel+"\n")
			if actualLabel == guessedLabel:
				numLabeledCorrectly += 1
				NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled[actualLabel] = NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled.get(actualLabel, 0) + 1

		print "numTested", numTested
		print "numLabeledCorrectly", numLabeledCorrectly
		NNWrapper.totalTested += numTested
		NNWrapper.totalCorrect += numLabeledCorrectly
		print "totalTested", NNWrapper.totalTested
		print "totalCorrect", NNWrapper.totalCorrect
		print "percentageCorrect", float(NNWrapper.totalCorrect)/NNWrapper.totalTested
		print "*****************"
		for key in NNWrapper.numThatActuallyHaveLabel:
			print key, NNWrapper.numThatActuallyHaveLabel.get(key,0), NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled.get(key,0), float(NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled.get(key,0))/NNWrapper.numThatActuallyHaveLabel.get(key,0)
		testingSummaryFile.close()

# **********************************************************************
# High level structure
# **********************************************************************

def makeSingleNodeNumericFeatureVectors(filename, trainingsetFilename, testingsetFilename, netFilename):
	docList = CSVHandling.csvToBoxlists(filename) # each boxList corresponds to a document

	trainingDocuments, testingDocuments = splitDocumentsIntoTrainingAndTestingSets(docList, .8)

	# get everything we need to make feature vectors from both training and testing data
	popularFeatures = popularSingleBoxFeatures(trainingDocuments, .07) # this was .4 when ran the last one

	trainingFeatureVectors = datasetToRelation(trainingDocuments, popularFeatures)
	testingFeatureVectors = datasetToRelation(testingDocuments, popularFeatures)

	# let's synthesize a filter
	numericalColIndexes = range(2, 2 + len(popularFeatures)) # recall first two rows are label and doc name.  todo: do this more cleanly in future
	noLabelFilter = synthesizeFilter(trainingFeatureVectors[1:], numericalColIndexes) # cut off that first row, since that's just the headings
	print noLabelFilter
	print noLabelFilter.stringWithHeadings(trainingFeatureVectors[0])
	noLabelFilter.test(testingFeatureVectors[1:])

	# now that we have a filter, we're ready to filter both the training set and the test set
	trainingFeatureVectorsFiltered = [trainingFeatureVectors[0]] + noLabelFilter.filterDataset(trainingFeatureVectors[1:])
	testingFeatureVectorsFiltered = [testingFeatureVectors[0]] + noLabelFilter.filterDataset(testingFeatureVectors[1:])

        print "len before", len(trainingFeatureVectorsFiltered[0])
        trainingFeatureVectorsFiltered, columnsToRemove = removeConstantColumns(trainingFeatureVectorsFiltered)
        print "identified", len(columnsToRemove), "constant columns"
        print "len after", len(trainingFeatureVectorsFiltered[0])
        
        print "len before", len(testingFeatureVectorsFiltered[0])
        testingFeatureVectorsFiltered = removeColumns(testingFeatureVectorsFiltered, columnsToRemove)
        print "removed constant columns from test set"
        print "len after", len(testingFeatureVectorsFiltered[0])

	# now we need to process the data for the NN -- scale everything to the [-1,1] range, split the labels (the output) off from the feature vectors (the input)
	labelHandler = LabelHandler(getLabelsFromDataset(trainingFeatureVectorsFiltered))
	trainingPairs, ranges = relationsToNNPairs(trainingFeatureVectorsFiltered, labelHandler)
	testingPairs, ranges = relationsToNNPairs(testingFeatureVectorsFiltered, labelHandler, ranges)
        print "converted to pairs"

	# now let's actually save the training and test sets to files
	NNWrapper.saveDatasetToFile(trainingPairs, trainingsetFilename)
        print "saved data"

	# now that we've saved the datasets we need, let's actually run the NN on them
	desired_error = 0.01
	max_iterations = 5000
	numInput = len(trainingPairs[0][0])
	numOutput = len(trainingPairs[0][1])
	layerStructure = (numInput, 750, 500, 250, 100, 50, 30,  numOutput)
	NNWrapper.trainNetwork(trainingsetFilename, netFilename, layerStructure, max_iterations, desired_error)
	NNWrapper.testNet(testingPairs, netFilename, labelHandler)

def main():
	makeSingleNodeNumericFeatureVectors("webDatasetFullCleaned.csv", "trainingSet.NNFormat",  "testSet.NNFormat", "net.net")
main()

