#!/usr/bin/python

from operator import attrgetter
import libfann
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
		self.relationships = {}
		self.toOneRelationships = {}
		self.boolFeatureVector = bitarray()
		self.numFeatureVector = array.array('f')
		self.name = name

	def __str__(self):
		return self.name

	def addFeature(self, featureName, value):
		self.features[featureName] = value

	def hasFeature(self, featureName):
		return featureName in self.features

	def getFeature(self, featureName):
		return self.features[featureName]

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

	def addFeatures(self):
		for coord in ["left","top","right","bottom"]:
			self.addFeature(coord, attrgetter(coord)(self))
		self.addFeature("width", self.right-self.left)
		self.addFeature("height", self.bottom-self.top)

		self.addWordFeatures()

		for feature in self.otherFeaturesDict:
			self.addFeature(feature, self.otherFeaturesDict[feature])

	def setBoolFeatureVector(self, booleanFeatureList):
		a = bitarray()
		for feature in booleanFeatureList:
			if self.hasFeature(feature):
				a.append(1)
			else:
				a.append(0)
		self.boolFeatureVector = a

	def setNumFeatureVector(self, numFeatureList, numFeaturesRanges):
		a = array.array('f')
		for feature in numFeatureList:
			if not self.hasFeature(feature):
				print "Freak out!  One of our boxes doesn't have a numeric feature so we don't know what value to put in.  Feature:", feature
				exit(1)
			else:
				minMax = numFeaturesRanges[feature]
				oldval = self.getFeature(feature)

				oldrange = (minMax[1] - minMax[0]) # we've thrown out features that don't vary, so will never be 0
				newrange = float(1) # [-.5, 1]
				newval = (((oldval - minMax[0]) * newrange) / oldrange) #TODO: is this what we want?

				a.append(newval)
		self.numFeatureVector = a

	def wholeSingleBoxFeatureVector(self):
		return Box.wholeFeatureVectorFromComponents(self.boolFeatureVector, self.numFeatureVector)

	@staticmethod
	def wholeFeatureVectorFromComponents(boolVec, numVec):
		return map(int, list(boolVec)) + list(numVec)

	def above(self, b2):
		return self.bottom <= b2.top

	def leftOf(self, b2):
		return self.right <= b2.left

	def smallestGap(self, b2):
		gaps = []

		if self.above(b2):
			gaps.append(b2.top - self.bottom)
		elif b2.above(self):
			gaps.append(self.top - b2.bottom)
		else:
			# they overlap, so the gap is 0
			gaps.append(0)

		if self.leftOf(b2):
			gaps.append(b2.left - self.right)
		elif b2.leftOf(self):
			gaps.append(self.left - b2.right)
		else:
			# they overlap, so the gap is 0
			gaps.append(0)

		return min(gaps)

class NNWrapper():
	connection_rate = 1
	learning_rate = 0.5
	num_hidden = 30

	desired_error = 0.005 # TODO: is this what we want?
	max_iterations = 300
	iterations_between_reports = 1

	testingSummaryFilename = "testingSummaryOldTraining.txt"
	totalTested = 0
	totalCorrect = 0

	numThatActuallyHaveLabel = None
	numThatActuallyHaveLabelCorrectlyLabeled = None

	@staticmethod
	def clearNNLogging():
		try:
			os.remove(testingSummaryFilename)
		except:
			print "already no such file"

	@staticmethod
	def saveTrainingSetToFile(trainingSet, filename):
		f = open(filename, "w")
		numPairs = len(trainingSet)
		inputSize = len(trainingSet[0][0])
		outputSize = len(trainingSet[0][1])
		f.write(str(numPairs)+" "+str(inputSize)+" "+str(outputSize)+"\n")
		for pair in trainingSet:
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
			with open(filename) as f:
				next(f)
				for line in f:
					outputfile.write(line)
		outputfile.close()

	@staticmethod
	def trainNetwork(dataFilename, netFilename, numInput, numOutput):
		ann = libfann.neural_net()
		#ann.create_sparse_array(NNWrapper.connection_rate, (numInput, 6, 4, numOutput)) #TODO: is this what we want? # the one that works in 40 seconds 4, 10, 6, 1.  the one that trained in 30 secs was 6,6
		ann.create_sparse_array(NNWrapper.connection_rate, (numInput, 200, 80, 40, 20, 10, numOutput))
		ann.set_learning_rate(NNWrapper.learning_rate)
		ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
		ann.set_bit_fail_limit(.2)
		#ann.randomize_weights(0,0)

		t0 = time.clock()
		ann.train_on_file(dataFilename, NNWrapper.max_iterations, NNWrapper.iterations_between_reports, NNWrapper.desired_error)
		t1 = time.clock()
		seconds = t1-t0

		m, s = divmod(seconds, 60)
		h, m = divmod(m, 60)
		print "Time to train:"
		print "%d:%02d:%02d" % (h, m, s)

		ann.save(netFilename)

	@staticmethod
	def testNet(testSet, netFilename, labelHandler):
		if NNWrapper.numThatActuallyHaveLabel == None:
			NNWrapper.numThatActuallyHaveLabel = [0]*len(labelHandler.labelIdsToLabels)
			NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled = [0]*len(labelHandler.labelIdsToLabels)

		testingSummaryFile = open(NNWrapper.testingSummaryFilename, "a")

		ann = libfann.neural_net()
		ann.create_from_file(netFilename)
		#ann.print_connections()

		numTested = 0
		numLabeledCorrectly = 0
		for pair in testSet:
			featureVec = pair[0]
			actualLabelId = pair[1].index(1)
			actualLabel = labelHandler.labelIdsToLabels[actualLabelId]

			result = ann.run(featureVec)
			#print result, actualLabel
			numTested += 1
			winningIndex = result.index(max(result))
			NNWrapper.numThatActuallyHaveLabel[actualLabelId] += 1
			#testingSummaryFile.write(labelHandler.labelIdsToLabels[winningIndex]+","+actualLabel+"\n")
                        testingSummaryFile.write(actualLabel + ";" +  str(actualLabelId) + ";" + str(result) + "\n")
			if winningIndex == actualLabelId:
				numLabeledCorrectly += 1
				NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled[actualLabelId] += 1

		print "numTested", numTested
		print "numLabeledCorrectly", numLabeledCorrectly
		NNWrapper.totalTested += numTested
		NNWrapper.totalCorrect += numLabeledCorrectly
		print "totalTested", NNWrapper.totalTested
		print "totalCorrect", NNWrapper.totalCorrect
		print "percentageCorrect", float(NNWrapper.totalCorrect)/NNWrapper.totalTested
		print "*****************"
		for i in range(len(NNWrapper.numThatActuallyHaveLabel)):
			print labelHandler.labelIdsToLabels[i], NNWrapper.numThatActuallyHaveLabel[i], NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled[i], float(NNWrapper.numThatActuallyHaveLabelCorrectlyLabeled[i])/NNWrapper.numThatActuallyHaveLabel[i]
		testingSummaryFile.close()

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

		return documents.values()

class LabelHandler():

	labelsToLabelIds = {}
	labelIdsToLabels = []
	numLabels = 0
	
	def __init__(self, labelLs):
		self.labelIdsToLabels = labelLs
		for i in range(len(labelLs)):
			self.labelsToLabelIds[labelLs[i]] = i
		self.numLabels = len(labelLs)

	def labelToOneInNRep(self, label):
		labelVec = [0]*self.numLabels
		labelVec[self.labelsToLabelIds[label]] = 1
		return labelVec

def values(list, attrName):
	return map(attrgetter(attrName),list)

def highest(list, attrName):
	attrVals = values(list, attrName)
	return max(attrVals)

def lowest(list, attrName):
	attrVals = values(list, attrName)
	return min(attrVals)

def addSmallestLargestRanksForNumerical(boxList):
	# for numerical features, compare each value to the range of values in the document
	# first collect the range of values in the document
	# TODO: the below relies on the assumption that all numeric values are shared by all boxes
	# if that ever changes, update this
	ranges = {}
	firstBox = boxList[0]
	firstBoxFeatures = firstBox.getFeatures()
	for feature in firstBoxFeatures:
		if (isNumber(firstBox.getFeature(feature))):
			rangeSet = set()
			for box in boxList:
				rangeSet.add(box.getFeature(feature))
			ranges[feature] = sorted(list(rangeSet))

	# then add a feature that gives the ranking
	for feature in firstBoxFeatures:
		if (isNumber(firstBox.getFeature(feature))):
			rangeLs = ranges[feature]
			rangeLsLen = len(rangeLs)
			for box in boxList:
				index = rangeLs.index(box.getFeature(feature))
				box.addFeature(feature+"-smallest-rank", index + 1)
				box.addFeature(feature+"-largest-rank", rangeLsLen - index)

def addPercentagesForWidthAndHeightRelated(boxList):
	# first figure out some whole-document stuff
	docTop = lowest(boxList, "top")
	docLeft = lowest(boxList, "left")
	docHeight = highest(boxList, "bottom") - docTop
	docWidth = highest(boxList, "right") - docLeft

	# for some features, compare to the docHeight, docWidth
	for box in boxList:
		for feature in ["right", "left"]:
			box.addFeature(feature+"-relative", float(box.getFeature(feature))-docLeft)
		for feature in ["right-relative", "left-relative", "width"]:
			box.addFeature(feature+"-percent", float(box.getFeature(feature))/docWidth)

		for feature in ["top", "bottom"]:
			box.addFeature(feature+"-relative", float(box.getFeature(feature))-docTop)
		for feature in ["top-relative", "bottom-relative", "height"]:
			box.addFeature(feature+"-percent", float(box.getFeature(feature))/docHeight)

def allBoxesFeatures(boxList):
	return reduce(lambda acc, box : acc.union(box.getFeatures()), boxList, set())

def isNumber(x):
	return (isinstance(x, (int, long, float, complex)) and not isinstance(x,bool))

def findRelationships(boxList):
	for i in range(len(boxList)):
		findOneBoxRelationships(i, boxList)

def closest(b1, boxList):
	minGapSoFar = sys.maxint
	closestBoxSoFar = None
	for b2 in boxList:
		gap = b1.smallestGap(b2)
		if gap < minGapSoFar:
			closestBoxSoFar = b2
	return closestBoxSoFar

# TODO: for now have a very particular definition of above -- anywhere above, overlap in x direction
# might eventually want to try just the thing that's closest above
# or them not having to overlap
# should experiment with this
relationshipTypes = {}
relationshipTypes["above"] = lambda b1, b2 : b1.above(b2) and not b1.leftOf(b2) and not b2.leftOf(b1)
relationshipTypes["leftOf"] = lambda b1, b2 : b1.leftOf(b2) and not b1.above(b2) and not b2.above(b1)
relationshipTypes["below"] = lambda b1, b2 : b2.above(b1) and not b1.leftOf(b2) and not b2.leftOf(b1)
relationshipTypes["rightOf"] = lambda b1, b2 : b2.leftOf(b1) and not b1.above(b2) and not b2.above(b1)
relationshipTypes["sameleft"] = lambda b1, b2 : b1.left == b2.left
relationshipTypes["sametop"] = lambda b1, b2 : b1.top == b2.top
relationshipTypes["sameright"] = lambda b1, b2 : b1.right == b2.right
relationshipTypes["samebottom"] = lambda b1, b2 : b1.bottom == b2.bottom

#for now the only to-one relationship types are the ones that grab the closest from multinode relationships
toOneRelationshipTypes = []
for relationshipType in relationshipTypes:
	toOneRelationshipTypes.append(relationshipType+"-closest")

# the number of times we want to follow relationship pointers to build feature vectors
retlationshipDepth = 1

class RelationshipType:
	def __init__(self, name, isSingle):
		self.name = name
		self.isSingle = isSingle

allRelationshipTypes = []
for relationshipType in relationshipTypes:
	allRelationshipTypes.append(RelationshipType(relationshipType, False))
for relationshipType in toOneRelationshipTypes:
	allRelationshipTypes.append(RelationshipType(relationshipType, True))

def findOneBoxRelationships(index, boxList):
	box = boxList[index]
	relationships = {}
	for relationshipType in relationshipTypes:
		relationships[relationshipType] = []

	for i in range(len(boxList)):
		if (i == index):
			continue
		box2 = boxList[i]
		for relationshipType in relationshipTypes:
			if relationshipTypes[relationshipType](box, box2):
				relationships[relationshipType].append(box2)

	box.relationships = relationships
	
	toOneRelationships = {}
	for relationshipType in relationshipTypes:
		relationshipBoxes = relationships[relationshipType]
		if len(relationshipBoxes) > 0:
			toOneRelationships[relationshipType+"-closest"] = closest(box, relationshipBoxes)

	box.toOneRelationships = toOneRelationships

def getSingleNodeFeaturesOneDocument(boxList):
	for box in boxList:
		# get the individual features for each box
		box.addFeatures()

	# for numerical features, compare each value to the range of values in the document
	addSmallestLargestRanksForNumerical(boxList)

	# for some features, compare to the docHeight, docWidth
	addPercentagesForWidthAndHeightRelated(boxList)

	# get all the features from all the boxes
	documentFeatures = allBoxesFeatures(boxList)
	return documentFeatures

def divideIntoBooleanAndNumericFeatures(features, box):
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

def makeFeatureVectorWithRelationshipsToDepth(boxes, depth, relationshipsOnThisBranchSoFar, defaultBoolFeaturesVector, defaultNumFeaturesVector, maxBoxes):
	if depth == 0:
		return []

	featureVector = []
	for relationshipType in allRelationshipTypes:
		allChildBoxesForRelationshipType = []
		for box in boxes:
			childBoxes = []
			if relationshipType.isSingle:
				if relationshipType.name in box.toOneRelationships:
					childBoxes = [box.toOneRelationships[relationshipType.name]]
			else:
				childBoxes = box.relationships[relationshipType.name]
			allChildBoxesForRelationshipType += childBoxes

		newrelationshipsOnThisBranchSoFar = relationshipsOnThisBranchSoFar + [relationshipType]

		# let's save this internal node's set of features since we don't only want the leaves (want above as well as above-leftOf)
		# let's figure out whether we should use all features (if only followed single-box relationships)
		# or just boolean features (if also followed multi-box relationships)
		useAllFeatures = reduce(lambda x, y : x and y.isSingle, newrelationshipsOnThisBranchSoFar, True) # all relationshipTypes in the list must be single

		if useAllFeatures:
			# there should be either 0 or 1 nodes at the end of a single node relationship chain
			boxesLength = len(allChildBoxesForRelationshipType)
			if boxesLength == 0:
				featureVectorComponent = Box.wholeFeatureVectorFromComponents(defaultBoolFeaturesVector, defaultNumFeaturesVector)
			elif boxesLength == 1:
				featureVectorComponent = allChildBoxesForRelationshipType[0].wholeSingleBoxFeatureVector()
			else:
				print "Freak out!  We followed only single-box relationships but got multiple boxes."
		else:
			bitvector = reduce(lambda x, y: x | y.boolFeatureVector, allChildBoxesForRelationshipType, defaultBoolFeaturesVector)
			featureVectorComponent = map(int, list(bitvector))
			#since this one has multiple boxes, let's also add a feature for the number of boxes with this relationship type
			featureVectorComponent.append(float(len(allChildBoxesForRelationshipType))/maxBoxes)

		featureVector += featureVectorComponent
		featureVectorAddition = makeFeatureVectorWithRelationshipsToDepth(allChildBoxesForRelationshipType, depth-1, newrelationshipsOnThisBranchSoFar, defaultBoolFeaturesVector, defaultNumFeaturesVector, maxBoxes)
		featureVector += featureVectorAddition

	return featureVector

def setSingleBoxFeatures(boxList, boolFeatures, numFeatures, numFeaturesRanges):
	for box in boxList:
		box.setBoolFeatureVector(boolFeatures)
		box.setNumFeatureVector(numFeatures, numFeaturesRanges)

def makeInputOutputPairsFromInputOutput(input, output):
	return [input, output]

def makeFeatureVectors(boxList, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, labelHandler, isLabeled):
	setSingleBoxFeatures(boxList, boolFeatures, numFeatures, numFeaturesRanges)
	"""
	for box in boxList:
		if box.numFeatureVector == None:
			print "None"
			print box
	"""

	defaultBoolFeaturesVector = bitarray("0"*len(boolFeatures))
	defaultNumFeaturesVector = [-1]*len(numFeatures)

	vectors = []
	for box in boxList:
		currBoxFeatures = box.wholeSingleBoxFeatureVector()
		featureVector = currBoxFeatures + makeFeatureVectorWithRelationshipsToDepth([box], retlationshipDepth, [], defaultBoolFeaturesVector, defaultNumFeaturesVector, maxBoxes)
		if isLabeled:
			featureVector = makeInputOutputPairsFromInputOutput(featureVector, labelHandler.labelToOneInNRep(box.label))
		vectors.append(featureVector)
	return vectors

def makeInputOutputPairs(boxList, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, labelHandler):
	return makeFeatureVectors(boxList, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, labelHandler, True)

def popularSingleBoxFeatures(boxLists, targetPercentDocuments):
	# first go through each document and figure out the single-node features for the document
	featureLists = []
	counter = 0
	for boxList in boxLists:
		print "getting features", counter
		counter += 1
		features = getSingleNodeFeaturesOneDocument(boxList)
		featureLists.append(features)

	# decide on the filtered set of single-node features that is interesting to us, based on how many
	# different document use each single-node feature
	featureScores = {}
	for featureList in featureLists:
		for feature in featureList:
			featureScores[feature] = featureScores.get(feature, 0) + 1

	numberOfDocumentsThreshold = int(len(boxLists)*targetPercentDocuments)
	popularFeatures = [k for k, v in featureScores.items() if v >= numberOfDocumentsThreshold]
	boolFeatures, numFeatures = divideIntoBooleanAndNumericFeatures(popularFeatures, boxLists[0][0])

	print "decided on a feature set with", len(boolFeatures), "bool features and", len(numFeatures), "numerical features"

	# figure out the min and max values observed for each of our numerical features
	# we're going to use this to scale to the [-1,1] range when we actually make the feature vectors
	numFeaturesRanges = {}
	maxBoxes = 0
	for feature in numFeatures:
		numFeaturesRanges[feature] = [sys.maxint, -sys.maxint - 1] # min value seen, max value seen
	for boxList in boxLists:
		for box in boxList:
			for feature in numFeatures:
				minMax = numFeaturesRanges[feature]
				value = box.getFeature(feature)
				if (value < minMax[0]):
					numFeaturesRanges[feature][0] = value
				elif (value > minMax[1]):
					numFeaturesRanges[feature][1] = value
		numBoxes = len(boxList)
		if numBoxes > maxBoxes:
			maxBoxes = numBoxes
	for feature in numFeaturesRanges:
		minMax = numFeaturesRanges[feature]
		if minMax[1]-minMax[0] == 0:
			# we're not going to keep a feature that always has the same value
			numFeatures.remove(feature)

	return boolFeatures, numFeatures, numFeaturesRanges, maxBoxes

def getLabelHandler(boxLists):
	# figure out how many labels we have in this dataset
	labels = set()
	for boxList in boxLists:
		for box in boxList:
			labelStr = box.label
			labels.add(labelStr)
	print labels
	labelHandler = LabelHandler(list(labels))
	return labelHandler

def processTrainingDocuments(boxLists, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, trainingSetFilename, labelHandler, netFilename):
	# now let's figure out relationships between documents' boxes
	counter = 0
	for boxList in boxLists:
		print "finding relationships", counter
		counter += 1
		findRelationships(boxList)

	# now we're ready to make feature vectors
	counter = 0
	veccounter = 0
	filenames = []
	inputSize = 0
	outputSize = 0
	for boxList in boxLists:
		inputOutputPairs = makeInputOutputPairs(boxList, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, labelHandler)
		inputSize = len(inputOutputPairs[0][0])
		outputSize = len(inputOutputPairs[0][1])
		filename = "tmpFiles/tmp"+str(counter)+".data"
		filenames.append(filename)
		NNWrapper.saveTrainingSetToFile(inputOutputPairs, filename)
		veccounter += len(inputOutputPairs)
		print "input output pairs so far in stage", counter, ":", veccounter
		counter += 1


	print "input size: ", inputSize
	print "output size: ", outputSize

	NNWrapper.mergeSingleDocumentTrainingFiles(filenames, trainingSetFilename)

	print "ready to train the net"
	NNWrapper.trainNetwork(trainingSetFilename, netFilename, inputSize, outputSize)
	print "finished training the net"

def processTestingDocuments(boxLists, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, labelHandler, netFilename):
	# first go through each document and figure out the single-node features for the document
	for boxList in boxLists:
		getSingleNodeFeaturesOneDocument(boxList)

	# now let's figure out relationships between documents' boxes
	for boxList in boxLists:
		findRelationships(boxList)

	# now we're ready to make feature vectors
	for boxList in boxLists:
		inputOutputPairs = makeInputOutputPairs(boxList, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, labelHandler)
		NNWrapper.testNet(inputOutputPairs, netFilename, labelHandler)

def splitDocumentsIntoTrainingAndTestingSets(boxLists, trainingPortion):
	numDocuments = len(boxLists)
	splitPoint = int(trainingPortion*numDocuments)
	trainingDocuments = boxLists[:splitPoint]
	testingDocuments = boxLists[splitPoint:]
	return trainingDocuments, testingDocuments

def runOnCSV(csvname):
	boxLists = CSVHandling.csvToBoxlists(csvname) # each boxList corresponds to a document

	trainingDocuments, testingDocuments = splitDocumentsIntoTrainingAndTestingSets(boxLists, .8)

	trainingsetFilename = "trainingset.data"
	netFilename = "trainingset.net"

	doTraining = True

	# get everything we need to make feature vectors from both training and testing data
	boolFeatures, numFeatures, numFeaturesRanges, maxBoxes = popularSingleBoxFeatures(trainingDocuments, .10)
	labelHandler = getLabelHandler(trainingDocuments)

	if doTraining:
		# train
		processTrainingDocuments(trainingDocuments, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, trainingsetFilename, labelHandler, netFilename)

	# test
	NNWrapper.clearNNLogging()
	processTestingDocuments(testingDocuments, boolFeatures, numFeatures, numFeaturesRanges, maxBoxes, labelHandler, netFilename)

def makeInputOutputPairsForBoxPairs(pairs, labelFunc, labelHandler, vectorFunc):
	inputOutputPairs = []
	for pair in pairs:
		label = labelFunc(pair[0], pair[1])
		vec = vectorFunc(pair[0], pair[1])
		inputOutputPairs.append(makeInputOutputPairsFromInputOutput(vec, labelHandler.labelToOneInNRep(label)))
	return inputOutputPairs

def boxlistsToPairInputOutputPairs(boxLists, labelFunc, labelHandler, vectorFunc):
	allIOPairs = []
	for boxList in boxLists:
		allIOPairs += boxlistToPairInputOutputPairs(boxList, labelFunc, labelHandler, vectorFunc)
	return allIOPairs

def boxlistToPairInputOutputPairs(boxList, labelFunc, labelHandler, vectorFunc):
		pairs = itertools.permutations(boxList, 2)
		inputOutputPairs = makeInputOutputPairsForBoxPairs(pairs, labelFunc, labelHandler, vectorFunc)
		return inputOutputPairs

def learnAboveRelationship(csvname):
	boxLists = CSVHandling.csvToBoxlists(csvname) # each boxList corresponds to a document
	trainingSet, testingSet = splitDocumentsIntoTrainingAndTestingSets(boxLists, .8)

	testingOnly = False

	labelHandler = LabelHandler(["True", "False"])

	# functions for figuring out label
	simpleAbove = lambda x, y: str(x.above(y))
	harderAbove = lambda x, y: str(x.above(y) and not x.leftOf(y) and not y.leftOf(x))

	# functions for figuring out vector
	wholeVector = lambda x, y: x.wholeSingleBoxFeatureVector() + y.wholeSingleBoxFeatureVector()
	justBottomTop = lambda x, y: [x.getFeature("bottom-relative-percent"), y.getFeature("top-relative-percent")]
	topHeightBottomHeight = lambda x, y: [x.getFeature("top-relative-percent"), x.getFeature("height-percent"), y.getFeature("bottom-relative-percent"), y.getFeature("height-percent")]
	randomFeature = lambda x, y: [x.getFeature("bottom-relative-percent"), y.getFeature("top-relative-percent"), random.random()]

	trainingSetFilename = "aboveset.data"
	netFilename = "trainingset.net"

	boolFeatures, numFeatures, numFeaturesRanges = popularSingleBoxFeatures(trainingSet, .8) # this will take care of calling getSingleNodeFeaturesOneDocument

	if (not testingOnly):
		counter = 0
		for boxList in trainingSet:
			counter += 1
			print "setting single box features for document", counter
			setSingleBoxFeatures(boxList, boolFeatures, numFeatures, numFeaturesRanges)

		counter = 0
		veccounter = 0
		filenames = []
		inputSize = 0
		outputSize = 0
		for boxList in trainingSet:
			inputOutputPairs = boxlistToPairInputOutputPairs(boxList, simpleAbove, labelHandler, topHeightBottomHeight)
			inputSize = len(inputOutputPairs[0][0])
			outputSize = len(inputOutputPairs[0][1])
			print "input size: ", inputSize
			print "output size: ", outputSize
			filename = "tmpFiles/tmp"+str(counter)+".data"
			filenames.append(filename)
			NNWrapper.saveTrainingSetToFile(inputOutputPairs, filename)
			veccounter += len(inputOutputPairs)
			print "input output pairs so far in stage", counter, ":", veccounter
			counter += 1

		NNWrapper.saveTrainingSetToFile(inputOutputPairs, trainingSetFilename)
		NNWrapper.trainNetwork(trainingSetFilename, netFilename, inputSize, outputSize)

	syntheticDataset = False

	inputOutputPairs = []
	if (not syntheticDataset):
		# first go through each document and figure out the single-node features for the document
		for boxList in testingSet:
			getSingleNodeFeaturesOneDocument(boxList)

		counter = 0
		for boxList in testingSet:
			counter += 1
			print "setting single box features for document", counter
			setSingleBoxFeatures(boxList, boolFeatures, numFeatures, numFeaturesRanges)

		# now we're ready to make feature vectors
		inputOutputPairs = boxlistsToPairInputOutputPairs(testingSet, simpleAbove, labelHandler, topHeightBottomHeight)
	else:
		for i in range(0,1000):
			inputOutputPairs.append(makeInputOutputPairsFromInputOutput([0,0+i*.000001], labelHandler.labelToOneInNRep(str(True))))

	NNWrapper.clearNNLogging()
	NNWrapper.testNet(inputOutputPairs, netFilename, labelHandler)

runOnCSV("webDatasetFullCleaned.csv")
#learnAboveRelationship("webDatasetFullCleaned.csv")

