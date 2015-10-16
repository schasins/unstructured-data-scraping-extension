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

connection_rate = 1
learning_rate = 0.7
num_input = 2
num_hidden = 50
num_output = 1

desired_error = 0.0007
max_iterations = 1000
iterations_between_reports = 1

class Box:
	def __init__(self, left, top, right, bottom, text, label, name="dontcare"):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom
		self.text = text
		self.label = label
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
				print "Freak out!  One of our boxes doesn't have a numeric feature so we don't know what value to put in."
			else:
				minMax = numFeaturesRanges[feature]
				oldval = self.getFeature(feature)

				oldrange = (minMax[1] - minMax[0]) # we've thrown out features that don't vary, so will never be 0
				newrange = float(1) # [0, 1]
				newval = (((oldval - minMax[0]) * newrange) / oldrange)

				a.append(newval)
		self.numFeatureVector = a

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

def train(trainingFilename):
	netFilename = "trainingset.net"
	trainNetwork(trainingFilename, netFilename, len(trainingSet[0][0]), len(trainingSet[0][1]))

def trainNetwork(dataFilename, netFilename, numInput, numOutput):
	ann = libfann.neural_net()
	ann.create_sparse_array(connection_rate, (numInput, num_hidden, numOutput))
	ann.set_learning_rate(learning_rate)
	ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

	ann.train_on_file(dataFilename, max_iterations, iterations_between_reports, desired_error)
	ann.save(netFilename)

testingSummaryFilename = "testingSummary.csv"
totalTested = 0
totalCorrect = 0
try:
	os.remove(testingSummaryFilename)
except:
	print "already no such file"

def testNet(trainingSet):
	global labelsToLabelIds, totalCorrect, totalTested, testingSummaryFilename

	testingSummaryFile = open(testingSummaryFilename, "a")

	netFilename = "trainingset.net"
	ann = libfann.neural_net()
	ann.create_from_file(netFilename)

	numTested = 0
	numLabeledCorrectly = 0
	for pair in trainingSet:
		featureVec = pair[0]
		actualLabel = pair[1]

		result = ann.run(featureVec)
		#print result, actualLabel
		numTested += 1
		winningIndex = result.index(max(result))
		actualLabelId = labelsToLabelIds[actualLabel]
		testingSummaryFile.write(str(winningIndex)+","+str(actualLabelId)+"\n")
		if winningIndex == actualLabelId:
			numLabeledCorrectly += 1

	print "numTested", numTested
	print "numLabeledCorrectly", numLabeledCorrectly
	totalTested += numTested
	totalCorrect += numLabeledCorrectly
	print "totalTested", totalTested
	print "totalCorrect", totalCorrect
	print "*****************"
	testingSummaryFile.close()

def testNetwork(netFilename):
	ann = libfann.neural_net()
	ann.create_from_file(netFilename)

	print ann.run([1, -1])

def values(list, attrName):
	return map(attrgetter(attrName),list)

def highest(list, attrName):
	attrVals = values(list, attrName)
	return max(attrVals)

def lowest(list, attrName):
	attrVals = values(list, attrName)
	return min(attrVals)

def addWordFeatures(box):
	wordsStr = box.text.strip().lower()
	words = re.split("[\s\.,\-\/\#\!\$%\^&\*\;\:\{\}=\-\_\`\~\(\)]*", wordsStr);
	uniqueWords = set(words);
	for word in uniqueWords:
		box.addFeature("hasword-"+word, True);

def addFeatures(box):
	for coord in ["left","top","right","bottom"]:
		box.addFeature(coord, attrgetter(coord)(box))
	box.addFeature("width", box.right-box.left)
	box.addFeature("height", box.bottom-box.top)

	addWordFeatures(box)

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

def above(b1, b2):
	return b1.bottom <= b2.top

def leftOf(b1, b2):
	return b1.right <= b2.left

def smallestGap(b1, b2):
	gaps = []

	if above(b1, b2):
		gaps.append(b2.top - b1.bottom)
	elif above(b2, b1):
		gaps.append(b1.top - b2.bottom)
	else:
		# they overlap, so the gap is 0
		gaps.append(0)

	if leftOf(b1, b2):
		gaps.append(b2.left - b1.right)
	elif leftOf(b2, b1):
		gaps.append(b1.left - b2.right)
	else:
		# they overlap, so the gap is 0
		gaps.append(0)

	return min(gaps)

def closest(b1, boxList):
	minGapSoFar = sys.maxint
	closestBoxSoFar = None
	for b2 in boxList:
		gap = smallestGap(b1, b2)
		if gap < minGapSoFar:
			closestBoxSoFar = b2
	return closestBoxSoFar

# TODO: for now have a very particular definition of above -- anywhere above, overlap in x direction
# might eventually want to try just the thing that's closest above
# or them not having to overlap
# should experiment with this
relationshipTypes = {}
relationshipTypes["above"] = lambda b1, b2 : above(b1, b2) and not leftOf(b1, b2) and not leftOf(b2, b1)
relationshipTypes["leftOf"] = lambda b1, b2 : leftOf(b1, b2) and not above(b1, b2) and not above(b2, b1)
relationshipTypes["below"] = lambda b1, b2 : above(b2, b1) and not leftOf(b1, b2) and not leftOf(b2, b1)
relationshipTypes["rightOf"] = lambda b1, b2 : leftOf(b2, b1) and not above(b1, b2) and not above(b2, b1)
relationshipTypes["sameleft"] = lambda b1, b2 : b1.left == b2.left
relationshipTypes["sametop"] = lambda b1, b2 : b1.top == b2.top
relationshipTypes["sameright"] = lambda b1, b2 : b1.right == b2.right
relationshipTypes["samebottom"] = lambda b1, b2 : b1.bottom == b2.bottom

#for now the only to-one relationship types are the ones that grab the closest from multinode relationships
toOneRelationshipTypes = []
for relationshipType in relationshipTypes:
	toOneRelationshipTypes.append(relationshipType+"-closest")

# the number of times we want to follow relationship pointers to build feature vectors
retlationshipDepth = 2

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
		addFeatures(box)

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

def wholeFeatureVector(box):
	return wholeFeatureVectorFromComponents(box.boolFeatureVector, box.numFeatureVector)

def wholeFeatureVectorFromComponents(boolVec, numVec):
	return map(int, list(boolVec)) + list(numVec)

def makeFeatureVectorWithRelationshipsToDepth(boxes, depth, relationshipsOnThisBranchSoFar, defaultBoolFeaturesVector, defaultNumFeaturesVector):
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
				featureVectorComponent = wholeFeatureVectorFromComponents(defaultBoolFeaturesVector, defaultNumFeaturesVector)
			elif boxesLength == 1:
				featureVectorComponent = wholeFeatureVector(allChildBoxesForRelationshipType[0])
			else:
				print "Freak out!  We followed only single-box relationships but got multiple boxes."
		else:
			bitvector = reduce(lambda x, y: x | y.boolFeatureVector, allChildBoxesForRelationshipType, defaultBoolFeaturesVector)
			featureVectorComponent = map(int, list(bitvector))

		featureVector += featureVectorComponent
		featureVectorAddition = makeFeatureVectorWithRelationshipsToDepth(allChildBoxesForRelationshipType, depth-1, newrelationshipsOnThisBranchSoFar, defaultBoolFeaturesVector, defaultNumFeaturesVector)
		featureVector += featureVectorAddition

	return featureVector

def makeFeatureVectors(boxList, boolFeatures, numFeatures, numFeaturesRanges, isLabeled):
	for box in boxList:
		box.setBoolFeatureVector(boolFeatures)
		box.setNumFeatureVector(numFeatures, numFeaturesRanges)

	defaultBoolFeaturesVector = bitarray("0"*len(boolFeatures))
	defaultNumFeaturesVector = [-1]*len(numFeatures)

	vectors = []
	for box in boxList:
		currBoxFeatures = wholeFeatureVector(box)
		featureVector = currBoxFeatures + makeFeatureVectorWithRelationshipsToDepth([box], retlationshipDepth, [], defaultBoolFeaturesVector, defaultNumFeaturesVector)
		if isLabeled:
			featureVector = [featureVector, box.label]
		vectors.append(featureVector)
	return vectors

def makeInputOutputPairs(boxList, boolFeatures, numFeatures, numFeaturesRanges):
	return makeFeatureVectors(boxList, boolFeatures, numFeatures, numFeaturesRanges, True)

labelIdCounter = 0
labelsToLabelIds = {}
numLabels = 0
def normalizeTrainingSet(trainingSet):
	global labelIdCounter, labelsToLabelIds, numLabels
	for pair in trainingSet:
		labelVec = [0]*numLabels
		labelVec[labelsToLabelIds[pair[1]]] = 1
		pair[1] = labelVec

	return trainingSet

def processTrainingDocuments(boxLists):
	# first go through each document and figure out the single-node features for the document
	featureLists = []
	counter = 0
	for boxList in boxLists:
		print "getting features", counter
		counter += 1
		features = getSingleNodeFeaturesOneDocument(boxList)
		featureLists.append(features)

	# figure out how many labels we have in this dataset
	global labelIdCounter, labelsToLabelIds, numLabels
	for boxList in boxLists:
		for box in boxList:
			labelStr = box.label
			if labelStr not in labelsToLabelIds:
				labelsToLabelIds[labelStr] = labelIdCounter
				labelIdCounter += 1
	numLabels = labelIdCounter

	# decide on the filtered set of single-node features that is interesting to us, based on how many
	# different document use each single-node feature
	featureScores = {}
	for featureList in featureLists:
		for feature in featureList:
			featureScores[feature] = featureScores.get(feature, 0) + 1

	targetPercentDocuments = .3 # it's enough to be in 30 percent of the documents
	numberOfDocumentsThreshold = int(len(boxLists)*targetPercentDocuments)
	popularFeatures = [k for k, v in featureScores.items() if v >= numberOfDocumentsThreshold]
	boolFeatures, numFeatures = divideIntoBooleanAndNumericFeatures(popularFeatures, boxLists[0][0])

	print "decided on a feature set with", len(boolFeatures), "bool features and", len(numFeatures), "numerical features"

	# figure out the min and max values observed for each of our numerical features
	# we're going to use this to scale to the [-1,1] range when we actually make the feature vectors
	numFeaturesRanges = {}
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
	for feature in numFeaturesRanges:
		minMax = numFeaturesRanges[feature]
		if minMax[1]-minMax[0] == 0:
			# we're not going to keep a feature that always has the same value
			numFeatures.remove(feature)

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
		inputOutputPairs = makeInputOutputPairs(boxList, boolFeatures, numFeatures, numFeaturesRanges)
		inputOutputPairs = normalizeTrainingSet(inputOutputPairs)
		inputSize = len(inputOutputPairs[0][0])
		outputSize = len(inputOutputPairs[0][1])
		filename = "tmpFiles/tmp"+str(counter)+".data"
		filenames.append(filename)
		saveTrainingSetToFile(inputOutputPairs, filename)
		veccounter += len(inputOutputPairs)
		print "input output pairs so far in stage", counter, ":", veccounter
		counter += 1

	trainingSetFilename = "trainingset.data"
	mergeSingleDocumentTrainingFiles(filenames, trainingSetFilename)

	print "ready to train the net"
	netFilename = "trainingset.net"
	trainNetwork(trainingSetFilename, netFilename, inputSize, outputSize)
	print "finished training the net"

	return boolFeatures, numFeatures, numFeaturesRanges

def processTestingDocuments(boxLists, boolFeatures, numFeatures, numFeaturesRanges):
	# first go through each document and figure out the single-node features for the document
	featureLists = []
	for boxList in boxLists:
		features = getSingleNodeFeaturesOneDocument(boxList)
		featureLists.append(features)

	# now let's figure out relationships between documents' boxes
	for boxList in boxLists:
		findRelationships(boxList)

	# now we're ready to make feature vectors
	for boxList in boxLists:
		inputOutputPairs = makeInputOutputPairs(boxList, boolFeatures, numFeatures, numFeaturesRanges)
		testNet(inputOutputPairs)
		
def test():
	b1 = Box(2,2,10,10,"Swarthmore College", "edu", "b1")
	b2 = Box(11,11,30,30, "Jeanie", "name", "b2")
	b3 = Box(28,32,40,50, "boilerplate", "", "b3")
	b4 = Box(28,51,40,58, "boilerplate 2", "", "b4")
	doc = [b1,b2,b3,b4]
	processTrainingDocuments([doc,copy.deepcopy(doc)])

def runOnCSV(csvname):
	csvfile = open(csvname, "rb")
	reader = csv.reader(csvfile, delimiter=",", quotechar="\"")

	documents = {}
	boxIdCounter = 0
	for row in reader:
		docName = row[0]
		left = int(row[1])
		top = int(row[2])
		right = int(row[3])
		bottom = int(row[4])
		text = row[5]
		label = row[6]
		boxId = str(boxIdCounter)

		boxIdCounter += 1

		box = Box(left, top, right, bottom, text, label, boxId)

		boxList = documents.get(docName, [])
		boxList.append(box)
		documents[docName] = boxList

	allDocuments = documents.keys()
	print allDocuments
	numDocuments = len(allDocuments)
	trainingPercentage = .3
	splitPoint = int(trainingPercentage*numDocuments)
	trainingDocuments = allDocuments[:splitPoint]
	testingDocuments = allDocuments[splitPoint:]

	trainingBoxLists = map(lambda x: documents[x], trainingDocuments)
	testingBoxLists = map(lambda x: documents[x], testingDocuments)

	boolFeatures, numFeatures, numFeaturesRanges = processTrainingDocuments(trainingBoxLists)
	processTestingDocuments(testingBoxLists, boolFeatures, numFeatures, numFeaturesRanges)

runOnCSV("trainingData/finaldataset.csv")