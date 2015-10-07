#!/usr/bin/python

from operator import attrgetter
#from fann2 import libfann
import re
import copy
import sys

connection_rate = 1
learning_rate = 0.7
num_input = 2
num_hidden = 4
num_output = 1

desired_error = 0.0001
max_iterations = 100000
iterations_between_reports = 1000

class Box:
	def __init__(self, left, top, right, bottom, text, label):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom
		self.text = text
		self.label = label
		self.features = {}
		self.relationships = {}
		self.toOneRelationships = {}

	def addFeature(self, featureName, value):
		self.features[featureName] = value

	def getFeature(self, featureName):
		return self.features[featureName]

	def getFeatures(self):
		return self.features.keys()

def trainNetwork(dataFilename, netFilename):
	ann = libfann.neural_net()
	ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
	ann.set_learning_rate(learning_rate)
	ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

	ann.train_on_file(dataFilename, max_iterations, iterations_between_reports, desired_error)
	ann.save(netFilename)

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
	print toOneRelationships

def getSingleNodeFeaturesOneDocument(boxList):
	for box in boxList:
		# get the individual features for each box
		addFeatures(box)

	# for numerical features, compare each value to the range of values in the document
	addSmallestLargestRanksForNumerical(boxList)

	# for some features, compare to the docHeight, docWidth
	addPercentagesForWidthAndHeightRelated(boxList)

	for box in boxList:
		print box.features

	# get all the features from all the boxes
	documentFeatures = allBoxesFeatures(boxList)
	return documentFeatures

def processSomeDocuments(boxLists):
	# first go through each document and figure out the single-node features for the document
	featureLists = []
	for boxList in boxLists:
		features = getSingleNodeFeaturesOneDocument(boxList)
		featureLists.append(features)

	# decide on the filtered set of single-node features that is interesting to us, based on how many
	# different document use each single-node feature
	featureScores = {}
	for featureList in featureLists:
		for feature in featureList:
			featureScores[feature] = featureScores.get(feature, 0) + 1
	numberOfDocumentsThreshold = 2
	popularFeatures = [k for k, v in featureScores.items() if v >= numberOfDocumentsThreshold]
	print popularFeatures

	# now let's figure out relationships between documents' boxes
	for boxList in boxLists:
		findRelationships(boxList)

	# for all the popular features, also build up all permutations of the relationships in front
	# decide what relationships to gather, and how to use them for a feature vector
		
def test():
	b1 = Box(2,2,10,10,"Swarthmore College", "edu")
	b2 = Box(11,11,30,30, "Jeanie", "name")
	b3 = Box(28,32,40,50, "boilerplate", "")
	b4 = Box(28,51,40,58, "boilerplate 2", "")
	doc = [b1,b2,b3,b4]
	processSomeDocuments([doc,copy.deepcopy(doc)])

test()