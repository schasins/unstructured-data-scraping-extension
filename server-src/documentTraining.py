#!/usr/bin/python

from operator import attrgetter
#from fann2 import libfann
import re

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

	def addFeature(self, featureName, value):
		self.features[featureName] = value

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

def allBoxesFeatures(boxList):
	return reduce(lambda acc, box : acc.union(box.getFeatures()), boxList, set())

def getSingleNodeFeaturesOneDocument(boxList):
	# first figure out some whole-document stuff
	docHeight = highest(boxList, "bottom") - lowest(boxList, "top")
	docWidth = highest(boxList, "right") - lowest(boxList, "left")

	for box in boxList:
		# get the individual features for each box
		addFeatures(box)

	# for numerical features, compare each value to the range of values in the document
	# first collect the range of values in the document
	# then add a feature that gives the ranking

	# for some features, compare to the docHeight, docWidth

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

	# for all the popular features, also build up all permutations of the relationships in front
	
def test():
	b1 = Box(2,2,10,10,"Swarthmore College", "edu")
	b2 = Box(11,11,30,30, "Jeanie", "name")
	b3 = Box(33,33,40,50, "boilerplate", "")
	doc = [b1,b2,b3]
	processSomeDocuments([doc,doc])

test()