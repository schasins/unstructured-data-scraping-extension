#!/usr/bin/python

import csv
import array
from operator import attrgetter
import re

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
			if (isNumber(firstBox.getFeature(feature)) and not feature.startswith("wordfreq") and not feature.startswith("charfreq")): # we don't want to add ranks and so on for wordfreqs
				rangeSet = set()
				for box in self.boxList:
					rangeSet.add(box.getFeature(feature))
				ranges[feature] = sorted(list(rangeSet))

		# then add a feature that gives the ranking
		for feature in firstBoxFeatures:
			if (isNumber(firstBox.getFeature(feature)) and not feature.startswith("wordfreq") and not feature.startswith("charfreq")):
				rangeLs = ranges[feature]
				rangeLsLen = len(rangeLs)
				for box in self.boxList:
					index = rangeLs.index(box.getFeature(feature))
					box.addFeature(feature+"_smallest_rank", index + 1)
					box.addFeature(feature+"_largest_rank", rangeLsLen - index)

	def addPercentagesForWidthAndHeightRelated(self):
		# first figure out some whole-document stuff
		docTop = lowest(self.boxList, "top")
		docLeft = lowest(self.boxList, "left")
		docHeight = highest(self.boxList, "bottom") - docTop
		docWidth = highest(self.boxList, "right") - docLeft

		# for some features, compare to the docHeight, docWidth
		for box in self.boxList:
			for feature in ["right", "left"]:
				box.addFeature(feature+"_relative", float(box.getFeature(feature))-docLeft)
			for feature in ["right_relative", "left_relative", "width"]:
				box.addFeature(feature+"_percent", float(box.getFeature(feature))/docWidth)

			for feature in ["top", "bottom"]:
				box.addFeature(feature+"_relative", float(box.getFeature(feature))-docTop)
			for feature in ["top_relative", "bottom_relative", "height"]:
				box.addFeature(feature+"_percent", float(box.getFeature(feature))/docHeight)

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
		try:
			wordsStr = (self.text.strip().lower()).decode('utf-8').encode("ascii","ignore")
		except Exception as e:
			print e
			print self.text
			return
		meaningfulCharsDict = {"?":"questionmark", ".":"period", "\"":"slash", "!":"exclamationpoint", "+":"plus","-":"minus", "|":"pipe", "@":"atsign", "[":"lbracket", "]":"rbracket", "=":"equalsign", "#":"hash"}
		charCount = 0
		for char in meaningfulCharsDict:
			currCharCount = wordsStr.count(char)
			self.addFeature("charfreq_"+meaningfulCharsDict[char], currCharCount)
			charCount += currCharCount

		words = re.split("[\s\.,\-\/\#\!\?\"+=\[\]\|@\^\*\;\:\{\}\=\-\_\`\~\(\)]*", wordsStr)
		numWords = len(words)
		self.addFeature("numwords", numWords)
		self.addFeature("numwordsandchars", charCount) # see if this gets us higher accuracy
		wordFreqs = {}
		for word in words:
			wordFreqs[word] = wordFreqs.get(word, 0) + 1
		for word in wordFreqs:
			self.addFeature("wordfreq_"+word, wordFreqs[word])
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
				a.append(self.getFeature(feature))
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

def popularSingleBoxFeatures(docList, targetNumDocuments):
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

	# numberOfDocumentsThreshold = int(len(docList)*targetPercentDocuments)
	popularFeatures = [k for k, v in featureScores.items() if v >= targetNumDocuments]

	print "decided on a feature set with", len(popularFeatures)
	return popularFeatures

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

def saveDatasetToFile(filename, dataset):
	f = open(filename, "w")
	strs = map(lambda x: ",".join(map(str, x)), dataset)
	strOutput = "\n".join(strs)
	f.write(strOutput)
	f.close()

# **********************************************************************
# High level structure
# **********************************************************************

def makeSingleNodeNumericFeatureVectors(filename, trainingsetFilename, testingsetFilename):
	docList = CSVHandling.csvToBoxlists(filename) # each boxList corresponds to a document

	trainingDocuments, testingDocuments = splitDocumentsIntoTrainingAndTestingSets(docList, .8)

	# get everything we need to make feature vectors from both training and testing data
	popularFeatures = popularSingleBoxFeatures(trainingDocuments, len(trainingDocuments)*.7) # TODO: how many documents should a feature actually need?

	trainingFeatureVectorsFiltered = datasetToRelation(trainingDocuments, popularFeatures)
	testingFeatureVectorsFiltered = datasetToRelation(testingDocuments, popularFeatures)

	print "len before", len(trainingFeatureVectorsFiltered[0])
	trainingFeatureVectorsFiltered, columnsToRemove = removeConstantColumns(trainingFeatureVectorsFiltered)
	print "identified", len(columnsToRemove), "constant columns"
	print "len after", len(trainingFeatureVectorsFiltered[0])
				
	print "len before", len(testingFeatureVectorsFiltered[0])
	testingFeatureVectorsFiltered = removeColumns(testingFeatureVectorsFiltered, columnsToRemove)
	print "removed constant columns from test set"
	print "len after", len(testingFeatureVectorsFiltered[0])

	# now let's actually save the datasets to file
	saveDatasetToFile(trainingsetFilename, trainingFeatureVectorsFiltered)
	saveDatasetToFile(testingsetFilename, testingFeatureVectorsFiltered)

def main():
	makeSingleNodeNumericFeatureVectors("../testDatasets/webDatasetFullCleaned.csv", "trainingSet.csv", "testSet.csv")
main()
