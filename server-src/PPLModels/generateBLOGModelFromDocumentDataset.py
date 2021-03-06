import numpy
import csv
import subprocess
import re
import time

featureStart = 2

def getDatasetFromFile(filename):
		csvfile = open(filename, "rb")
		dataset = []
		reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
		for row in reader:
			dataset.append(row)
		return dataset

def convertToIntOrFloat(numStr):
	num = float(numStr)
	if num % 1 == 0:
		return int(num) # for the test set, we'll be converting these nums directly to strings in BLOG models, and BLOG freaks out if we "observe" 0.0 for an integer valued variable
	return num

def split(dataset):
	# the first row is headers
	headers = dataset[0]

	# but let's go through and get rid of anything bad
	headers = map(lambda st: re.sub("[-'&$%]","_", st), headers) # these characters will all mess up BLOG if put in variable names
	# since we've replaced these characters all the same way, we could have given two features the same name.  fix that.
	counter = 0
	for i in range(len(headers)):
		currFeatureName = headers[i]
		if currFeatureName in headers[i+1:]:
			headers[i] = currFeatureName+str(counter)
			counter += 1

	data = dataset[1:]
	for i in range(len(data)):
		data[i] = data[i][:featureStart] + map(convertToIntOrFloat, data[i][featureStart:]) # turn these into numbers
	return headers, data

def divideByLabel(dataset):
	# the first col is labels
	# the second call is document name
	labelDict = {}
	for row in dataset:
		label = row[0]
		features = row[featureStart:]
		ls = labelDict.get(label, [])
		ls.append(features)
		labelDict[label] = ls
	return labelDict

def getFeatureType(featureName):
	if featureName.startswith("wordfreq"):
		return "wordFreq"
	elif featureName.startswith("charfreq"):
		return "charFreq"
	else:
		return "everythingElse"

numWordsVarName = "numWords"
numWordsAndCharsVarName = "numWordsAndChars"
numTextboxesPlaceholder = "###NUMTEXTBOXESPLACEHOLDER###"

def makeGaussianStringFromValues(values):
			 # for now assuming everything else (the width, height, so on) are normally distributed.  should revisit this in future
			 mean = numpy.mean(values)
			 variance = numpy.var(values)
			 if variance == 0: # 0 variance is not ok
								variance = .0000000001
			 distribString = "Gaussian(" + str(mean) + ", " + "{0:.10f}".format(variance) + ")"
			 return distribString 

def makeBLOGModel(headers, dataset, modelFilename):
	labelDict = divideByLabel(dataset)
	outputStr = "type Textbox;\n\ntype Label;\n\n"

	outputStr += "distinct Textbox Textbox[" + numTextboxesPlaceholder + "];\n\n"
	
	numFeatures = len(headers) - featureStart # remember the first col is labels, second is doc name
	numRows = len(dataset)

	labels = labelDict.keys()
	outputStr += "distinct Label " + ",".join(labels) + ";\n\n"

	outputStr += "random Label L(Textbox t) ~ Categorical({"
	weightStrs = []
	for key in labels:
		weightStrs.append(key + " -> " + str(float(len(labelDict[key]))/numRows))
	outputStr += ", ".join(weightStrs) + "});\n\n"

	numWordsIndex = headers.index("numwords") - featureStart # the feature that has number of words in textbox
	numWordsAndCharsIndex = headers.index("numwordsandchars") - featureStart # the feature that has number of words in textbox
	totalNumWordsDict = {}
	totalNumWordsAndCharsDict = {}
	for label in labels:
			wordCountsForLabel = map(lambda x: x[numWordsIndex], labelDict[label])
			totalNumWordsDict[label] = sum(wordCountsForLabel)
			wordAndCharCountsForLabel = map(lambda x: x[numWordsAndCharsIndex], labelDict[label])
			totalNumWordsAndCharsDict[label] = sum(wordAndCharCountsForLabel)
	print totalNumWordsDict
	print totalNumWordsAndCharsDict

	variableStrs = []
	for i in range(numFeatures): 
		# now we'll add one new variable per feature
		featureName = headers[i + featureStart]
		featureType = getFeatureType(featureName)
		varType = "Real"
		if featureType == "wordFreq" or featureType == "charFreq":
			varType = "Integer"

		variableStr = "random " + varType + " " + featureName + "(Textbox t, Integer " + numWordsVarName + ", Integer " + numWordsAndCharsVarName + ") ~ "
		for label in labels:
			featureValsForLabel = map(lambda x: x[i], labelDict[label]) # just extract the current feature

			# based on the type of the feature, figure out what distribution to use, what parameters
			distribString = ""
			if featureType == "wordFreq" or featureType == "charFreq":
				if featureType == "wordFreq":
					divisor = totalNumWordsDict[label]
					varName = numWordsVarName
				else:
					divisor = totalNumWordsAndCharsDict[label]
					varName = numWordsAndCharsVarName
				probEachWordIsCurrWord = float(sum(featureValsForLabel))/divisor # this val is reasonable for words since it's the word count, but doesn't really make sense for the char counts, since that's not included in the word count.  todo: fix this
				if probEachWordIsCurrWord == 0:
					probEachWordIsCurrWord = .0000000001
				distribString = "Binomial(" + varName + ",  {0:.10f})".format(probEachWordIsCurrWord)
			else:
				# for now assuming everything else (the width, height, so on) are normally distributed.  should revisit this in future
				distribString = makeGaussianStringFromValues(featureValsForLabel)
			# build up the string with the distrib
			if label != labels[-1]:
				variableStr += "if (L(t) == " + label + ") then " + distribString + " else "
			else:
				# last label, so this is the else case
				variableStr += distribString
		variableStr += ";"
		variableStrs.append(variableStr)

	outputStr += "\n".join(variableStrs)

	o = open("models/"+modelFilename, "w")
	o.write(outputStr)
	o.close()

def testBLOGModel(headers, dataset, modelFilename):
	tmpFilename = "tmp"+modelFilename

	numFeatures = len(headers) - featureStart # remember the first col is labels, second is doc name
	numWordsIndex = headers.index("numwords")
	numWordsAndCharsIndex = headers.index("numwordsandchars")

	summaryFile = open("summaries/summaryFile_"+timeStr+".csv", "w")
	t0Outer = time.time()
	# obs width = 17.0;
	correctCount = 0
        falsePositiveCount = 0
        falseNegativeCount = 0
        wrongLabelCount = 0

        numLabeledCount = 0
        numLabeledCorrectlyCount = 0
        numWeLabeledCount = 0

	documents = {}
	for row in dataset:
		textBoxes = documents.get(row[1],[]) # row[1] is the document name
		textBoxes.append(row)
		documents[row[1]] = textBoxes

				# now we say how many textboxes we expect to have per document
	for docName in documents:
		documentLength = len(documents[docName])
								
		modelStr = open("models/"+modelFilename, "r").read()
		modelStr = modelStr.replace(numTextboxesPlaceholder, str(documentLength), 1)

		rows = documents[docName]
		obsStrings = []
		for ind in range(documentLength):
			row = rows[ind]
			for i in range(numFeatures):
				featureName = headers[i + featureStart]
				featureVal = row[i + featureStart]
				obsStrings.append("obs " + featureName + "(Textbox[" + str(ind) + "]," + str(row[numWordsIndex]) + "," + str(row[numWordsAndCharsIndex]) + ") = " + str(featureVal) + ";")
		
		outputStr = modelStr + "\n".join(obsStrings) + "\n"
								
		for ind in range(documentLength):
			outputStr += "\nquery L(Textbox[" + str(ind) + "]);"

		o = open("tmpmodels/"+tmpFilename, "w")
		o.write(outputStr)
		o.close()

		# now we just have to run this model, extract the results
		t0 = time.time()
		try:
			strOutput = subprocess.check_output(("blog -n 100000 tmpmodels/"+tmpFilename+" -s blog.sample.MHSampler").split(" ")) # TODO: how many samples should we actually take?
		except:
			raise Exception("Couldn't get output from running BLOG.")

		t1 = time.time()
		seconds = t1-t0

		m, s = divmod(seconds, 60)
		h, m = divmod(m, 60)
		print "%d:%02d:%02d" % (h, m, s)

		result = strOutput.split("======== Query Results =========")[1]

		for ind in range(documentLength):
			row = rows[ind]
			# Distribution of values for L(Textbox[8])
			results = result.split("Distribution of values for L(Textbox[" + str(ind) + "]")[1].split("\n")
			# print results
			winningLabel = results[1] # first entry is just empty space
			winningLabel = winningLabel.strip().split("\t")
			guessedLabel = winningLabel[0]
			prob = winningLabel[1]
			if row[0] != "nolabel":
				numLabeledCount += 1

			if guessedLabel != "nolabel":
				numWeLabeledCount += 1

			correct = row[0] == guessedLabel
			if correct:
				correctCount += 1
				if row[0] != "nolabel":
					numLabeledCorrectlyCount += 1 # this is only for items that aren't nolabel
			else:
				if row[0] == "nolabel":
					# we guessed a label when it was a nolabel
					falsePositiveCount += 1
				elif guessedLabel == "nolabel":
					# ugh this was actually a labeled thing
					falseNegativeCount += 1
				else:
					wrongLabelCount += 1

			summaryFileLine = row[0]+","+guessedLabel+","+prob+","+str(correct)
			print summaryFileLine
			summaryFile.write(summaryFileLine+"\n")
			summaryFile.flush()
	t1Outer = time.time()
	seconds = t1Outer-t0Outer

	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print "Total labeling time: %d:%02d:%02d" % (h, m, s)
	summaryFile.write(str(seconds))
	summaryFile.close()

	print "Num correct: ", correctCount
	print "False positives: ", falsePositiveCount
	print "False negatives: ", falseNegativeCount
	print "Wrong label: ", wrongLabelCount
	print "--------------------"
	print "Total datapoints: ", len(dataset)
	print "===================="
	print "Labeled datapoints in test set:", numLabeledCount
	print "Points that we labeled in test set:", numWeLabeledCount
	print "Labeled datatpoints in test ste that we labeled correctly:", numLabeledCorrectlyCount
	print "===================="
	print "Precision (strict): ", float(numLabeledCorrectlyCount)/numWeLabeledCount # percentage of things we labeled that actually should have been labeled
	print "Precision (weak): ", float(numLabeledCorrectlyCount + wrongLabelCount)/numWeLabeledCount # percentage of things we labeled that actually should have been labeled
	print "Recall (strict): ", float(numLabeledCorrectlyCount)/numLabeledCount # percentage of things that should have been labeled that we actually did label
	print "Recall (weak): ", float(numLabeledCorrectlyCount + wrongLabelCount)/numLabeledCount
	print "Percent correct: ", float(correctCount)/len(dataset)

timeStr = time.strftime("%d_%m_%Y_%H_%M")

def main():
	modelFilename = "model_"+timeStr+".blog"
	dataset = getDatasetFromFile("trainingSet.csv")
	headers, data = split(dataset)
	makeBLOGModel(headers, data, modelFilename)
	# now that we have a model, let's actually test it with the training set
	dataset = getDatasetFromFile("testSet.csv")
	headers, data = split(dataset)
	testBLOGModel(headers, data, modelFilename)

main()
