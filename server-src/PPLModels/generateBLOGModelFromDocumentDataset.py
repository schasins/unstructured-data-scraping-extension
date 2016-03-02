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
		data[i] = data[i][:featureStart] + map(float, data[i][featureStart:]) # turn these into numbers
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

def makeBLOGModel(headers, dataset, modelFilename):
	labelDict = divideByLabel(dataset)
	outputStr = "type Label;\n\n"
	
	numFeatures = len(headers) - featureStart # remember the first col is labels, second is doc name
	numRows = len(dataset)

	labels = labelDict.keys()
	outputStr += "distinct Label " + ",".join(labels) + ";\n\n"

	outputStr += "random Label L ~ Categorical({"
	weightStrs = []
	for key in labels:
		weightStrs.append(key + " -> " + str(float(len(labelDict[key]))/numRows))
	outputStr += ", ".join(weightStrs) + "});\n\n"

	variableStrs = []
	for i in range(numFeatures): 
		# now we'll add one new variable per feature
		variableStr = "random Real " + headers[i + featureStart] + " ~ "
		for label in labels:
			featureValsForLabel = map(lambda x: x[i], labelDict[label]) # just extract the current feature
			mean = numpy.mean(featureValsForLabel)
			variance = numpy.var(featureValsForLabel)
			if variance == 0: # 0 variance is not ok
				variance = .000000001
			if label != labels[-1]:
				variableStr += "if (L == " + label + ") then Gaussian(" + str(mean) + ", " + "{0:.10f}".format(variance) + ") else "
			else:
				# last label, so this is the else case
				variableStr += "Gaussian(" + str(mean) + ", " + "{0:.10f}".format(variance) + ")"
		variableStr += ";"
		variableStrs.append(variableStr)

	outputStr += "\n".join(variableStrs)

	o = open(modelFilename, "w")
	o.write(outputStr)
	o.close()

def testBLOGModel(headers, dataset, modelFilename):
	numFeatures = len(headers) - featureStart # remember the first col is labels, second is doc name

	# obs width = 17.0;
	for row in dataset:
		obsStrings = [];
		for i in range(numFeatures):
			featureName = headers[i + featureStart]
			featureVal = row[i + featureStart]
			obsStrings.append("obs " + featureName + " = " + str(featureVal) + ";")
		
		modelStr = open(modelFilename, "r").read()
		outputStr = modelStr + "\n".join(obsStrings)
		outputStr += "\n\nquery L;"

		o = open("tmp.blog", "w")
		o.write(outputStr)
		o.close()

		# now we just have to run this model, extract the results
		try:
			t0 = time.time()
			strOutput = subprocess.check_output("blog -n 5000 tmp.blog".split(" ")) # TODO: how many samples should we actually take?
			t1 = time.time()
			seconds = t1-t0

			m, s = divmod(seconds, 60)
			h, m = divmod(m, 60)
			print "Time to infer:"
			print "%d:%02d:%02d" % (h, m, s)

			print "actual label: ", row[0]
			print strOutput.split("======== Query Results =========")[1]
			print "****************"
		except:
			raise Exception("Couldn't get output from running BLOG.")

def main():
	makeModel = False

	if makeModel:
		dataset = getDatasetFromFile("trainingSet.csv")
		headers, data = split(dataset)
		makeBLOGModel(headers, data, "model.blog")
	# now that we have a model, let's actually test it with the training set
	dataset = getDatasetFromFile("testSet.csv")
	headers, data = split(dataset)
	testBLOGModel(headers, data, "model.blog")

main()