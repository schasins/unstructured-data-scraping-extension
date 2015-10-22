var all_script_results = [];
function setUp(){

  $("button").button();	

  //messages received by this component
  utilities.listenForMessage("content", "mainpanel", "newTrainingData", handleNewTrainingData);
  utilities.listenForMessage("content", "background", "newTrainingDataPairs", handleNewTrainingDataPairs);
  utilities.listenForMessage("content", "mainpanel", "onePageDataset", handleNewOnePageDataset);
  
  //messages sent by this component
  //utilities.sendMessage("mainpanel", "content", "startProcessingList", "");

  $("#addLabel").click(handleNewLabel);
  $("#download").click(handleDownload);

  var currentTabId = null;
  chrome.tabs.create({ url: websites[0] }, function(tabInfo){console.log(tabInfo); currentTabId = tabInfo.id; openNewPageIfNoneOpenNow()}); // websites is defined in an additional file
  var currentWebsiteIndex = 1;

  function openNewPageIfNoneOpenNow(){
  	chrome.tabs.get(currentTabId, function(tab){
			if (tab === undefined) {
				if (currentWebsiteIndex > websites.length){
					console.log("Out of websites to label.");
				}
				chrome.tabs.create({ url: websites[currentWebsiteIndex] }, function(tabInfo){currentTabId = tabInfo.id});
				currentWebsiteIndex += 1;
			}
			setTimeout(openNewPageIfNoneOpenNow, 100);	
  	});
  }

}

$(setUp);

var pageFeatureLists = {};
var needToTrain = false;

function handleNewTrainingData(data){
	var tabId = data.tab_id; // TODO: Fix this.  In future if we get messages from multiple frames in a single tab, we might overwrite existing training data from this tab.
	pageFeatureLists[tabId] = data.globalFeaturesLs;
}

var trainingDataPairs = {};

function handleNewTrainingDataPairs(data){
	var tabId = data.tab_id; // TODO: Fix this.  In future if we get messages from multiple frames in a single tab, we might overwrite existing training data from this tab.
	trainingDataPairs[tabId] = data.pairs;

	if (Object.keys(trainingDataPairs).length === Object.keys(pageFeatureLists).length){
		trainOnCurrentTrainingData();
	}
}

var chosenFeatures;

function makeNewFeatureSet(){
	// decide on a set of features to use
	// we'll use the set of features that appears on all pages
	// TODO: in future change this to the set of features that appears on more than one page
	var perPageFeatures = [];
	for (key in pageFeatureLists){
		perPageFeatures.push(Object.keys(pageFeatureLists[key]));
	}
	console.log(perPageFeatures);
	chosenFeatures = _.intersection.apply(_, perPageFeatures);
	console.log("chosenFeatures: ", chosenFeatures);

	// clear out old training data pairs so we don't mix up old ones with new ones
	trainingDataPairs = {};

	// send message out to labeled tabs to get their training data vectors
	utilities.sendMessage("mainpanel", "content", "getTrainingDataWithFeatureSet", {targetFeatures:chosenFeatures});
}

function trainOnCurrentTrainingData(){
	// make net with correct vec length
	var net = makeNeuralNet(chosenFeatures.length, 2); // currently output fixed at 2
	var trainer = makeTrainer(net);

	// train net
	var trainingDataVectors = [];
	for (var tabId in trainingDataPairs){
		var currData = trainingDataPairs[tabId];
		for (var i = 0; i < currData.length; i++){
			var pair = currData[i];

			var featureVector = pair[0];
			var isTarget = pair[1];
			var category = 0;
			if (isTarget) {category = 1;}

      trainingDataVectors.push([featureVector, category]);
		}
	}

  for (var i = 0; i < 100; i++){
    for (var j = 0; j < trainingDataVectors.length; j++){
      train(trainer, trainingDataVectors[j][0], trainingDataVectors[j][1]);
    }
  }

	// send net and set of acceptable features to the content scripts
	var serializedNet = serializeNet(net);
	utilities.sendMessage("mainpanel", "content", "newNet", {net: serializedNet, targetFeatures: chosenFeatures});
}

function makeNeuralNet(inputVectorLength, outputNumClasses){
	// specifies a 2-layer neural network with one hidden layer of 20 neurons
	var layer_defs = [];
	// input layer declares size of input. here: 2-D data
	// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
	// then the first two dimensions (sx, sy) will always be kept at size 1
	layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:inputVectorLength});
	// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
	layer_defs.push({type:'fc', num_neurons:20, activation:'relu'}); 
	// declare the linear classifier on top of the previous hidden layer
	layer_defs.push({type:'softmax', num_classes:outputNumClasses});

	var net = new convnetjs.Net();
	net.makeLayers(layer_defs);
	return net;
}

function makeTrainer(net){
	var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
	return trainer;
}

// inputArray is an array of floats
// outputClass is an integer from 0 to max class index
function train(trainer, inputArray, outputClass){
	var x = new convnetjs.Vol(inputArray);
	trainer.train(x,outputClass);
}

function serializeNet(net){
	// network outputs all of its parameters into json object
	var json = net.toJSON();
	// the entire object is now simply string. You can save this somewhere
	var str = JSON.stringify(json);
	return str;
}

var colors = ["#9EE4FF","#9EB3FF", "#BA9EFF", "#9EFFEA", "#E4FF9E", "#FFBA9E", "#FF8E61"];
var buttonsSoFar = 0;

function handleNewLabel(){
	var labelText = $("#newLabel").val();
	var color = colors[buttonsSoFar];
	var buttons = $("#labelButtons");
	var newButton = $("<div class='labelBox'>"+labelText+"</div>");
	newButton.css("background-color", color);
	newButton.click(function(){
		$(".labelBox").removeClass("labelBoxCurrent"); 
		newButton.addClass("labelBoxCurrent"); 
		utilities.sendMessage("mainpanel", "content", "currentLabel", {currentLabel: labelText, currentColor: color});
	});
	buttons.append(newButton);
	buttonsSoFar += 1;
}

var allPageDatasets = {};

// form:
// utilities.sendMessage("content", "mainpanel", "onePageDataset", {featureDicts: _.map(textNodes, function(x){return x.__features__})});
function handleNewOnePageDataset(data){
	var tabId = data.tab_id; // TODO: Fix this.  In future if we get messages from multiple frames in a single tab, we might overwrite existing training data from this tab.
	var featureDicts = data.featureDicts;
	allPageDatasets[tabId] = featureDicts;
}

function handleDownload(){
	console.log("Downloading.");
	var csvString = "";
	var keys = null;
	for (var tabId in allPageDatasets){
		featureDicts = allPageDatasets[tabId];
		if (keys === null){
			keys = Object.keys(featureDicts[0]);
			csvString += keys.join(",") + "\n";
		}
		for (var i = 0; i < featureDicts.length; i++) {
			var dict = featureDicts[i];
			var row = [];
			for (var j = 0; j < keys.length; j++){
				var val = dict[keys[j]];
				val = val === null ? '' : val.toString();
        val = val.replace(/"/g, '""');
        if (val.search(/("|,|\n)/g) >= 0) {
        	val = '"' + val + '"';
        }
				row.push(val);
			}
			csvString += row.join(",") + "\n";
		}
	}
	var blob = new Blob([csvString], { type: "text/csv;charset=utf-8" });
	saveAs(blob, "web_dataset.csv");

}

