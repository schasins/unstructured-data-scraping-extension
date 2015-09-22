var all_script_results = [];
function setUp(){

  //messages received by this component
  utilities.listenForMessage("content", "mainpanel", "newTrainingData", handleNewTrainingData);
  
  //messages sent by this component
  //utilities.sendMessage("mainpanel", "content", "startProcessingList", "");


  var urls = ["http://www.cs.berkeley.edu/~schasins/#/resume","https://www.linkedin.com/pub/fanny-zhao/31/4aa/853", "https://www.linkedin.com/in/lizelting", "http://www.indeed.com/r/Robert-DeKoch/8e4112cb91465768"];
  for (var i = 0; i < urls.length; i++){
    chrome.tabs.create({ url: urls[i] });
  }

}

$(setUp);

var trainingData = {};
var needToTrain = false;

function handleNewTrainingData(data){
	var tabId = data.tab_id; // TODO: Fix this.  In future if we get messages from multiple frames in a single tab, we might overwrite existing training data from this tab.
	trainingData[tabId] = data.data;
	console.log(trainingData);
  needToTrain = true;
}

function trainOnCurrentTrainingData(){
	// decide on a set of features to use
	// we'll use the set of features that appears on all pages
	// TODO: in future change this to the set of features that appears on more than one page
	var perPageFeatures = [];
	for (var tabId in trainingData){
		// TODO: using Object.keys here probably isn't the most efficient way
		perPageFeatures.push(_.union.apply(_, _.map(trainingData[tabId], function(pageNode){return Object.keys(pageNode[0]);})));
	}
	var chosenFeatures = _.intersection.apply(_, perPageFeatures);
	console.log("chosenFeatures: ", chosenFeatures);

	// make net with correct vec length
	var net = makeNeuralNet(chosenFeatures.length, 2); // currently output fixed at 2
	var trainer = makeTrainer(net);

	// train net
  var trainingDataVectors = [];
	for (var tabId in trainingData){
		var currData = trainingData[tabId];
		for (var i = 0; i < currData.length; i++){
			var pair = currData[i];

			var featureStringsDict = pair[0];
			var featureVector = common.makeFeatureVector(chosenFeatures, featureStringsDict);

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

// continuously check if we need to be rerunning training
function trainingLoop(){
  if (needToTrain){ needToTrain = false; trainOnCurrentTrainingData();}
  setTimeout(trainingLoop, 0);
}
trainingLoop();

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

function trainOnCurrentData(){

}