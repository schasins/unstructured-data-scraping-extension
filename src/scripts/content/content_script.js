/**********************************************************************
 * Author: S. Chasins
 **********************************************************************/

/**********************************************************************
 * Listeners and general set up
 **********************************************************************/

//messages received by this component
//utilities.listenForMessage("mainpanel", "content", "getMoreItems", getMoreItems);

//messages sent by this component
//utilities.sendMessage("content", "mainpanel", "selectorAndListData", data);

//user event handling
//document.addEventListener('mouseover', outline, true);

//for debugging purposes, print this tab's tab id
var tabId = 'setme';
utilities.listenForMessage("background", "content", "tabID", function(msg){tabID = msg; console.log("tab id: ", msg);});
utilities.sendMessage("content", "background", "requestTabID", {});

/**********************************************************************
 * Finding all text nodes
 **********************************************************************/

function getTextNodesIn(node, includeWhitespaceNodes) {
    var textNodes = [], nonWhitespaceMatcher = /\S/;

    function getTextNodes(node) {
        if (node.nodeType == 3) {
            if (includeWhitespaceNodes || nonWhitespaceMatcher.test(node.nodeValue)) {
                textNodes.push(node);
            }
        } else {
            for (var i = 0, len = node.childNodes.length; i < len; ++i) {
                getTextNodes(node.childNodes[i]);
            }
        }
    }

    getTextNodes(node);
    return textNodes;
}

function getTextNodeBoundingBox(textNode) {
    var height = 0;
    if (document.createRange) {
        var range = document.createRange();
        range.selectNodeContents(textNode);
        if (range.getBoundingClientRect) {
            var rect = range.getBoundingClientRect();
            return rect;
        }
    }
}

highlightCount = 0;
/* Highlight a node with a green rectangle */
function highlightNode(target, time) {
  var offset = $(target).offset();
  var boundingBox = getTextNodeBoundingBox(target);
  var newDiv = $('<div/>');
  var idName = 'hightlight-' + highlightCount;
  newDiv.attr('id', idName);
  newDiv.css('width', boundingBox.width);
  newDiv.css('height', boundingBox.height);
  newDiv.css('top', boundingBox.top);
  newDiv.css('left', boundingBox.left);
  newDiv.css('position', 'absolute');
  newDiv.css('z-index', 1000);
  newDiv.css('background-color', '#00FF00');
  newDiv.css('opacity', .1);
  $(document.body).append(newDiv);
  newDiv.hover(function(){newDiv.css('opacity', .4);},function(){newDiv.css('opacity', .1);});
  newDiv.click(function(){newDiv.css('background-color', '#0000FF'); processLabelingClick(target);});

  if (time) {
    setTimeout(function() {
      dehighlightNode(idName);
    }, 100);
  }

  return idName;
}

var thisPageHasBeenLabeledByHand = false;

function processLabelingClick(node){
  thisPageHasBeenLabeledByHand = true;
  node.__label__ = true;
  console.log(textNodes);
  utilities.sendMessage("content", "mainpanel", "newTrainingData", {data: makeFeatureLabelPairs(textNodes)});
}

function makeFeatureLabelPairs(nodeList){
  var pairs = [];
  for (var i = 0; i < nodeList.length; i++){
    pairs.push([nodeList[i].__features__, nodeList[i].__label__]);
  }
  return pairs;
}

function getFeatures(node){
  var features = [];

  // text
  text = node.nodeValue;
  words = text.trim().toLowerCase().split(/[\s\.,\-\/\#\!\$%\^&\*\;\:\{\}=\-\_\`\~\(\)"]+/g);
  uniqueWords = _.uniq(words);
  for (var i = 0; i < uniqueWords.length; i++){
    features.push("hasword-"+uniqueWords[i]);
  }

  // bounding box features
  var bbFeatures = ["top", "right", "bottom", "left", "width", "height"];
  var boundingBox = getTextNodeBoundingBox(node);
  for (var i = 0; i < bbFeatures.length; i++){
    var featureName = bbFeatures[i];
    features.push("has"+featureName+"-"+boundingBox[featureName]);
  }

  node.__features__ = features;
  node.__label__ = false;
}

var textNodes;
function processTextNodes(){
  textNodes = getTextNodesIn(document.body);
  for (var i = 0; i < textNodes.length; i++){
    highlightNode(textNodes[i]);
    getFeatures(textNodes[i]);
  }
}

setTimeout(processTextNodes, 2000);

function reproduceNet(serializedNet){
  var json = JSON.parse(serializedNet); // creates json object out of a string
  var net = new convnetjs.Net(); // create an empty network
  net.fromJSON(json); // load all parameters from JSON
  return net;
}

// inputArray is an array of floats
function classify(net, inputArray){
  var x = new convnetjs.Vol(inputArray);
  var prob = net.forward(x); 
  // TODO: right now just returning true if it's in the one 'true' category that the current user interaction allows
  // later we'll want to allow users to show many different categories, so we'll need to test all possible categories for highest probability
  return prob.w[1] > prob.w[0];
}

// accept feature set and serialized neural net from mainpanel
// run the net on all our textNodes here