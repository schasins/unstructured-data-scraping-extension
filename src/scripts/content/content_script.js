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
utilities.listenForMessage("mainpanel", "content", "newNet", handleNewNet);

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

/* Highlight a node with a green rectangle */
function highlightNode(target, idAddition) {
  var offset = $(target).offset();
  var boundingBox = getTextNodeBoundingBox(target);
  var newDiv = $('<div/>');
  var idName = 'highlight-' + idAddition;
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

  return idName;
}

var thisPageHasBeenLabeledByHand = false;

function processLabelingClick(node){
  node.__label__ = true;
  if (!thisPageHasBeenLabeledByHand){
    // we're only just labeling this by hand, so we might have some old guesses on here.  let's get rid of them
    for (var i = 0; i < textNodes.length; i++){ if (!textNodes[i].__label__){$("#highlight-"+i).css('background-color', '#00FF00'); }}
  }
  thisPageHasBeenLabeledByHand = true;
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
  var features = {};

  // text
  text = node.nodeValue;
  words = text.trim().toLowerCase().split(/[\s\.,\-\/\#\!\$%\^&\*\;\:\{\}=\-\_\`\~\(\)"]+/g);
  uniqueWords = _.uniq(words);
  for (var i = 0; i < uniqueWords.length; i++){
    features["hasword-"+uniqueWords[i]] = true;
  }

  // bounding box features
  var bbFeatures = ["top", "right", "bottom", "left", "width", "height"];
  var boundingBox = getTextNodeBoundingBox(node);
  for (var i = 0; i < bbFeatures.length; i++){
    var featureName = bbFeatures[i];
    features["has"+featureName+"-"+boundingBox[featureName]] = true;
  }

  // css/style features
  var styleFeatures = ["font-size", "font-family", "font-style", "font-weight", "color", "background-color"];
  var style = window.getComputedStyle(node.parentNode, null);
  for (var i = 0; i < styleFeatures.length; i++){
    var featureName = styleFeatures[i];
    features["has"+featureName+"-"+style.getPropertyValue(featureName)] = true;
  }

  node.__features__ = features;
  node.__label__ = false;
}

function populateGlobalPageInfo(textNodes){
  var fontSizeList = [];
  var textHeightList = [];
  var textWidthList = [];
  for (var i = 0; i < textNodes.length; i++){
    var node = textNodes[i];
    // font size
    var style = window.getComputedStyle(node.parentNode, null);
    fontSizeList.push(style.getPropertyValue("font-size"));
    // height and width
    var boundingBox = getTextNodeBoundingBox(node);
    textHeightList.push(boundingBox.height);
    textWidthList.push(boundingBox.width);
  }
  fontSizeList = _.uniq(fontSizeList).sort();
  textHeightList = _.uniq(textHeightList).sort();
  textWidthList = _.uniq(textWidthList).sort();
  return [fontSizeList, textHeightList, textWidthList];
}

var textNodes;
function processTextNodes(){
  textNodes = getTextNodesIn(document.body);

  // get some info we're going to use to determine the features
  var pageWidth = $(window).width();
  var pageHeight = $(window).height();
  var res = populateGlobalPageInfo(textNodes);
  var fontSizeList = res[0];
  var textHeightList = res[1];
  var textWidthList = res[2];
  console.log(fontSizeList, textWidthList, textHeightList);

  // get the actual features for the nodes
  for (var i = 0; i < textNodes.length; i++){
    highlightNode(textNodes[i], i);
    getFeatures(textNodes[i]);
  }
}

setTimeout(processTextNodes, 4000);

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
  console.log(prob.w);
  console.log("isTarget: "+(prob.w[1] > .1));
  return prob.w[1] > .1;
}

// accept feature set and serialized neural net from mainpanel
// run the net on all our textNodes here

// data has form {net: serializeNet, targetFeatures: chosenFeatures}
function handleNewNet(data){
  if (thisPageHasBeenLabeledByHand){
    return; // only run on pages that we haven't hand labeled
  }

  // figure out which nodes are in the target set with our new net
  var net = reproduceNet(data.net);
  var targetFeatures = data.targetFeatures;
  for (var i = 0; i < textNodes.length; i++){
    console.log(textNodes[i]);
    var isTarget = classify(net, common.makeFeatureVector(targetFeatures, textNodes[i].__features__));
    correspondingHighlight = $("#highlight-"+i);
    if (isTarget){
      correspondingHighlight.css('background-color', '#FF0000');
    }
    else {
      // clear any old highlighting
      correspondingHighlight.css('background-color', '#00FF00');
    }
  }
}
