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
    pairs.push([nodeList[i].__features2__, nodeList[i].__label__]);
  }
  return pairs;
}

function getContinuousFeature(features, featureName, featureValue, featureValueList){
  features["has"+featureName] = featureValue; // continuous
  var loc = featureValueList.indexOf(featureValue);
  features["has"+featureName+"smallest"] = loc; // continous
  features["has"+featureName+"largest"] = featureValueList.length - loc; // continous
}

function getFeatures(node, pageWidth, pageHeight, featureValueLists){
  var features = {};

  // text
  text = node.nodeValue;
  words = text.trim().toLowerCase().split(/[\s\.,\-\/\#\!\$%\^&\*\;\:\{\}=\-\_\`\~\(\)"]+/g);
  uniqueWords = _.uniq(words);
  for (var i = 0; i < uniqueWords.length; i++){
    features["hasword-"+uniqueWords[i]] = true; // discrete
  }

  // bounding box features
  var bbFeaturesWidth = ["right", "left", "width"];
  var bbFeaturesHeight = ["top", "bottom", "height"];
  var boundingBox = getTextNodeBoundingBox(node);
  for (var i = 0; i < bbFeaturesWidth.length; i++){
    var featureName = bbFeaturesWidth[i];
    var featureValue = boundingBox[featureName];
    getContinuousFeature(features, featureName, featureValue, featureValueLists[featureName]);
    features["haspercent"+featureName] = boundingBox[featureName]/pageWidth; // continous
  }
  for (var i = 0; i < bbFeaturesHeight.length; i++){
    var featureName = bbFeaturesHeight[i];
    var featureValue = boundingBox[featureName];
    getContinuousFeature(features, featureName, featureValue, featureValueLists[featureName]);
    features["haspercent"+featureName] = boundingBox[featureName]/pageHeight; // continous
  }

  // css/style features
  var styleFeatures = ["font-family", "font-style", "font-weight", "color", "background-color"];
  var style = window.getComputedStyle(node.parentNode, null);
  for (var i = 0; i < styleFeatures.length; i++){
    var featureName = styleFeatures[i];
    features["has"+featureName+"-"+style.getPropertyValue(featureName)] = true; // discrete
  }
  // fontsize is the one continous feature in the css ones
  var fontSize = parseInt(style.getPropertyValue("font-size"));
  getContinuousFeature(features, "fontsize", fontSize, featureValueLists.fontsize);

  node.__features__ = features;
  node.__label__ = false;
}

function findRelationships(nodes, idx){
  var node = nodes[idx];
  node.__relationships__ = {above:[],leftof:[],below:[],rightof:[],precededby:[],precedes:[],sameleft:[],sametop:[],sameright:[],samebottom:[]};

  // always preceded by the one before it in the list
  if (idx > 0){
    node.__relationships__.precededby.push(nodes[idx-1]);
  }
  // always preceds the one after it in the list
  if (idx < nodes.length - 1){
    node.__relationships__.precedes.push(nodes[idx+1]);
  }

  for (var i = 0; i < nodes.length; i++){
    if (i === idx){
      continue;
    }
    var candidateNode = nodes[i];

    if (candidateNode.__features__.hastop >= node.__features__.hasbottom) {node.__relationships__.above.push(candidateNode);}
    if (candidateNode.__features__.hasleft >= node.__features__.hasright) {node.__relationships__.leftof.push(candidateNode);}
    if (candidateNode.__features__.hasbottom <= node.__features__.hastop) {node.__relationships__.below.push(candidateNode);}
    if (candidateNode.__features__.hasright <= node.__features__.hasleft) {node.__relationships__.rightof.push(candidateNode);}

    if (candidateNode.__features__.hasleft == node.__features__.hasleft) {node.__relationships__.sameleft.push(candidateNode);}
    if (candidateNode.__features__.hastop == node.__features__.hastop) {node.__relationships__.sametop.push(candidateNode);}
    if (candidateNode.__features__.hasright == node.__features__.hasright) {node.__relationships__.sameright.push(candidateNode);}
    if (candidateNode.__features__.hasbottom == node.__features__.hasbottom) {node.__relationships__.samebottom.push(candidateNode);}
  }
}

function populateGlobalPageInfo(textNodes){
  var featureValueLists = {};
  for (var i = 0; i < textNodes.length; i++){
    var node = textNodes[i];

    // font size
    var style = window.getComputedStyle(node.parentNode, null);
    var val = style.getPropertyValue("font-size");
    var ls = featureValueLists.fontsize;
    if (ls){ls.push(val);}
    else {featureValueLists.fontsize = [val];}

    // height and width
    var bbFeatures = ["right", "left", "width", "top", "bottom", "height"];
    var boundingBox = getTextNodeBoundingBox(node);
    for (var j = 0; j < bbFeatures.length; j++){
      var featureName = bbFeatures[j];
      var featureValue = boundingBox[featureName];
      var ls = featureValueLists[featureName];
      if (ls){ls.push(featureValue);}
      else {featureValueLists[featureName] = [featureValue];}
    }
  }

  for (key in featureValueLists){
    featureValueLists[key] = _.uniq(featureValueLists[key]).sort();
  }
  return featureValueLists;
}

function useRelationships(nodes, currentFeaturesName, nextFeaturesName){
  for (var i = 0; i < nodes.length; i++){
    var node = nodes[i];
    var newFeatures = {};
    var relationships = node.__relationships__;
    for (var relationship in relationships){
      var nodesWithRelationship = relationships[relationship];
      for (var j = 0; j < nodesWithRelationship.length; j++){
        var node = nodesWithRelationship[j];
        var nodeFeatures = node[currentFeaturesName];
        for (var featureName in nodeFeatures){
          // "above-above" features aren't interesting since all our relationships are currently transitive
          // if we add different relationships, may need to change this
          if (featureName.indexOf(relationship) === 0){continue;}
          var value = nodeFeatures[featureName];
          newFeatures[relationship+"-"+featureName] = value;
        }
      }
    }
    nodes[i][nextFeaturesName] = newFeatures;
  }
}

var textNodes;
function processTextNodes(){
  var unfilteredTextNodes = getTextNodesIn(document.body);
  // get rid of nodes that aren't actually displaying any text
  textNodes = [];
  for (var i = 0; i < unfilteredTextNodes.length; i++){
    var boundingBox = getTextNodeBoundingBox(unfilteredTextNodes[i]);
    if (boundingBox.height > 0) {textNodes.push(unfilteredTextNodes[i]);}
  }

  // get some info we're going to use to determine the features
  var pageWidth = $(window).width();
  var pageHeight = $(window).height();
  var featureValueLists = populateGlobalPageInfo(textNodes);

  // get the actual features for the nodes
  for (var i = 0; i < textNodes.length; i++){
    highlightNode(textNodes[i], i);
    getFeatures(textNodes[i], pageWidth, pageHeight, featureValueLists);
  }

  // find relationships between nodes
  for (var i = 0; i < textNodes.length; i++){
    findRelationships(textNodes, i);
  }

  useRelationships(textNodes, "__features__","__features1__");
  useRelationships(textNodes, "__features1__","__features2__");
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
    var isTarget = classify(net, common.makeFeatureVector(targetFeatures, textNodes[i].__features2__));
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
