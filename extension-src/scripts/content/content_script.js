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
utilities.listenForMessage("mainpanel", "content", "getTrainingDataWithFeatureSet", handleNewFeatureSet);
utilities.listenForMessage("mainpanel", "content", "currentLabel", handleCurrentLabel);

utilities.sendMessage("content", "background", "requestTabID", {});

var globalFeaturesLs = {};

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
  newDiv.attr('class', "highlight-node");
  newDiv.css('width', boundingBox.width);
  newDiv.css('height', boundingBox.height);
  newDiv.css('top', boundingBox.top);
  newDiv.css('left', boundingBox.left);
  newDiv.css('position', 'absolute');
  newDiv.css('z-index', 1000);
  newDiv.css('background-color', '#CCFFCC');
  newDiv.css('opacity', .5);
  $(document.body).append(newDiv);
  newDiv.hover(function(){newDiv.css('opacity', .8);},function(){newDiv.css('opacity', .5);});

  return newDiv;
}

function removeHighlightNodes(){
  $(".highlight-node").remove();
}

var thisPageHasBeenLabeledByHand = false;

function processLabelingClick(textBox, highlightNode){
  /*
  node.__label__ = true;
  if (!thisPageHasBeenLabeledByHand){
    // we're only just labeling this by hand, so we might have some old guesses on here.  let's get rid of them
    for (var i = 0; i < textNodes.length; i++){ if (!textNodes[i].__label__){$("#highlight-"+i).css('background-color', '#00FF00'); }}
  }
  thisPageHasBeenLabeledByHand = true;
  */

  thisPageHasBeenLabeledByHand = true;
  
  highlightNode.css("background-color", currentColor);
  textBox.__features__.label = currentLabel;

  utilities.sendMessage("content", "mainpanel", "onePageDataset", {featureDicts: _.map(textNodes, function(x){return x.__features__})});
}

function makeFeatureVectorLabelPairs(nodeList, targetFeatures){
  var pairs = [];
  for (var i = 0; i < nodeList.length; i++){
    var featureVec = common.makeFeatureVector(targetFeatures, nodeList[i].__features2__.getDict());
    pairs.push([featureVec, nodeList[i].__label__]);
  }
  return pairs;
}

function getContinuousFeature(features, featureName, featureValue, featureValueList){
  features.add("has"+featureName, featureValue); // continuous
  var loc = featureValueList.indexOf(featureValue);
  features.add("has"+featureName+"smallest", loc); // continous
  features.add("has"+featureName+"largest", featureValueList.length - loc); // continous
}

function getFeaturesSimple(node){
  var features = {};

  // text
  text = node.nodeValue;
  features.text = text;

  // bounding box features  
  var bbFeatures = ["left", "top", "right", "bottom"];
  var boundingBox = getTextNodeBoundingBox(node);
  for (var i = 0; i < bbFeatures.length; i++){
    var featureName = bbFeatures[i];
    var featureValue = boundingBox[featureName];
    features[featureName] = featureValue;
  }

  // css/style features
  var styleFeatures = ["font-family", "font-style", "font-weight", "color", "background-color", "font-size"];
  var style = window.getComputedStyle(node.parentNode, null);
  for (var i = 0; i < styleFeatures.length; i++){
    var featureName = styleFeatures[i];
    var featureValue = style.getPropertyValue(featureName);
    if (featureName == "font-size"){
      featureValue = parseInt(featureValue);
    }
    features[featureName] = featureValue;
  }

  features.label = "nolabel";

  node.__features__ = features;
}

function getFeatures(node, pageWidth, pageHeight, featureValueLists){
  var features = new FeaturesDict(globalFeaturesLs);

  // text
  text = node.nodeValue;
  words = text.trim().toLowerCase().split(/[\s\.,\-\/\#\!\$%\^&\*\;\:\{\}=\-\_\`\~\(\)"]+/g);
  uniqueWords = _.uniq(words);
  for (var i = 0; i < uniqueWords.length; i++){
    features.add("hasword-"+uniqueWords[i], true); // discrete
  }

  // bounding box features
  var bbFeaturesWidth = ["right", "left", "width"];
  var bbFeaturesHeight = ["top", "bottom", "height"];
  var boundingBox = getTextNodeBoundingBox(node);
  for (var i = 0; i < bbFeaturesWidth.length; i++){
    var featureName = bbFeaturesWidth[i];
    var featureValue = boundingBox[featureName];
    getContinuousFeature(features, featureName, featureValue, featureValueLists[featureName]);
    features.add("haspercent"+featureName, boundingBox[featureName]/pageWidth); // continous
  }
  for (var i = 0; i < bbFeaturesHeight.length; i++){
    var featureName = bbFeaturesHeight[i];
    var featureValue = boundingBox[featureName];
    getContinuousFeature(features, featureName, featureValue, featureValueLists[featureName]);
    features.add("haspercent"+featureName, boundingBox[featureName]/pageHeight); // continous
  }

  // css/style features
  var styleFeatures = ["font-family", "font-style", "font-weight", "color", "background-color"];
  var style = window.getComputedStyle(node.parentNode, null);
  for (var i = 0; i < styleFeatures.length; i++){
    var featureName = styleFeatures[i];
    features.add("has"+featureName+"-"+style.getPropertyValue(featureName), true); // discrete
  }
  // fontsize is the one continous feature in the css ones
  var fontSize = parseInt(style.getPropertyValue("font-size"));
  getContinuousFeature(features, "fontsize", fontSize, featureValueLists.fontsize);

  node.__features__ = features;
  node.__label__ = false;
}

function above(n1,n2){
  return n1.__features__.get("hasbottom") <= n2.__features__.get("hastop");
}

function leftOf(n1,n2){
  return n1.__features__.get("hasright") <= n2.__features__.get("hasleft");
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
    var cNode = nodes[i];

    if (above(node,cNode) && !(leftOf(node,cNode) || leftOf(cNode, node))) {node.__relationships__.above.push(cNode);}
    if (leftOf(node, cNode) && !(above(node,cNode) || above(cNode, node))) {node.__relationships__.leftof.push(cNode);}
    if (above(cNode,node) && !(leftOf(node,cNode) || leftOf(cNode, node))) {node.__relationships__.below.push(cNode);}
    if (leftOf(cNode, node) && !(above(node,cNode) || above(cNode, node))) {node.__relationships__.rightof.push(cNode);}

    if (cNode.__features__.get("hasleft") == node.__features__.get("hasleft")) {node.__relationships__.sameleft.push(cNode);}
    if (cNode.__features__.get("hastop") == node.__features__.get("hastop")) {node.__relationships__.sametop.push(cNode);}
    if (cNode.__features__.get("hasright") == node.__features__.get("hasright")) {node.__relationships__.sameright.push(cNode);}
    if (cNode.__features__.get("hasbottom") == node.__features__.get("hasbottom")) {node.__relationships__.samebottom.push(cNode);}
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

  var oneNodeRelationships = function(i){
    //console.log(i, nodes, currentFeaturesName, nextFeaturesName);
    var node = nodes[i];
    var newFeatures = new FeaturesDict(globalFeaturesLs);
    var relationships = node.__relationships__;
    for (var relationship in relationships){
      var nodesWithRelationship = relationships[relationship];
      for (var j = 0; j < nodesWithRelationship.length; j++){
        var node = nodesWithRelationship[j];
        var nodeFeatures = node[currentFeaturesName];
        for (var featureName in nodeFeatures.getDict()){
          // "above-above" features aren't interesting since all our relationships are currently transitive
          // if we add different relationships, may need to change this
          if (featureName.indexOf(relationship) === 0){continue;}
          var value = nodeFeatures.get(featureName);
          newFeatures.add(relationship+"-"+featureName, value);
        }
      }
    }
    nodes[i][nextFeaturesName] = newFeatures;
  }

  for (var i = 0; i < nodes.length; i++){
    (function(){
      var x = i;
      setTimeout(function(){oneNodeRelationships(x);}, 0);
    })(); // simulating block scope
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

  //useRelationships(textNodes, "__features__","__features1__");
  //useRelationships(textNodes, "__features1__","__features2__");
}

function processTextNodesSimple(){
  var unfilteredTextNodes = getTextNodesIn(document.body);
  // get rid of nodes that aren't actually displaying any text
  textNodes = [];
  for (var i = 0; i < unfilteredTextNodes.length; i++){
    var boundingBox = getTextNodeBoundingBox(unfilteredTextNodes[i]);
    if (boundingBox.height > 0) {textNodes.push(unfilteredTextNodes[i]);}
  }

  removeHighlightNodes();

  // get the actual features for the nodes
  for (var i = 0; i < textNodes.length; i++){
    var hNode = highlightNode(textNodes[i], i);
    textNodes[i].__highlightNode__ = hNode;
    getFeaturesSimple(textNodes[i]);
  }
}

function keepProcessingTextNodesSimpleUntilWeGetALabel(){
  // we'll use the user starting labeling as an indication that we now have the right set of textboxes
  if (!thisPageHasBeenLabeledByHand){
    processTextNodesSimple();
    setTimeout(keepProcessingTextNodesSimpleUntilWeGetALabel,0);
  }
}

$(function(){setTimeout(processTextNodesSimple, 4000);});

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
    var isTarget = classify(net, common.makeFeatureVector(targetFeatures, textNodes[i].__features2__.getDict()));
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

// data has form {targetFeatures:chosenFeatures}
function handleNewFeatureSet(data){
  if (!thisPageHasBeenLabeledByHand){
    return; // only send training data from pages that have been hand labeled
  }

  var trainingData = makeFeatureVectorLabelPairs(textNodes, data.targetFeatures);
  utilities.sendMessage("content", "background", "newTrainingDataPairs", {pairs:trainingData});
}


var currentLabel = "";
var currentColor = "";
function handleCurrentLabel(data){
  currentLabel = data.currentLabel;
  currentColor = data.currentColor;
}

function intersectRect(r1, r2) {
  return !(r2.left > r1.right || 
           r2.right < r1.left || 
           r2.top > r1.bottom ||
           r2.bottom < r1.top);
}

var downX;
var downY;
function handleMousedown(e){
  downX = e.pageX;
  downY = e.pageY;
}

function handleMouseup(e){
  var features = {};
  if (downX < e.pageX){
    features.left = downX;
    features.right = e.pageX;
  }
  else {
    features.left = e.pageX;
    features.right = downX;
  }
  if (downY < e.pageY){
    features.top = downY;
    features.bottom = e.pageY;
  }
  else {
    features.top = e.pageY;
    features.bottom = downY;
  }

  for (var i = 0; i < textNodes.length; i++){
    var textBox = textNodes[i];
    if (intersectRect(features, textBox.__features__)) {
      processLabelingClick(textBox, textBox.__highlightNode__);
    }
  }
}

function setup(){
  $("*").mousedown(handleMousedown);
  $("*").mouseup(handleMouseup);

  var b = $("*");
  b.css("-webkit-touch-callout", "none");
  b.css("-webkit-user-select", "none");
  b.css("-khtml-user-select", "none");
  b.css("-moz-user-select", "none");
  b.css("-ms-user-select", "none");
  b.css("user-select", "none");
}
$(setup);

