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

function processLabelingClick(node){
  node.__label__ = true;
  console.log(textNodes);
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





