var all_script_results = [];
function setUp(){

  //messages received by this component
  //utilities.listenForMessage("content", "mainpanel", "selectorAndListData", processSelectorAndListData);
  
  //messages sent by this component
  //utilities.sendMessage("mainpanel", "content", "startProcessingList", "");


  var urls = ["http://www.cs.berkeley.edu/~schasins/#/resume","https://www.linkedin.com/pub/fanny-zhao/31/4aa/853", "http://www.indeed.com/r/Robert-DeKoch/8e4112cb91465768"];
  for (var i = 0; i < urls.length; i++){
    chrome.tabs.create({ url: urls[i] });
  }


}

$(setUp);