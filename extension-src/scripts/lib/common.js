var common = {};

// targetFeatures is an array of string feature names representing the features we want to include in the feature vector
// nodeFeatures is a dictionary from the features a node has to the value (not all nodes have all features)
common.makeFeatureVector = function(targetFeatures, nodeFeatures){
	var featureVector = [];
	for (var i = 0; i < targetFeatures.length; i++){
		var feature = targetFeatures[i];
		if (feature in nodeFeatures && nodeFeatures[feature] === true){
			featureVector.push(1);
		}
		else if (feature in nodeFeatures){
			featureVector.push(nodeFeatures[feature]);
		}
		else {
			featureVector.push(0);
		}
	}
	return featureVector;
}

/**********************************************************************
 * Keeping track of features
 **********************************************************************/

function FeaturesDict(globalFeaturesLs) {
    this.dict = {};
    this.globalFeaturesLs = globalFeaturesLs;
}
 
FeaturesDict.prototype.add = function(name, val) {
    this.dict[name] = val;
    this.globalFeaturesLs[name] = true;
};

FeaturesDict.prototype.get = function(name) {
    return this.dict[name];
};

FeaturesDict.prototype.getDict = function() {
    return this.dict;
};