//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// MyFeatures: Sample class on how to create your own features
//   Similar for Raw features but returns the square value of the
//   raw features instead of the raw feature value.
//
//	June 18, 2007

#ifndef MY_FEATURES_H
#define MY_FEATURES_H

#include "featuregenerator.h"

// We dont want our feature ID to conflict with futur version.
#define MY_FEATURE_ID LAST_FEATURE_ID+1

class MyFeatures : public FeatureType
{
public:
	MyFeatures();

	// Called once to initialize this type of features
	virtual void init(const DataSet& dataset, const Model& m);
	// Called for every sample of a data sequence. Returns the features (square of the raw feature) for every states for that sample.
	virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	virtual bool isEdgeFeatureType();

	// Utility function: returns all possible features that this class can produce
	void getAllFeatures(featureVector& listFeatures, Model* m, int NbRawFeatures);
};

#endif 
