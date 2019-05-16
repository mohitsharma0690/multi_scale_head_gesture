//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Window raw features
//
//	February 20, 2006

#ifndef WINDOW_RAW_FEATURES_H
#define WINDOW_RAW_FEATURES_H

#include "featuregenerator.h"


class WindowRawFeatures : public FeatureType
{
public:
	WindowRawFeatures(int windowSize = 0);

	virtual void init(const DataSet& dataset, const Model& m);
	virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	virtual bool isEdgeFeatureType();

	void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
private:
	int WindowSize;
};

#endif 
