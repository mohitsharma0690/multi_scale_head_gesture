//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// raw features
//
//	February 20, 2006

#ifndef FEATURES_ONE_H
#define FEATURES_ONE_H

#include "featuregenerator.h"


class FeaturesOne : public FeatureType
{
public:
	FeaturesOne();

	virtual void init(const DataSet& dataset, const Model& m);
	virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	virtual bool isEdgeFeatureType();

	void getAllFeatures(featureVector& listFeatures, Model* m, int NbRawFeatures);
};

#endif 
