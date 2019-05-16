//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// raw features
//
//	February 20, 2006

#ifndef RAW_FEATURES_SQUARE_H
#define RAW_FEATURES_SQUARE_H

#include "featuregenerator.h"


class RawFeaturesSquare : public FeatureType
{
public:
	RawFeaturesSquare();

	virtual void init(const DataSet& dataset, const Model& m);
	virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	virtual bool isEdgeFeatureType();

	void getAllFeatures(featureVector& listFeatures, Model* m, int NbRawFeatures);
};

#endif 
