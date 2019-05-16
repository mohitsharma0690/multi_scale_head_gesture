//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// label edge features
//
//	April 27, 2006

#ifndef LABEL_EDGE_FEATURES_H
#define LABEL_EDGE_FEATURES_H

#include "featuregenerator.h"


class LabelEdgeFeatures : public FeatureType
{
public:
	LabelEdgeFeatures ();

	void init(const DataSet& dataset, const Model& m);
	void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
	bool isEdgeFeatureType();

	void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
};

#endif 
