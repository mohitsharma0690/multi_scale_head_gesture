//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// edge features
//
//	February 20, 2006

#ifndef EDGE_OBSERVATION_FEATURES_H
#define EDGE_OBSERVATION_FEATURES_H

#include "featuregenerator.h"

class EdgeObservationFeatures : public FeatureType
{
  public:
   EdgeObservationFeatures();
   void init(const DataSet& dataset, const Model& m);
   void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m,
                    int nodeIndex, int prevNodeIndex, int seqLabel = -1);
   void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
   bool isEdgeFeatureType();
   void getAllFeatures(featureVector& listFeatures, Model* m,
                       int nbRawFeatures);
protected:
	int nbObservationFeatures;
};

#endif
