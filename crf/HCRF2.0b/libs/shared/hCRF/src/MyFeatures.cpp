#include "MyFeatures.h"



MyFeatures::MyFeatures():FeatureType()
{
	strFeatureTypeName = "My Feature Type";
	featureTypeId = MY_FEATURE_ID;
}

void MyFeatures::getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
	// If raw/precomputed features are available and the getFeatures() call is for state features
	if(X->getPrecomputedFeatures() != NULL && prevNodeIndex == -1)
	{
		dMatrix * preFeatures = X->getPrecomputedFeatures();
		int nbFeatures = preFeatures->getHeight();
		feature* pFeature;
		int nbStateLabels = m->getNumberOfStates();

		// For every possible state
		for(int s = 0; s < nbStateLabels; s++)
		{
			// For every features in the precomputed feature matrix
			for(int f = 0; f < nbFeatures; f++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset(seqLabel) + f + s*nbFeatures;
				pFeature->globalId = getIdOffset() + f + s*nbFeatures;
				pFeature->nodeIndex = nodeIndex;
				pFeature->nodeState = s;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = seqLabel;
				// Returns the square of the precomputed value
				pFeature->value = preFeatures->getValue(f,nodeIndex) * preFeatures->getValue(f,nodeIndex); // TODO: optimize
			}
		}
	}
}

void MyFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int NbRawFeatures)
{
	int nbStateLabels = m->getNumberOfStates();
	feature* pFeature;

	// For every possible state
	for(int s = 0; s < nbStateLabels; s++)
	{
		// For every features in the precomputed feature matrix
		for(int f = 0; f < NbRawFeatures; f++)
		{
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset() + f + s*NbRawFeatures;
			pFeature->globalId = getIdOffset() + f + s*NbRawFeatures;
			pFeature->nodeIndex = featureTypeId;
			pFeature->nodeState = s;
			pFeature->prevNodeIndex = -1;
			pFeature->prevNodeState = -1;
			pFeature->sequenceLabel = -1;
			pFeature->value = f; 
		}
	}
}


// Determine the number of features
void MyFeatures::init(const DataSet& dataset, const Model& m)
{
	FeatureType::init(dataset,m);
	if(dataset.size() > 0)
	{
		int nbStates = m.getNumberOfStates();
		int nbSeqLabels = m.getNumberOfSequenceLabels();
		int nbFeaturesPerStates = dataset.at(0)->getPrecomputedFeatures()->getHeight();

		nbFeatures = nbStates * nbFeaturesPerStates;
		for(int i = 0; i < nbSeqLabels; i++)
			nbFeaturesPerLabel[i] = nbFeatures;
	}
}

bool MyFeatures::isEdgeFeatureType()
{
	return false;
}
