#include "RawFeatures.h"

using namespace std;


RawFeatures::RawFeatures():FeatureType()
{
	strFeatureTypeName = "Raw Feature Type";
	featureTypeId = RAW_FEATURE_ID;
}


void RawFeatures::getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
	if(X->getPrecomputedFeatures() != NULL && prevNodeIndex == -1)
	{
		dMatrix * preFeatures = X->getPrecomputedFeatures();
		int nbFeatures = preFeatures->getHeight();
		feature* pFeature;
		int idState = 0;
		int nbStateLabels = m->getNumberOfStates();

		for(int s = 0; s < nbStateLabels; s++)
		{
			for(int f = 0; f < nbFeatures; f++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset(seqLabel) + f + idState*nbFeatures;
				pFeature->globalId = getIdOffset() + f + idState*nbFeatures;
				pFeature->nodeIndex = nodeIndex;
				pFeature->nodeState = s;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = preFeatures->getValue(f,nodeIndex); 
			}
			idState ++;
		}
	}
}

void RawFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int NbRawFeatures)
{
	int nbStateLabels = m->getNumberOfStates();
	feature* pFeature;

	for(int s = 0; s < nbStateLabels; s++)
	{
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



void RawFeatures::init(const DataSet& dataset, const Model& m)
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

bool RawFeatures::isEdgeFeatureType()
{
	return false;
}
