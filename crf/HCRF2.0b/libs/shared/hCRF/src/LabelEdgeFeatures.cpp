#include "LabelEdgeFeatures.h"

using namespace std;


LabelEdgeFeatures::LabelEdgeFeatures():FeatureType()
{
	strFeatureTypeName = "Label Edge Feature Type";
	featureTypeId = LABEL_EDGE_FEATURE_ID;
}


void LabelEdgeFeatures::getFeatures(featureVector& listFeatures, DataSequence*, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
	if(seqLabel != -1 && m->getNumberOfSequenceLabels() > 0 && prevNodeIndex == -1)
	{
		feature* pFeature;
		int nbStateLabels = m->getNumberOfStates();
		for(int s = 0; s < nbStateLabels; s++)
		{
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset(seqLabel) + s;
			pFeature->globalId = getIdOffset() + s + seqLabel*nbStateLabels ;
			pFeature->nodeIndex = nodeIndex;
			pFeature->nodeState = s;
			pFeature->prevNodeIndex = -1;
			pFeature->prevNodeState = -1;
			pFeature->sequenceLabel = seqLabel;
			pFeature->value = 1.0f;
		}
	}
}

void LabelEdgeFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int)
{
	// TODO: Should we test for statesPerLabel matrix for authorized features.
	if(m->getNumberOfSequenceLabels() > 0)
	{
		int nbStateLabels = m->getNumberOfStates();
		int nbSeqLabels = m->getNumberOfSequenceLabels();
		feature* pFeature;
		for(int seqLabel = 0; seqLabel < nbSeqLabels; seqLabel++)
		{
			for(int s = 0; s < nbStateLabels; s++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset() + s + seqLabel*nbStateLabels ;
				pFeature->globalId = getIdOffset() + s + seqLabel*nbStateLabels ;
				pFeature->nodeIndex = featureTypeId;
				pFeature->nodeState = s;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = 1.0f;
			}
		}
	}
}


void LabelEdgeFeatures::init(const DataSet& dataset, const Model& m)
{
	FeatureType::init(dataset,m);
	int nbStateLabels = m.getNumberOfStates();
	int nbSeqLabels = m.getNumberOfSequenceLabels();

	if(nbSeqLabels == 0)
		nbFeatures = 0;
	else
	{
		nbFeatures = nbStateLabels*nbSeqLabels;
		for(int i = 0; i < nbSeqLabels; i++)
			nbFeaturesPerLabel[i] = nbStateLabels;
	}
}

void LabelEdgeFeatures::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
{
	int nbLabels = m.getNumberOfSequenceLabels();
	int firstOffset = idOffset;

	for(int j = 0; j < nbLabels; j++)
	{
		int lastOffset = firstOffset + nbFeaturesPerLabel[j];
	
		for(int i = firstOffset; i < lastOffset; i++)
			matFeautureMask(i,j) = 1;

		firstOffset += nbFeaturesPerLabel[j];
	}
}

bool LabelEdgeFeatures::isEdgeFeatureType()
{
	// This is surprising. Label edge is no considered an edge features.
	return false;
}
