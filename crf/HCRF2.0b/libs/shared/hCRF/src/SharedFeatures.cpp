#include "SharedFeatures.h"

SharedFeatures::SharedFeatures():FeatureType()
{
	strFeatureTypeName = "Shared Features Type";
	featureTypeId = LATENT_LABEL_FEATURE_ID;
}

void SharedFeatures::init(const DataSet& dataset, const Model& m)
{
  	FeatureType::init(dataset,m);
	int nbStates = m.getNumberOfStates();
	int nbLabels = m.getNumberOfStateLabels();
	int nbSeqLabels = m.getNumberOfSequenceLabels();
	if(nbSeqLabels == 0){
		nbFeatures = nbStates*nbLabels;
	} else {
		throw HcrfBadModel("Shared Features does not support sequence labels");
	}
}

void SharedFeatures::getFeatures(featureVector& listFeatures, DataSequence* X, 
								 Model* m, int nodeIndex, int prevNodeIndex, 
								 int seqLabel)
{
/**
Seqlabel should always be zero as we do not support several sequence label. Also
note that this is an edge features between a label and an hidden states
**/
	if(prevNodeIndex == nodeIndex + X->length())
	{
		feature* pFeature;
		int nbStates = m->getNumberOfStates();
		int nbStatesLabel = m->getNumberOfStateLabels();

		for(int state = 0; state < nbStates; state++) {
			for(int label = 0; label < nbStatesLabel; label++) {
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset(seqLabel) + label + 
					state*nbStatesLabel;
				pFeature->globalId = getIdOffset() + label + 
					state*nbStatesLabel;
				pFeature->nodeIndex = nodeIndex;
				pFeature->nodeState = state;
				pFeature->prevNodeIndex = prevNodeIndex;
				pFeature->prevNodeState = label;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = 1.0;
			}
		}
	}
}

void SharedFeatures::getAllFeatures(featureVector& listFeatures, Model* m, 
									int)
{
	feature* pFeature;
	int nbStates = m->getNumberOfStates();
	int nbStatesLabel = m->getNumberOfStateLabels();
	for(int state = 0; state < nbStates; state++)
	{
		for(int label = 0; label < nbStatesLabel; label++)
		{
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset() + label + state*nbStatesLabel;
			pFeature->globalId = getIdOffset() + label + state*nbStatesLabel;
			pFeature->nodeIndex = featureTypeId;
			pFeature->nodeState = state;
			pFeature->prevNodeIndex = -1;
			pFeature->prevNodeState = label;
			pFeature->sequenceLabel = -1;
			pFeature->value = 1.0;
		}
	}
}

void SharedFeatures::computeFeatureMask(iMatrix& matFeatureMask, const Model& m)
{
	int nbSeqLabels = m.getNumberOfSequenceLabels();
	if (nbSeqLabels != 1)
		throw std::logic_error("SharedFeatures::computeFeatureMask should"
							   " only be used with model having 1 sequence"
							   " label");
	int firstOffset = idOffset;
	for(int j = 0; j < nbSeqLabels; j++)
	{
		int lastOffset = firstOffset + nbFeaturesPerLabel[j];
	
		for(int i = firstOffset; i < lastOffset; i++)
			matFeatureMask(i,j) = 1;
		firstOffset += nbFeaturesPerLabel[j];
	}
}



