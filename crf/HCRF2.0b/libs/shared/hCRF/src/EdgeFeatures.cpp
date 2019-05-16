#include "EdgeFeatures.h"

using namespace std;


EdgeFeatures::EdgeFeatures():FeatureType()
{
	strFeatureTypeName = "Edge Feature Type";
	featureTypeId = EDGE_FEATURE_ID;
}

void EdgeFeatures::getFeatures(featureVector& listFeatures, DataSequence* X, 
							   Model* m, int nodeIndex, int prevNodeIndex, 
							   int seqLabel)
{
	// These features are only used for adjacent edge in the chain	
	int nbNodes = -1;
	
	if(X->getPrecomputedFeatures())
		nbNodes = X->getPrecomputedFeatures()->getWidth();
	else
		nbNodes = (int)X->getPrecomputedFeaturesSparse()->getWidth();
	
	if( ((prevNodeIndex == nodeIndex-1) || prevNodeIndex == nodeIndex-1 + nbNodes)
		&& (prevNodeIndex != -1))
	{
		feature* pFeature;
		int nbStateLabels = m->getNumberOfStates();
		for(int s1 = 0; s1 < nbStateLabels;s1++)
		{
			for(int s2 = 0; s2 < nbStateLabels;s2++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset(seqLabel) + s2 + s1*nbStateLabels ;
				pFeature->globalId = getIdOffset() + s2 + s1*nbStateLabels + 
					seqLabel*nbStateLabels*nbStateLabels ;
				pFeature->nodeIndex = nodeIndex;
				pFeature->nodeState = s2;
				pFeature->prevNodeIndex = prevNodeIndex;
				pFeature->prevNodeState = s1;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = 1.0f;
			}
		}
	}
}

void EdgeFeatures::getAllFeatures(featureVector& listFeatures, Model* m, 
								  int)
/* We dont need the number of raw features as the number of edge feature is
 * independant from the size of the windows
 */
{
	int nbStateLabels = m->getNumberOfStates();
	int nbSeqLabels = m->getNumberOfSequenceLabels();
	feature* pFeature;
	if(nbSeqLabels == 0)
		nbSeqLabels = 1;
	for(int seqLabel = 0; seqLabel < nbSeqLabels;seqLabel++)
	{
		for(int s1 = 0; s1 < nbStateLabels;s1++)
		{
			for(int s2 = 0; s2 < nbStateLabels;s2++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset() + s2 + s1*nbStateLabels + seqLabel*nbStateLabels*nbStateLabels ;
				pFeature->globalId = getIdOffset() + s2 + s1*nbStateLabels + seqLabel*nbStateLabels*nbStateLabels ;
				pFeature->nodeIndex = featureTypeId;
				pFeature->nodeState = s2;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = s1;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = 1.0f;
			}
		}
	}
}


void EdgeFeatures::init(const DataSet& dataset, const Model& m)
{
	FeatureType::init(dataset,m);
	int nbStateLabels = m.getNumberOfStates();
	int nbSeqLabels = m.getNumberOfSequenceLabels();

	if(nbSeqLabels == 0)
		nbFeatures = nbStateLabels*nbStateLabels;
	else
	{
		nbFeatures = nbStateLabels*nbStateLabels*nbSeqLabels;
		for(int i = 0; i < nbSeqLabels; i++)
			nbFeaturesPerLabel[i] = nbStateLabels*nbStateLabels;
	}
}

void EdgeFeatures::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
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

bool EdgeFeatures::isEdgeFeatureType()
{
	return true;
}
