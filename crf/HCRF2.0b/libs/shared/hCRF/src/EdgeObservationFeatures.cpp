#include "EdgeObservationFeatures.h"

using namespace std;


EdgeObservationFeatures::EdgeObservationFeatures():FeatureType()
{
	strFeatureTypeName = "Edge Observation Feature Type";
	featureTypeId = EDGE_OBSERVATION_FEATURE_ID;
	nbObservationFeatures = 4;
}

void EdgeObservationFeatures::getFeatures(featureVector& listFeatures, DataSequence* X, 
							   Model* m, int nodeIndex, int prevNodeIndex, 
							   int seqLabel)
{
	// These features are only used for adjacent edge in the chain	
	int nbNodes = -1;
	
	if(X->getPrecomputedFeatures())
	{
		nbNodes = X->getPrecomputedFeatures()->getWidth();
	
		if( ((prevNodeIndex == nodeIndex-1) || prevNodeIndex == nodeIndex-1 + nbNodes)
			&& (prevNodeIndex != -1))
		{
			dMatrix * preFeatures = X->getPrecomputedFeatures();
			feature* pFeature;
			int nbStateLabels = m->getNumberOfStates();
			for(int f1 = 0; f1 < nbObservationFeatures;f1++)
			{
				for(int f2 = 0; f2 < nbObservationFeatures;f2++)
				{
					if(preFeatures->getValue(f1,prevNodeIndex) && preFeatures->getValue(f2,nodeIndex))
					{
						for(int s1 = 0; s1 < nbStateLabels;s1++)
						{
							for(int s2 = 0; s2 < nbStateLabels;s2++)
							{
								pFeature = listFeatures.addElement();
								pFeature->id = getIdOffset(seqLabel) + f2+ s2*nbObservationFeatures + f1*nbStateLabels*nbObservationFeatures + s1*nbStateLabels*nbObservationFeatures*nbObservationFeatures;
								pFeature->globalId = getIdOffset() + f2+ s2*nbObservationFeatures + f1*nbStateLabels*nbObservationFeatures + s1*nbStateLabels*nbObservationFeatures*nbObservationFeatures+ seqLabel*nbStateLabels*nbStateLabels*nbObservationFeatures*nbObservationFeatures ;
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
			}
		}
	}
	else
	{
		nbNodes = (int)X->getPrecomputedFeaturesSparse()->getWidth();
		if( ((prevNodeIndex == nodeIndex-1) || prevNodeIndex == nodeIndex-1 + nbNodes)
			&& (prevNodeIndex != -1))
		{
			dMatrixSparse * preFeaturesSparse = X->getPrecomputedFeaturesSparse();		
			int nbFeaturesSparse = (int)preFeaturesSparse->getHeight();
			feature* pFeature;
			int nbStateLabels = m->getNumberOfStates();

			int irIndex1 = (int)preFeaturesSparse->getJc()->getValue(prevNodeIndex);
			size_t numElementsInCol1 = preFeaturesSparse->getJc()->getValue(prevNodeIndex+1) - irIndex1;
			int irIndex2 = (int)preFeaturesSparse->getJc()->getValue(nodeIndex);
			size_t numElementsInCol2 = preFeaturesSparse->getJc()->getValue(nodeIndex+1) - irIndex2;

			for(int i1 = 0; i1 < numElementsInCol1;i1++)
			{
				int f1 = (int)preFeaturesSparse->getIr()->getValue(irIndex1 + i1);
				if (f1 >= nbObservationFeatures)
					break;
				for(int i2 = 0; i2 < numElementsInCol2;i2++)
				{
					int f2 = (int)preFeaturesSparse->getIr()->getValue(irIndex2 + i2);
					if (f2 >= nbObservationFeatures)
						break;
					for(int s1 = 0; s1 < nbStateLabels;s1++)
					{
						for(int s2 = 0; s2 < nbStateLabels;s2++)
						{
							pFeature = listFeatures.addElement();
							pFeature->id = getIdOffset(seqLabel) + f2+ s2*nbObservationFeatures + f1*nbStateLabels*nbObservationFeatures + s1*nbStateLabels*nbObservationFeatures*nbObservationFeatures;
							pFeature->globalId = getIdOffset() + f2+ s2*nbObservationFeatures + f1*nbStateLabels*nbObservationFeatures + s1*nbStateLabels*nbObservationFeatures*nbObservationFeatures+ seqLabel*nbStateLabels*nbStateLabels*nbObservationFeatures*nbObservationFeatures ;
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
		}
	}

}

void EdgeObservationFeatures::getAllFeatures(featureVector& listFeatures, Model* m, 
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
				for(int f1 = 0; f1 < nbObservationFeatures;f1++)
				{
					for(int f2 = 0; f2 < nbObservationFeatures;f2++)
					{
						pFeature = listFeatures.addElement();
						pFeature->id = getIdOffset() + f2+ s2*nbObservationFeatures + f1*nbStateLabels*nbObservationFeatures + s1*nbStateLabels*nbObservationFeatures*nbObservationFeatures + seqLabel*nbStateLabels*nbStateLabels*nbObservationFeatures*nbObservationFeatures ;
						pFeature->globalId = getIdOffset() + + f2+ s2*nbObservationFeatures + f1*nbStateLabels*nbObservationFeatures + s1*nbStateLabels*nbObservationFeatures*nbObservationFeatures + seqLabel*nbStateLabels*nbStateLabels*nbObservationFeatures*nbObservationFeatures ;
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
	}
}


void EdgeObservationFeatures::init(const DataSet& dataset, const Model& m)
{
	FeatureType::init(dataset,m);
	int nbStateLabels = m.getNumberOfStates();
	int nbSeqLabels = m.getNumberOfSequenceLabels();

	if(nbSeqLabels == 0)
		nbFeatures = nbStateLabels*nbStateLabels*nbObservationFeatures*nbObservationFeatures;
	else
	{
		nbFeatures = nbStateLabels*nbStateLabels*nbSeqLabels*nbObservationFeatures*nbObservationFeatures;
		for(int i = 0; i < nbSeqLabels; i++)
			nbFeaturesPerLabel[i] = nbStateLabels*nbStateLabels*nbObservationFeatures*nbObservationFeatures;
	}
}

void EdgeObservationFeatures::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
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

bool EdgeObservationFeatures::isEdgeFeatureType()
{
	return true;
}
