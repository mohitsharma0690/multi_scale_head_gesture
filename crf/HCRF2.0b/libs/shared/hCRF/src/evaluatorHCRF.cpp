//-------------------------------------------------------------
// Hidden Conditional Random Field Library - EvaluatorHCRF
// Component
//
//	May 1st, 2006

#include "evaluator.h"
#include <assert.h>
using namespace std;

/////////////////////////////////////////////////////////////////////
// Evaluator HCRF Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

EvaluatorHCRF::EvaluatorHCRF() : Evaluator()
{
}

EvaluatorHCRF::EvaluatorHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) : Evaluator(infEngine, featureGen)
{
}

EvaluatorHCRF::~EvaluatorHCRF()
{

}

// *
// Public Methods
// *

//computes OVERALL error of the datasequence
double EvaluatorHCRF::computeError(DataSequence* X, Model* m)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorHCRF::computeError");
	}

	double sumError = 0;
	double partition = 0;
	double dataSequenceGroundTruthLabel = X->getSequenceLabel();
	double dataSequenceGroundTruthPartition = -1000;
	double m1;
	double m2;
	double sub;
	int labelCounter = 0;

	//For each class label, compute the partition of the datasequence 
	//and add up all these partitions
	for(labelCounter = 0;labelCounter < m->getNumberOfSequenceLabels();labelCounter++)
	{
		partition = pInfEngine->computePartition(pFeatureGen, X, m,labelCounter);
		if(labelCounter == dataSequenceGroundTruthLabel){
			dataSequenceGroundTruthPartition = partition;
		}

		//Do a logSum
		if(labelCounter == 0)
			sumError = partition;
		else
		{
			if(sumError >= partition)
			{
				m1=sumError;
				m2=partition;
			} else
			{
				m1=partition;
				m2=sumError;
			}
			sub=m2-m1;
			sumError=m1 + log(1 + exp(sub));
		}
	}
	//return log(Sum_y' Z(y'|x)) - log(Z(y|x)) 
	return  sumError - dataSequenceGroundTruthPartition;
}


int EvaluatorHCRF::computeSequenceLabel(DataSequence* X, Model* m, dMatrix * probabilities)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorHCRF::computeSequenceLabel");
	}

	Beliefs bel;
	int labelCounter=0;
	int numberofSequenceLabels = m->getNumberOfSequenceLabels();
	double partition = 0;
	double bestScore = -100000000;
	int bestLabel = -1;
	if(probabilities)
		probabilities->create(1,numberofSequenceLabels);

	//Compute the State Labels i.e.
	for(labelCounter=0;labelCounter<numberofSequenceLabels;labelCounter++){

		partition = pInfEngine->computePartition(pFeatureGen, X, m,labelCounter);
		probabilities->setValue(labelCounter,0, partition);
		if(bestScore<partition){
			bestScore = partition;
			bestLabel = labelCounter;
		}
	}

	return bestLabel;
}

