//-------------------------------------------------------------
// Hidden Conditional Random Field Library - EvaluatorCRF
// Component
//
//	May 1st, 2006

#include "evaluator.h"
#include <assert.h>
using namespace std;

/////////////////////////////////////////////////////////////////////
// Evaluator CRF Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

EvaluatorCRF::EvaluatorCRF() : Evaluator()
{
}


EvaluatorCRF::EvaluatorCRF(InferenceEngine* infEngine, 
						   FeatureGenerator* featureGen)
  : Evaluator(infEngine, featureGen)
{
}

EvaluatorCRF::~EvaluatorCRF()
{
}

// *
// Public Methods
// *

double EvaluatorCRF::computeError(DataSequence* X, Model* m)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorCRF::computeError");
	}
	double phi;
	double partition;
//	#pragma omp critical
	{
		phi = pFeatureGen->evaluateLabels(X,m);
		partition = pInfEngine->computePartition(pFeatureGen, X, m);
	}
	// Returns -log() instead of log()	
	return partition - phi;
}
