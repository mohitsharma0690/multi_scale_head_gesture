//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Evaluator
// Component
//
//	January 30, 2006

#include "evaluator.h"
#include <assert.h>
using namespace std;

/////////////////////////////////////////////////////////////////////
// Evaluator Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

Evaluator::Evaluator(): pInfEngine(NULL), pFeatureGen(NULL)
{
}

Evaluator::Evaluator(InferenceEngine* infEngine, FeatureGenerator* featureGen):
	pInfEngine(infEngine), pFeatureGen(featureGen)
{
}

Evaluator::~Evaluator()
{
	pInfEngine = NULL;
	pFeatureGen = NULL;
}

Evaluator::Evaluator(const Evaluator& other)
	:pInfEngine(other.pInfEngine), pFeatureGen(other.pFeatureGen)
{
}

Evaluator& Evaluator::operator=(const Evaluator& other)
{
	pInfEngine = other.pInfEngine;
	pFeatureGen = other.pFeatureGen;
	return *this;
}
// *
// Public Methods
// *

void Evaluator::init(InferenceEngine* infEngine, FeatureGenerator* featureGen)
{
	pInfEngine = infEngine;
	pFeatureGen = featureGen;
}

double Evaluator::computeError(DataSet* X, Model* m)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In Evaluator::computeError(DataSet*, model*)");
	}
	double error = 0;
	int TID = 0;
	int NumIters = (int)X->size();
	// Initialize the buffers (vecFeaturesMP) for each thread
#ifdef _OPENMP
	pInfEngine->setMaxNumberThreads(omp_get_max_threads());
	pFeatureGen->setMaxNumberThreads(omp_get_max_threads());
#endif

#pragma omp parallel shared(std::cout, X, m, NumIters, error)	\
	private(TID) \
	default(none)
	{
#ifdef _OPENMP
		TID = omp_get_thread_num();
#else
		TID = 0;
#endif
#pragma omp for reduction(+:error)
		for(int i = 0; i<NumIters; i++)
		{
			if (m->getDebugLevel()>=4)
			{
#pragma omp critical(output)
				cout << "Thread "<<TID<<" Computing error for sequence " << i << " out of " 
					 << (int)X->size() << " (Size: " << X->at(i)->length() 
					 << ")" << endl;
			}
			error += this->computeError(X->at(i), m) * X->at(i)->getWeightSequence();
		}
	}
	//error = error/X->size(); // Make the influence of regularization factor to be independent of the size of training data  
	if(m->getRegL2Sigma() != 0.0f)
	{
		double weightNorm = m->getWeights()->l2Norm(false);
		error += weightNorm / (2.0*m->getRegL2Sigma()*m->getRegL2Sigma());
	}
	return error;
}


void Evaluator::computeStateLabels(DataSequence* X, Model* m, 
								   iVector* vecStateLabels, 
								   dMatrix * probabilities)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In Evaluator::computeStateLabels");
	}

	Beliefs bel;
	pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, 0);
	computeLabels(bel,vecStateLabels,probabilities);
}

int Evaluator::computeSequenceLabel(DataSequence* , Model* , 
									dMatrix * )
{
	/* This is only valide for model that have a sequence label, such as HCRF */
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In Evaluator::computeSequenceLabel");
	}
	// To be implemented
	throw HcrfNotImplemented("Evaluator::computeSequenceLabel");
}

// *
// Private Methods
// *

void Evaluator::computeLabels(Beliefs& bel, iVector* vecStateLabels, 
							  dMatrix * probabilities)
{
	int nbNodes = (int)bel.belStates.size();
	int nbStates = 0;
	if(nbNodes > 0)
		nbStates = bel.belStates[0].getLength();

	vecStateLabels->create(nbNodes);
	if(probabilities)
		probabilities->create(nbNodes,nbStates);

	for(int n = 0; n<nbNodes; n++) 
	{
		// find max value
		vecStateLabels->setValue(n, 0);
		double MaxBel = bel.belStates[n][0];
		probabilities->setValue(0,n,bel.belStates[n][0]);
		for (int s = 1; s<nbStates; s++) {
			if(probabilities)
				probabilities->setValue(s,n,bel.belStates[n][s]);
			if(MaxBel < bel.belStates[n][s]) {
				vecStateLabels->setValue(n, s);
				MaxBel = bel.belStates[n][s];
			}
		}
	}
}
