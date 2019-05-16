//-------------------------------------------------------------
// Hidden Conditional Random Field Library - GradientLDCRF
// Component
//
//	June 5, 2006

#include "gradient.h"
using namespace std;

GradientLDCRF::GradientLDCRF(InferenceEngine* infEngine,
							 FeatureGenerator* featureGen):
Gradient(infEngine, featureGen)
{
}

double GradientLDCRF::computeGradient(dVector& vecGradient, Model* m,
									DataSequence* X)
{
	//compute beliefs
	Beliefs bel;
	Beliefs belMasked;
	pInfEngine->computeBeliefs(bel,pFeatureGen, X, m, true,-1, false);
	pInfEngine->computeBeliefs(belMasked,pFeatureGen, X, m, true,-1, true);
	//Check the size of vecGradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);

#if !defined(_VEC_FEATURES) && !defined(_OPENMP)
	featureVector* vecFeatures;
#endif
#if defined(_OPENMP)
	int ThreadID = omp_get_thread_num();
	if (ThreadID >= nbThreadsMP)
		ThreadID = 0;
#else
	int ThreadID = 0;
#endif
	//Loop over nodes to compute features and update the gradient
	for(int i = 0; i < X->length(); i++)
	{
		//Get nodes features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,i,-1);
		// Loop over features
		feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
		for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++, pFeature++)
#else
		vecFeatures = pFeatureGen->getFeatures(X,m,i,-1);
		// Loop over features
		feature* pFeature = vecFeatures->getPtr();
		for(int j = 0; j < vecFeatures->size(); j++, pFeature++)
#endif
		{
			//p(y_i=s|x)*f_k(i,s,x)
			vecGradient[pFeature->id] -= bel.belStates[i][pFeature->nodeState]*pFeature->value;
			vecGradient[pFeature->id] += belMasked.belStates[i][pFeature->nodeState]*pFeature->value;
		}
	}

	for(int i = 0; i < X->length()-1; i++) // Loop over all rows (the previous node index)
	{
		//Get nodes features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,i+1,i);
		// Loop over features
		feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
		for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++, pFeature++)
#else
		vecFeatures = pFeatureGen->getFeatures(X,m,i+1,i);
		// Loop over features
		feature* pFeature = vecFeatures->getPtr();
		for(int j = 0; j < vecFeatures->size(); j++, pFeature++)
#endif
		{
			//p(y_i=s1,y_j=s2|x)*f_k(i,j,s1,s2,x) is subtracted from the gradient 
			vecGradient[pFeature->id] -= bel.belEdges[i](pFeature->prevNodeState,pFeature->nodeState)*pFeature->value;
			vecGradient[pFeature->id] += belMasked.belEdges[i](pFeature->prevNodeState,pFeature->nodeState)*pFeature->value;
		}
	}

	//Return -log instead of log() [Moved to Gradient::ComputeGradient by LP]
	//vecGradient.negate();
	return bel.partition - belMasked.partition;
}

