//-------------------------------------------------------------
// Hidden Conditional Random Field Library - GradientCRF
// Component
//
//	February 2, 2006

#include "gradient.h"

GradientCRF::GradientCRF(InferenceEngine* infEngine, 
						 FeatureGenerator* featureGen) 
: Gradient(infEngine, featureGen)
{
}

double GradientCRF::computeGradient(dVector& vecGradient, Model* m, 
									DataSequence* X)
{
	//compute beliefs
	Beliefs bel;
	pInfEngine->computeBeliefs(bel,pFeatureGen, X, m, false);
	double phi = pFeatureGen->evaluateLabels(X,m);
	double partition = bel.partition;
	//Get adjency matrix
	uMatrix adjMat;
	m->getAdjacencyMatrix(adjMat, X);
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
		// Read the label for this state
		int s = X->getStateLabels(i);
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

			// If feature has same state label as the label from the
			// dataSequence, then add this to the gradient
			if(pFeature->nodeState == s)
				vecGradient[pFeature->id] += pFeature->value;
			//p(y_i=s|x)*f_k(i,s,x) is subtracted from the gradient 
			vecGradient[pFeature->id] -= bel.belStates[i][pFeature->nodeState]*pFeature->value;
		}
	}
	//Loop over edges to compute features and update the gradient
	int edgeIndex = 0;
	for(int row = 0; row < X->length(); row++) // Loop over all rows (the previous node index)
	{
		for(int col = row; col < X->length() ; col++) //Loop over all columns (the current node index)
		{
			if(adjMat(row,col) == 1)
			{
				int s1 = X->getStateLabels(row);
				int s2 = X->getStateLabels(col);

				//Get nodes features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
				pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,col,row);
				// Loop over features
				feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
				for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++, pFeature++)
#else
				vecFeatures = pFeatureGen->getFeatures(X,m,col,row);
				// Loop over features
				feature* pFeature = vecFeatures->getPtr();
				for(int j = 0; j < vecFeatures->size(); j++, pFeature++)
#endif
				{
					// ++ Forward edge ++
					// If edge feature has same state labels as the labels from the dataSequence, then add it to the gradient
					if(pFeature->nodeState == s2 && pFeature->prevNodeState == s1)
						vecGradient[pFeature->id] += pFeature->value;

					//p(y_i=s1,y_j=s2|x)*f_k(i,j,s1,s2,x) is subtracted from the gradient 
					vecGradient[pFeature->id] -= bel.belEdges[edgeIndex](pFeature->prevNodeState,pFeature->nodeState)*pFeature->value;
				}
				edgeIndex++;
			}
		}
	}
	//Return -log instead of log() [Moved to Gradient::ComputeGradient by LP]
//	vecGradient.negate();
	return partition-phi;
}

