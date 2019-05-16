//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Inference Engine
//
//	February 2, 2006

#include "inferenceengine.h"
#ifdef _OPENMP
#include <omp.h>
#endif
//-------------------------------------------------------------
// InferenceEngineBP Class
//-------------------------------------------------------------

//*
// Constructor and Destructor
//*

InferenceEngine::InferenceEngine()
{
	nbThreadsMP = 1;
	vecFeaturesMP = new featureVector[1];

}

InferenceEngine::~InferenceEngine()
{
	if(vecFeaturesMP)
	{
		delete []vecFeaturesMP;
		vecFeaturesMP = 0;
		nbThreadsMP = 0;
	}
}

int InferenceEngine::CountEdges(const uMatrix& AdjacencyMatrix, int nbNodes) const
{
    int NumEdges=0;
    // We only scan the upper triangle (Assumption is that Adjacency matrix is
    // symmetric)
	for(int i = 0 ; i < nbNodes ; i++)
	{
	    for(int j = i+1 ; j < nbNodes ; j++)
		{
#ifdef _DEBUG
			if (AdjacencyMatrix.getValue(i,j) == 1)
			{
			  NumEdges += 1;
			}
#else
		  // We assume that the adjacency matrix contains only 0 and 1
		  NumEdges += AdjacencyMatrix.getValue(i,j);
#endif
		}
	}
	return NumEdges;
}

void InferenceEngine::setMaxNumberThreads(int maxThreads)
{
	if (nbThreadsMP < maxThreads)
	{
		if (vecFeaturesMP)
			delete []vecFeaturesMP;
		nbThreadsMP = maxThreads;
		vecFeaturesMP = new featureVector[nbThreadsMP];
	}

}


void InferenceEngine::computeLogMi(FeatureGenerator* fGen, Model* model,
								   DataSequence* X, int i, int seqLabel,
								   dMatrix& Mi_YY, dVector& Ri_Y,
								   bool takeExp, bool bUseStatePerNodes)
{
	// This function compute Ri_Y = Sum of features for every states at time i
	// and Mi_YY, the sum of feature for every transition.
	Mi_YY.set(0);
	Ri_Y.set(0);
	dVector* lambda=model->getWeights(seqLabel);

#if defined(_OPENMP)
	int ThreadID = omp_get_thread_num();
	if (ThreadID >= nbThreadsMP)
		ThreadID = 0;
#else
	int ThreadID = 0;
#endif

#if defined(_VEC_FEATURES) || defined(_OPENMP)
	fGen->getFeatures(vecFeaturesMP[ThreadID], X,model,i,-1,seqLabel);
	feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
	for (int j = 0; j < vecFeaturesMP[ThreadID].size() ; j++, pFeature++)
#else
	featureVector* vecFeature = fGen->getFeatures(X,model, i, -1 ,seqLabel);
	feature* pFeature = vecFeature->getPtr();
	for (int j = 0; j < vecFeature->size() ; j++, pFeature++)
#endif
	{
		int f = pFeature->id;
		int yp = pFeature->nodeState;
		double val = pFeature->value;
		double oldVal = Ri_Y.getValue(yp);
		Ri_Y.setValue(yp,oldVal+(*lambda)[f]*val);
	}	

	if(i>0 ) {
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		fGen->getFeatures(vecFeaturesMP[ThreadID], X, model, i, i-1, seqLabel);
		pFeature = vecFeaturesMP[ThreadID].getPtr();
		for (int j = 0; j < vecFeaturesMP[ThreadID].size() ; j++, pFeature++) {
#else
		vecFeature = fGen->getFeatures(X,model,i,i-1,seqLabel);
		pFeature = vecFeature->getPtr();
		for (int j = 0; j < vecFeature->size() ; j++, pFeature++) {
#endif
			int f = pFeature->id;
			int yp = pFeature->nodeState;
			int yprev = pFeature->prevNodeState;
			double val = pFeature->value;
			Mi_YY.setValue(yprev, yp, Mi_YY.getValue(yprev,yp)+(*lambda)[f]*val);
		}
	}
	double maskValue = -INF_VALUE;
	if (takeExp) {
		Ri_Y.eltExp();
		Mi_YY.eltExp();
		maskValue = 0;
	}
	if(bUseStatePerNodes) {
		// This take into account the sharing of the state.
		iMatrix* pStatesPerNodes = model->getStateMatrix(X);
		for(int s = 0; s < Ri_Y.getLength(); s++) {
			if(pStatesPerNodes->getValue(s,i) == 0)
				Ri_Y.setValue(s,maskValue);
		}
	}
}


void InferenceEngine::LogMultiply(dMatrix& Potentials, dVector& Beli, 
								  dVector& LogAB)
{
	// The output is stored in LogAB. . This function compute
	// log(exp(P) * exp(B)).
	// It is safe to have Beli and LogAB reference the same variable
	//TODO: Potential may not be correctly initilized when we enter
	int row;
	int col;
	dMatrix temp;
	double m1;
	double m2;
	double sub;
	temp.create(Potentials.getWidth(),Potentials.getHeight());
	for(row=0;row<Potentials.getHeight();row++){
		for(col=0;col<Potentials.getWidth();col++){
			temp.setValue(row, col, Potentials.getValue(row,col) + 
						  Beli.getValue(col));
		}
	}
	// WARNING: Beli should not be used after this point as it can be the same
	// as LogAB and thus may be overwritten by the results.
	for(row=0;row<Potentials.getHeight();row++){
		LogAB.setValue(row,temp.getValue(row,0));
		for(col=1;col<temp.getWidth();col++){
			if(LogAB.getValue(row) >= temp.getValue(row,col)){
				m1=LogAB.getValue(row);
				m2=temp.getValue(row,col);
			} else {
				m1=temp.getValue(row,col);
				m2=LogAB.getValue(row);
			}
			sub=m2-m1;
			LogAB.setValue(row,m1 + log(1 + exp(sub)));
		}

	}
}
