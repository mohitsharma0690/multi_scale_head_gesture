//-------------------------------------------------------------
// Hidden Conditional Random Field Library - GradientHCRF
// Component
//
//	February 2, 2006

#include "gradient.h"

GradientHCRF::GradientHCRF(InferenceEngine* infEngine, 
						   FeatureGenerator* featureGen) 
  : Gradient(infEngine, featureGen)
{
}

double GradientHCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X)
{
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	int NumSeqLabels=m->getNumberOfSequenceLabels();
	//Get adjency matrix
	uMatrix adjMat;
	m->getAdjacencyMatrix(adjMat, X);
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);
	dVector Partition;
	Partition.resize(1,NumSeqLabels);
	std::vector<Beliefs> ConditionalBeliefs(NumSeqLabels);

	// Step 1 : Run Inference in each network to compute marginals conditioned on Y
	for(int i=0;i<NumSeqLabels;i++)
	{
		pInfEngine->computeBeliefs(ConditionalBeliefs[i],pFeatureGen, X, m, true,i);
		Partition[i] = ConditionalBeliefs[i].partition;
	}
	double f_value = Partition.logSumExp() - Partition[X->getSequenceLabel()];
	// Step 2: Compute expected values for feature nodes conditioned on Y
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
	double value;
	dMatrix CEValues;
	CEValues.resize(nbFeatures,NumSeqLabels);
	//Loop over nodes to compute features and update the gradient
	for(int j=0;j<NumSeqLabels;j++) {//For every labels
		for(int i = 0; i < X->length(); i++) {//For every nodes
#if defined(_VEC_FEATURES) || defined(_OPENMP)
			pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,i,-1,j);
			// Loop over features
			feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
			for(int k = 0; k < vecFeaturesMP[ThreadID].size(); k++, pFeature++)
#else
		vecFeatures =pFeatureGen->getFeatures(X,m,i,-1,j);
		  // Loop over features
		  feature* pFeature = vecFeatures->getPtr();
		  for(int k = 0; k < vecFeatures->size(); k++, pFeature++)
#endif
			{   
                //p(s_i=s|x,Y) * f_k(i,s,x,y) 
				value=ConditionalBeliefs[j].belStates[i][pFeature->nodeState] * pFeature->value;
				CEValues.setValue(j,pFeature->globalId, CEValues(j,pFeature->globalId) + value); // one row for each Y
			}// end for every feature
		}// end for every node
	}// end for ever Sequence Label
	// Step 3: Compute expected values for edge features conditioned on Y
	//Loop over edges to compute features and update the gradient
	for(int j=0;j<NumSeqLabels;j++){
		int edgeIndex = 0;
	    for(int row = 0; row < X->length(); row++){
			// Loop over all rows (the previous node index)
		    for(int col = row; col < X->length() ; col++){
				//Loop over all columns (the current node index)
				if(adjMat(row,col) == 1) {
					//Get nodes features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
					pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,col,row,j);
					// Loop over features
					feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
					for(int k = 0; k < vecFeaturesMP[ThreadID].size(); k++, pFeature++)
#else
					vecFeatures = pFeatureGen->getFeatures(X,m,col,row,j);
					// Loop over features
					feature* pFeature = vecFeatures->getPtr();
					for(int k = 0; k < vecFeatures->size(); k++, pFeature++)
#endif
					{
                        //p(y_i=s1,y_j=s2|x,Y)*f_k(i,j,s1,s2,x,y) 
						value=ConditionalBeliefs[j].belEdges[edgeIndex](pFeature->prevNodeState,pFeature->nodeState) * pFeature->value;
						CEValues.setValue(j,pFeature->globalId, CEValues(j,pFeature->globalId) + value);
					}
					edgeIndex++;
				}
			}
		}
	}
	// Step 4: Compute Joint Expected Values
	dVector JointEValues;
	JointEValues.resize(1,nbFeatures);
	JointEValues.set(0);
	dVector rowJ;
	rowJ.resize(1,nbFeatures);
	dVector GradientVector;
	double sumZLog=Partition.logSumExp();
	for (int j=0;j<NumSeqLabels;j++)
	 {
         CEValues.getRow(j, rowJ);
		 rowJ.multiply(exp(Partition.getValue(j)-sumZLog));
		 JointEValues.add(rowJ);
	 }
  // Step 5 Compute Gradient as Exi[i,*,*] -Exi[*,*,*], that is difference
  // between expected values conditioned on Sequence Labels and Joint expected
  // values
	 CEValues.getRow(X->getSequenceLabel(), rowJ); // rowJ=Expected value
												   // conditioned on Sequence
												   // label Y
	// [Negation moved to Gradient::ComputeGradient by LP]
//	 rowJ.negate(); 
	 JointEValues.negate();
	 rowJ.add(JointEValues);
	 vecGradient.add(rowJ);
	 return f_value;
}

