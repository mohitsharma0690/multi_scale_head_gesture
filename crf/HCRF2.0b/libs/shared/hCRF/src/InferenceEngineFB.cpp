#include "inferenceengine.h"
#include <assert.h>

int InferenceEngineFB::forwardBeliefLog(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob){
	int NSTATES = model->getNumberOfStates();
	int NNODES = dataSequence->length();
	int pos = dataSequence->getPosition();
	int windowSize = dataSequence->getWindowSize();
	
	// Compute Mi_YY, in our case Mi_YY are the same for all positions
	dMatrix Mi_YY(NSTATES,NSTATES);	
	dVector Ri_Y(NSTATES);	
	computeLogMi(fGen, model, dataSequence, 1, -1, Mi_YY, Ri_Y, false, false);
	//
	dMatrix Mi_YY2(NSTATES,NSTATES);	
	dVector tmp_Y(NSTATES);
	// Update Alpha
	dVector* alpha;
	if(dataSequence->getAlpha() == 0){				
		dataSequence->initializeAlpha(NSTATES);
		alpha = dataSequence->getAlpha();
		computeLogMi(fGen, model, dataSequence, (pos+windowSize) % NNODES, -1, Mi_YY2, Ri_Y, false, false);
		alpha->set(Ri_Y);
	}
	else{
		computeLogMi(fGen, model, dataSequence, (pos+windowSize) % NNODES, -1, Mi_YY2, Ri_Y, false, false);
		alpha = dataSequence->getAlpha();
		tmp_Y.set(*alpha);
		Mi_YY.transpose();
		LogMultiply(Mi_YY, tmp_Y, *alpha);
		alpha->add(Ri_Y);
	}
				
	// Calculate beta for node in pos
	dVector beta(NSTATES);		
	beta.set(0);
	for(int i=pos+NNODES-1; i>pos+windowSize; i--)
	{
		int index = i%NNODES;
		computeLogMi(fGen, model, dataSequence, index, -1, Mi_YY2, Ri_Y, false, false);
		tmp_Y.set(beta);
		tmp_Y.add(Ri_Y);
		LogMultiply(Mi_YY,tmp_Y,beta);
	}		
	
	// Calculate probability distribution
	prob->set(*alpha);
	prob->add(beta);
	double LZx = prob->logSumExp();
	prob->add(-LZx);
	prob->eltExp();
	
	return 1;
}

double InferenceEngineFB::computePartition(FeatureGenerator* fGen,
											 DataSequence* X, Model* model,
											 int seqLabel,
											 bool bUseStatePerNodes)
{
	Beliefs bel;
	computeBeliefsLog(bel, fGen,X, model, true,seqLabel, bUseStatePerNodes);
	return bel.partition;
}

// Inference Functions
void InferenceEngineFB::computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
									  DataSequence* X, Model* model,
									  int bComputePartition,
									  int seqLabel, bool bUseStatePerNodes)
{
	computeBeliefsLog(bel, fGen,X, model, bComputePartition, seqLabel, bUseStatePerNodes);
}


void InferenceEngineFB::computeBeliefsLinear(Beliefs& bel, FeatureGenerator* fGen,
									  DataSequence* X, Model* model,
									  int, int seqLabel, bool bUseStatePerNodes)
{
	if(model->getAdjacencyMatType()!=CHAIN){
		throw HcrfBadModel("InferenceEngineFB need a model based on a Chain");
	}
	int NNODES=X->length();
	int NSTATES = model->getNumberOfStates();
	bel.belStates.resize(NNODES);
	for(int i=0;i<NNODES;i++)
	{
		bel.belStates[i].create(NSTATES);
		bel.belStates[i].set(0);
	}
	int NEDGES = NNODES-1;
	bel.belEdges.resize(NEDGES);
	for(int i=0;i<NEDGES;i++)
	{
		bel.belEdges[i].create(NSTATES,NSTATES);
		bel.belEdges[i].set(0);
	}
	dMatrix Mi_YY (NSTATES,NSTATES);
	dVector Ri_Y (NSTATES);
	dVector alpha_Y(NSTATES);
	dVector newAlpha_Y(NSTATES);
	dVector tmp_Y(NSTATES);
	alpha_Y.set(1);
	bel.belStates[NNODES-1].set(1.0);
	for (int i = NNODES-1; i > 0; i--)
	{
		// compute the Mi matrix
		computeLogMi(fGen,model,X,i,seqLabel,Mi_YY,Ri_Y,true,
					 bUseStatePerNodes);
		tmp_Y.set(bel.belStates[i]);
		tmp_Y.eltMpy(Ri_Y);
		bel.belStates[i-1].multiply(Mi_YY,tmp_Y);
	}
	for (int i = 0; i < NNODES; i++)
	{
		// compute the Mi matrix
		computeLogMi(fGen,model,X,i,seqLabel,Mi_YY,Ri_Y, true,
					 bUseStatePerNodes);

		if (i > 0)
		{
			tmp_Y.set(alpha_Y);
			Mi_YY.transpose();
			newAlpha_Y.multiply(Mi_YY,tmp_Y);
			newAlpha_Y.eltMpy(Ri_Y);
		}
		else
		{
			newAlpha_Y.set(Ri_Y);
		}

		if (i > 0)
		{
			tmp_Y.set(Ri_Y);
			tmp_Y.eltMpy(bel.belStates[i]);
			tmp_Y.transpose();
			bel.belEdges[i-1].multiply(alpha_Y,tmp_Y);
			Mi_YY.transpose();
			bel.belEdges[i-1].eltMpy(Mi_YY);
		}

		bel.belStates[i].eltMpy(newAlpha_Y);
		alpha_Y.set(newAlpha_Y);
	}
	double Zx = alpha_Y.sum();
	for (int i = 0; i < NNODES; i++)
	{
		bel.belStates[i].multiply(1.0/Zx);
	}
	for (int i = 0; i < NEDGES; i++)
	{
		bel.belEdges[i].multiply(1.0/Zx);
	}
	bel.partition = log(Zx);
}

void InferenceEngineFB::computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
										  DataSequence* X, Model* model,
										  int, int seqLabel,
										  bool bUseStatePerNodes)
{
	if(model->getAdjacencyMatType()!=CHAIN){
		throw HcrfBadModel("InferenceEngineFB need a model based on a Chain");
	}
	int NNODES=X->length();
	int NSTATES = model->getNumberOfStates();
	bel.belStates.resize(NNODES);

	for(int i=0;i<NNODES;i++)
	{
		bel.belStates[i].create(NSTATES);
	}
	//int NEDGES = (NNODES-1)*2; // LP: Should it really be *2 ?
	int NEDGES = NNODES-1;
	bel.belEdges.resize(NEDGES);
	for(int i=0;i<NEDGES;i++)
	{
		bel.belEdges[i].create(NSTATES,NSTATES, 0);
	}
	dMatrix Mi_YY (NSTATES,NSTATES);
	dVector Ri_Y (NSTATES);
	dVector alpha_Y(NSTATES);
	dVector newAlpha_Y(NSTATES);
	dVector tmp_Y(NSTATES);

	alpha_Y.set(0);
	// compute beta values in a backward scan.
	// also scale beta-values to 1 to avoid numerical problems.
	bel.belStates[NNODES-1].set(0);
	for (int i = NNODES-1; i > 0; i--)
	{
		// compute the Mi matrix
		computeLogMi(fGen, model, X, i, seqLabel, Mi_YY, Ri_Y, false,
					 bUseStatePerNodes);
		tmp_Y.set(bel.belStates[i]);
		tmp_Y.add(Ri_Y);
		LogMultiply(Mi_YY,tmp_Y,bel.belStates[i-1]);
	}

	// Compute Alpha values
	for (int i = 0; i < NNODES; i++) {
		// compute the Mi matrix
		computeLogMi(fGen,model, X, i, seqLabel, Mi_YY, Ri_Y,false,
					 bUseStatePerNodes);
		if (i > 0)
		{
			tmp_Y.set(alpha_Y);
			Mi_YY.transpose();
			LogMultiply(Mi_YY, tmp_Y, newAlpha_Y);
			newAlpha_Y.add(Ri_Y);
		}
		else
		{
			newAlpha_Y.set(Ri_Y);
		}
		if (i > 0)
		{
			tmp_Y.set(Ri_Y);
			tmp_Y.add(bel.belStates[i]);
			Mi_YY.transpose();
			bel.belEdges[i-1].set(Mi_YY);
			for(int yprev = 0; yprev < NSTATES; yprev++)
				for(int yp = 0; yp < NSTATES; yp++)
					bel.belEdges[i-1](yprev,yp) += tmp_Y[yp] + alpha_Y[yprev];
		}
	  
		bel.belStates[i].add(newAlpha_Y);
		alpha_Y.set(newAlpha_Y);
	}
	double lZx = alpha_Y.logSumExp();
	for (int i = 0; i < NNODES; i++)
	{
		bel.belStates[i].add(-lZx);
		bel.belStates[i].eltExp();
	}
	for (int i = 0; i < NEDGES; i++)
	{
		bel.belEdges[i].add(-lZx);
		bel.belEdges[i].eltExp();
	}
	bel.partition = lZx;
}







