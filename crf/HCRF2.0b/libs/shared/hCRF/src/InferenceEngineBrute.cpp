#include "inferenceengine.h"
#include <iostream>

using namespace std;

InferenceEngineBrute::InferenceEngineBrute()
{

}

InferenceEngineBrute::~InferenceEngineBrute()
{

}

int InferenceEngineBrute::forwardBeliefLog(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob) 
{
	return 0;
}
void InferenceEngineBrute::computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
										  DataSequence* X, Model* crf, 
										  int bComputePartition, int seqLabel, 
										  bool bUseStatePerNodes)
{
	if(bUseStatePerNodes){
		computeMaskedBeliefs(bel,fGen,X,crf,bComputePartition,seqLabel);
		return;
	}
	// Backup the real state labels
	if(X->getStateLabels() == NULL)
		X->setStateLabels(new iVector(X->length()));
	iVector oldY;
	oldY.set(*X->getStateLabels());

	bel.partition = exp(computePartition(fGen,X, crf,seqLabel));

	int NNODES=X->length();
	int NSTATES = crf->getNumberOfStates();
	X->getStateLabels()->set(0);
	uMatrix AdjacencyMatrix;
	crf->getAdjacencyMatrix(AdjacencyMatrix, X);
	int NEDGES=CountEdges(AdjacencyMatrix, NNODES);
	bel.belStates.resize(NNODES);
	for(int i=0;i<NNODES;i++)
	{
		bel.belStates[i].create(NSTATES);
		bel.belStates[i].set(0);
	}
	bel.belEdges.resize(NEDGES*2);
	for(int i=0;i<NEDGES*2;i++)
	{
		bel.belEdges[i].create(NSTATES,NSTATES);
		bel.belEdges[i].set(0);
	}

	int nbY = (int)pow((float)NSTATES,NNODES-1);

	//Compute beliefs for nodes
	for(int i = 0; i < NNODES; i++)
	{
		for(int s = 0; s < NSTATES; s++)
		{
			X->getStateLabels()->set(0);
			(*X->getStateLabels())[i] = s;
			bel.belStates[i][s] = 0.0;
			for ( int j = 0 ; j < nbY; j++)
			{
				//Compute exp(PHI)
				bel.belStates[i][s] += exp(fGen->evaluateLabels(X,crf,seqLabel));

				//Increment state labels
				int k = 0;
				if(i == 0)
					k++;

				(*X->getStateLabels())[k] = (*X->getStateLabels())[k]+1;
				while((*X->getStateLabels())[k] >= NSTATES && k < NNODES)
				{
					(*X->getStateLabels())[k] = 0;
					k++;
					if(k == i)
						k++;
					if(k < NNODES)
						(*X->getStateLabels())[k] = (*X->getStateLabels())[k] + 1;
				}
			}
			bel.belStates[i][s] /= bel.partition;
		}
	}


	int nbYedges = (int)pow((float)NSTATES,NNODES-2);

	//Compute beliefs for edges
	int edgeindex = 0;
	for(int i = 0; i < NNODES; i++)
	{
		for(int j = 0; j < NNODES; j++)
		{
			if( i == j)
				continue;
			if(AdjacencyMatrix(i,j) == 1)
			{
				for(int si = 0; si < NSTATES; si++)
				{
					for(int sj = 0; sj < NSTATES; sj++)
					{
						X->getStateLabels()->set(0);
						(*X->getStateLabels())[i] = si;
						(*X->getStateLabels())[j] = sj;
						bel.belEdges[edgeindex](si,sj) = 0.0;
						for ( int z = 0 ; z < nbYedges; z++)
						{
							//Compute exp(PHI)
							bel.belEdges[edgeindex](si,sj) += exp(fGen->evaluateLabels(X,crf,seqLabel));

							//Increment state labels
							int k = 0;
							if(i == 0 || j==0)
							{
								k++;
								if(i == 1 || j==1)
									k++;
							}

							if(k < NNODES)
							{
								(*X->getStateLabels())[k] = (*X->getStateLabels())[k]+1;
								while((*X->getStateLabels())[k] >= NSTATES && k < NNODES)
								{
									(*X->getStateLabels())[k] = 0;
									k++;
									if(k == i || k == j)
										k++;
									if(k < NNODES)
										(*X->getStateLabels())[k] = (*X->getStateLabels())[k] + 1;
								}
							}
						}
						bel.belEdges[edgeindex](si,sj) /= bel.partition;
					}
				}
				edgeindex++;
			}
		}
	}

	// Restore the real state labels
	X->getStateLabels()->set(oldY);
}


int InferenceEngineBrute::computeMaskedBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X, Model* crf, int, int seqLabel)
/* We always get the partition, for free, so we dont use the parameter bComputePartition */
{
	// Backup the real state labels
	if(X->getStateLabels() == NULL)
		X->setStateLabels(new iVector(X->length()));
	iVector oldY;
	oldY.set(*X->getStateLabels());

	bel.partition = exp(computeMaskedPartition(fGen,X, crf,seqLabel));

	int NNODES=X->length();
	int NSTATES = crf->getNumberOfStates();
	X->getStateLabels()->set(0);
	iVector firstY(NNODES);
	uMatrix AdjacencyMatrix;
	crf->getAdjacencyMatrix(AdjacencyMatrix, X);
	iMatrix* pStatesPerNodes = crf->getStateMatrix(X);
	int NEDGES=CountEdges(AdjacencyMatrix, NNODES);
	bel.belStates.resize(NNODES);
	for(int i=0;i<NNODES;i++)
	{
		bel.belStates[i].create(NSTATES);
		bel.belStates[i].set(0);
	}
	bel.belEdges.resize(NEDGES*2);
	for(int i=0;i<NEDGES*2;i++)
	{
		bel.belEdges[i].create(NSTATES,NSTATES);
		bel.belEdges[i].set(0);
	}

	// Initialise the first Y
	firstY.set(0);
	for (int kk = 0; kk < NNODES; kk++)
	{
		for (int ss = 0; ss < NSTATES; ss++)
		{
			if(pStatesPerNodes->getValue(ss,kk) == 1)
			{
				firstY[kk] = ss;
				break;
			}
		}
	}


	//Compute beliefs for nodes
	for(int i = 0; i < NNODES; i++)
	{
		for(int s = 0; s < NSTATES; s++)
		{
			if(pStatesPerNodes->getValue(s,i) == 1)
			{
				X->getStateLabels()->set(firstY);
				(*X->getStateLabels())[i] = s;
				bel.belStates[i][s] = 0.0;
				// compute number of masked tate labels Y
				int nbY = 1;
				for (int n = 0; n < NNODES; n++)
				{
					if(n != i)
						nbY *= pStatesPerNodes->colSum(n);
				}

				for ( int j = 0 ; j < nbY; j++)
				{
					//Compute exp(PHI)
					bel.belStates[i][s] += exp(fGen->evaluateLabels(X,crf,seqLabel));

					//Increment state labels
					int k = 0;
					if(i == 0)
						k++;

					(*X->getStateLabels())[k] = (*X->getStateLabels())[k]+1;
					while((*X->getStateLabels())[k] < NSTATES && pStatesPerNodes->getValue((*X->getStateLabels())[k],k) == 0)
						(*X->getStateLabels())[k]++;
					while((*X->getStateLabels())[k] >= NSTATES && k < NNODES)
					{
						(*X->getStateLabels())[k] = 0;
						while((*X->getStateLabels())[k] < NSTATES && pStatesPerNodes->getValue((*X->getStateLabels())[k],k) == 0)
							(*X->getStateLabels())[k]++;
						k++;
						if(k == i)
							k++;
						if(k < NNODES)
						{
							(*X->getStateLabels())[k] = (*X->getStateLabels())[k] + 1;
							while((*X->getStateLabels())[k] < NSTATES && pStatesPerNodes->getValue((*X->getStateLabels())[k],k) == 0)
								(*X->getStateLabels())[k]++;
						}
					}
				}
				bel.belStates[i][s] /= bel.partition;
			}
		}
	}


	//Compute beliefs for edges
	int edgeindex = 0;
	for(int i = 0; i < NNODES; i++)
	{
		for(int j = 0; j < NNODES; j++)
		{
			if( i == j){
				continue;
			}
			if(AdjacencyMatrix(i,j) == 1)
			{
				for(int si = 0; si < NSTATES; si++)
				{
					for(int sj = 0; sj < NSTATES; sj++)
					{
						if(pStatesPerNodes->getValue(si,i) == 1 && pStatesPerNodes->getValue(sj,j) == 1)
						{
							X->getStateLabels()->set(firstY);
							(*X->getStateLabels())[i] = si;
							(*X->getStateLabels())[j] = sj;
							bel.belEdges[edgeindex](si,sj) = 0.0;
							// compute number of masked tate labels Y
							int nbYedges = 1;
							for (int n = 0; n < NNODES; n++)
							{
								if(n != i && n != j)
									nbYedges *= pStatesPerNodes->colSum(n);
							}
							for ( int z = 0 ; z < nbYedges; z++)
							{
								//Compute exp(PHI)
								bel.belEdges[edgeindex](si,sj) += exp(fGen->evaluateLabels(X,crf,seqLabel));

								//Increment state labels
								int k = 0;
								if(i == 0 || j==0)
								{
									k++;
									if(i == 1 || j==1)
										k++;
								}

								if(k < NNODES)
								{
									(*X->getStateLabels())[k] = (*X->getStateLabels())[k]+1;
									while((*X->getStateLabels())[k] < NSTATES  && pStatesPerNodes->getValue((*X->getStateLabels())[k],k) == 0)
										(*X->getStateLabels())[k]++;
									while((*X->getStateLabels())[k] >= NSTATES && k < NNODES)
									{
										(*X->getStateLabels())[k] = 0;
										k++;
										if(k == i || k == j)
											k++;
										if(k < NNODES)
										{
											(*X->getStateLabels())[k] = (*X->getStateLabels())[k] + 1;
											while((*X->getStateLabels())[k] < NSTATES && pStatesPerNodes->getValue((*X->getStateLabels())[k],k) == 0)
												(*X->getStateLabels())[k]++;
										}
									}
								}
							}
							bel.belEdges[edgeindex](si,sj) /= bel.partition;
						}
					}
				}
				edgeindex++;
			}
		}
	}

	// Restore the real state labels
	X->getStateLabels()->set(oldY);
	return 1;
}

double InferenceEngineBrute::computePartition(FeatureGenerator* fGen,DataSequence* X, Model* crf, int seqLabel, bool bUseStatePerNodes)
{
	if(bUseStatePerNodes)
		return computeMaskedPartition(fGen,X,crf,seqLabel);

	// Backup the real state labels
	if(X->getStateLabels() == NULL)
		X->setStateLabels(new iVector(X->length()));
	iVector oldY;
	oldY.set(*X->getStateLabels());

	int NNODES=X->length();
	int NSTATES = crf->getNumberOfStates();
	X->getStateLabels()->set(0);
	double partition = 0.0;

	int nbY = (int)pow((float)NSTATES,NNODES);
	for ( int j = 0 ; j < nbY; j++)
	{
		//Compute exp(PHI)
		partition += exp(fGen->evaluateLabels(X,crf,seqLabel));

		//Increment state labels
		X->getStateLabels()->setValue(0, X->getStateLabels()->getValue(0)+1);
		int i = 0;
		while(X->getStateLabels()->getValue(i) >= NSTATES && i < NNODES)
		{
			X->getStateLabels()->setValue(i,0);
			i++;
			if(i < NNODES)
				X->getStateLabels()->setValue(i, X->getStateLabels()->getValue(i)+1);
		}
	}

	// Restore the real state labels
	X->getStateLabels()->set(oldY);
	return log(partition);
}

double InferenceEngineBrute::computeMaskedPartition(FeatureGenerator* fGen,DataSequence* X, Model* m,int seqLabel)
{
	// Backup the real state labels
	if(X->getStateLabels() == NULL)
		X->setStateLabels(new iVector(X->length()));
	iVector oldY;
	oldY.set(*X->getStateLabels());

	int NNODES = X->length();
	int NSTATES = m->getNumberOfStates();
	iMatrix* pStatesPerNodes = m->getStateMatrix(X);
	X->getStateLabels()->set(0);
	double partition = 0.0;

	// Initialise the first Y
	for (int k = 0; k < NNODES; k++)
	{
		for (int s = 0; s < NSTATES; s++)
		{
			if(pStatesPerNodes->getValue(s,k) == 1)
			{
				(*X->getStateLabels())[k] = s;
				break;
			}
		}
	}


	// compute number of masked tate labels Y
	int nbY = pStatesPerNodes->colSum(0);
	for (int k = 1; k < NNODES; k++)
		nbY *= pStatesPerNodes->colSum(k);

	for ( int j = 0 ; j < nbY; j++)
	{
		//Compute exp(PHI)
		partition += exp(fGen->evaluateLabels(X,m,seqLabel));

		//Increment state labels
		(*X->getStateLabels())[0] = (*X->getStateLabels())[0]+1;
		while((*X->getStateLabels())[0] < NSTATES && pStatesPerNodes->getValue((*X->getStateLabels())[0],0) == 0)
			(*X->getStateLabels())[0]++;
		int i = 0;
		while((*X->getStateLabels())[i] >= NSTATES && i < NNODES)
		{
			(*X->getStateLabels())[i] = 0;
			while((*X->getStateLabels())[i] < NSTATES && pStatesPerNodes->getValue((*X->getStateLabels())[i],i) == 0)
				(*X->getStateLabels())[i]++;
			i++;
			if(i < NNODES)
			{
				(*X->getStateLabels())[i] = (*X->getStateLabels())[i] + 1;
				while((*X->getStateLabels())[i] < NSTATES && pStatesPerNodes->getValue((*X->getStateLabels())[i],i) == 0)
					(*X->getStateLabels())[i]++;
			}
		}
	}

	// Restore the real state labels
	X->getStateLabels()->set(oldY);
	return log(partition);
}

