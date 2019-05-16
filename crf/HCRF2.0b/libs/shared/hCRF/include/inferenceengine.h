//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Inference Engine
//
//	January 30, 2006

#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

//Standard Template Library includes
#include <vector>
#include <list>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>



//hCRF Library includes
#include "hcrfExcep.h"

#include "featuregenerator.h"
#include "matrix.h"


//#define INF_VALUE -DBL_MAX
#define INF_VALUE 1e100


#if defined(__VISUALC__)||defined(__BORLAND__)
    #define wxFinite(n) _finite(n)
#elseif defined(__GNUC__)
    #define wxFinite(n) finite(n)
#else
    #define wxFinite(n) ((n) == (n))
#endif


struct Beliefs {
public:
    std::vector<dVector> belStates;
    std::vector<dMatrix> belEdges;
    double partition;
    Beliefs()
    :belStates(), belEdges(), partition(0.0) {};
    
};

class InferenceEngine
{
  public:
    //Constructor/Destructor
    InferenceEngine();
    virtual ~InferenceEngine();

    virtual void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
                               DataSequence* X, Model* m,
                               int bComputePartition,int seqLabel=-1,
                               bool bUseStatePerNodes = false)=0;
    virtual double computePartition(FeatureGenerator* fGen,DataSequence* X,
                                    Model* m,int seqLabel=-1,
                                    bool bUseStatePerNodes = false) = 0;
	virtual int forwardBeliefLog(FeatureGenerator* fGen, Model* model,
								DataSequenceRealtime* dataSequence, dVector* prob) = 0;
	virtual void setMaxNumberThreads(int maxThreads);
  protected:
    // Private function that are used as utility function for several
    // beliefs propagations algorithms.
    int CountEdges(const uMatrix& AdjacencyMatrix, int nbNodes) const;

    void computeLogMi(FeatureGenerator* fGen, Model* model, DataSequence* X,
                      int i, int seqLabel, dMatrix& Mi_YY, dVector& Ri_Y,
                      bool takeExp, bool bUseStatePerNodes) ;
    void LogMultiply(dMatrix& Potentials,dVector& Beli, dVector& LogAB);
	featureVector *vecFeaturesMP;
	int nbThreadsMP;

};


class InferenceEngineBP: public InferenceEngine {
  public:
    //Constructor/Destructor
    InferenceEngineBP();
    InferenceEngineBP(const InferenceEngineBP&);
    InferenceEngineBP& operator=(const InferenceEngineBP&);
    ~InferenceEngineBP();

    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen, DataSequence* X,
                       Model* crf, int bComputePartition, int seqLabel=-1,
                       bool bUseStatePerNodes = false);
    double computePartition(FeatureGenerator* fGen,DataSequence* X, Model* crf,
                            int seqLabel=-1, bool bUseStatePerNodes = false);
	int forwardBeliefLog(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob); 
  private:
    void MakeLocalEvidence(dMatrix& LogLocalEvidence, FeatureGenerator* fGen, 
                           DataSequence* X, Model* crf, int seqLabel);
    void MakeEdgeEvidence(std::vector<dMatrix>& LogEdgeEvidence, 
                          FeatureGenerator* fGen,DataSequence* X, Model* crf, 
                          int seqLabel);
    int GetEdgeMatrix(int nodeI, int nodeJ, dMatrix& EvidenceEdgeIJ, 
                      std::vector<dMatrix>& EdgePotentials);
    void TreeInfer(Beliefs& bel, FeatureGenerator* fGen, DataSequence* X,
                   iVector &PostOrder, iVector &PreOrder, Model* crf, 
                   int seqLabel, bool bUseStatePerNodes);
    int computeEdgeBel(Beliefs& bel, std::vector<dMatrix> Messages, 
                       std::vector<dMatrix>& LogEdgeEvidence);
    double computeZ(int i, int j, dVector& lastm, dMatrix& LogLocalEvidence);

    int children(int NodeNumber, iVector& Child, uMatrix& Ad);
    int parents(int NodeNumber,iVector& Parent,uMatrix& Ad);
    int neighboor(int NodeNumber,iVector& Neighboors,uMatrix& Ad);
    int dfs(iVector& Pred, iVector& Post);
    void Dfs_Visit(int u,uMatrix& Ad);
    int findNbrs(int nodeI, iVector& NBRS);
    void BuildEdgeMatrix2();

    void GetRow (dMatrix& M, int j, dVector& rowfVectorj);
    void PrintfVector(dVector& v1);
    double MaxValue(dVector& v1);
    void Transpose(dMatrix& M, dMatrix& MTranspose);
    void LogNormalise(dVector& OldV, dVector& NewV);
    void LogNormalise2(dMatrix& OldM, dMatrix& NewM);
    void repMat1(dVector& beli,int nr, dMatrix& Result);
    void repMat2(dVector& beli,int nr, dMatrix& Result);

    // Global Variables ( for the inference helper function that gets postorder
    int white_global, gray_global, black_global;
    int sizepre_global, sizepost_global;
    int time_stamp_global, cycle_global;
    iVector color_global, d_global, f_global;
    iVector pre_global, post_global, pred_global;
    int NNODES;
    int NEDGES;
    int NFEATURES;
    uMatrix AdjacencyMatrix;
    dVector *theta;
    iMatrix EMatrix;
    uMatrix Tree;

    int NSTATES;
};



class InferenceEngineDC:public InferenceEngine
{
/**
This class is used for beliefs propagation on a tree composed of chain
of hidden states where each is connected to labels:

y1   y2   y3   y4   y5   y6
|    |    |    |    |    |
h1---h2---h3---h4---h5---h6
|    |    |    |    |    |
X    X    X    X    X    X

X represents the observations. 
**/   
  public:
    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X,
                        Model* crf, int bComputePartition,int seqLabel=-1,
                        bool bUseStatePerNodes = false);
    double computePartition(FeatureGenerator* fGen, DataSequence* X,
                            Model* crf, int seqLabel=-1,
                            bool bUseStatePerNodes = false);
	int forwardBeliefLog(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob); 
  protected:
    void computeObsMsg(FeatureGenerator* fGen, Model* model, DataSequence* X,
                       int i, dMatrix& Mi_HY, dVector& Ri_Y, dVector& Pi_Y,
                       bool takeExp, bool bUseStatePerNodes);
    void computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
                           DataSequence* X, Model* model,
                           int bComputePartition,int seqLabel,
                           bool bUseStatePerNodes);
    void computeBeliefsLinear(Beliefs& bel, FeatureGenerator* fGen,
						  DataSequence* X, Model* model,
						  int bComputePartition,
						  int seqLabel, bool bUseStatePerNodes);
};

class InferenceEngineFB:public InferenceEngine
{
public:
    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X,
                        Model* crf, int bComputePartition,int seqLabel=-1,
                        bool bUseStatePerNodes = false);
    double computePartition(FeatureGenerator* fGen, DataSequence* X,
                            Model* crf, int seqLabel=-1,
                            bool bUseStatePerNodes = false);	
	int forwardBeliefLog(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob); 
private:
    void computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
                           DataSequence* X, Model* model,
                           int bComputePartition,int seqLabel,
                           bool bUseStatePerNodes);
    void computeBeliefsLinear(Beliefs& bel, FeatureGenerator* fGen,
                           DataSequence* X, Model* model,
                           int bComputePartition,int seqLabel,
                           bool bUseStatePerNodes);
};

class InferenceEngineDummy:public InferenceEngine
{
  public:
    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X, 
                       Model* crf, int bComputePartition,int seqLabel=-1, 
                       bool bUseStatePerNodes = false);
    double computePartition(FeatureGenerator* fGen, DataSequence* X, 
                            Model* crf,int seqLabel = -1, 
                            bool bUseStatePerNodes = false);
};

class InferenceEngineBrute:public InferenceEngine
{
  public:  
    InferenceEngineBrute();
    ~InferenceEngineBrute();

    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen, DataSequence* X, 
                       Model* crf, int bComputePartition, int seqLabel=-1, 
                       bool bUseStatePerNodes = false);
    double computePartition(FeatureGenerator* fGen, DataSequence* X, 
                            Model* crf,int seqLabel=-1, 
                            bool bUseStatePerNodes = false);
	int forwardBeliefLog(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob);  
  private:
    int computeMaskedBeliefs(Beliefs& bel, FeatureGenerator* fGen,
                             DataSequence* X, Model* m, int bComputePartition,
                             int seqLabel=-1);
    double computeMaskedPartition(FeatureGenerator* fGen, DataSequence* X, 
                                  Model* m,int seqLabel=-1);
};

class InferenceEnginePerceptron// Similar to InferenceEngineFB, InferenceEnginePerceptron may be used for both CRF and LDCRF 
{
  public:    
    InferenceEnginePerceptron();
    ~InferenceEnginePerceptron();
   
	void computeViterbiPath(iVector& viterbiPath, FeatureGenerator* fGen,
                       DataSequence* X, Model* m,
                       int seqLabel = -1, bool bUseStatePerNodes = false);
	private:    
    void computeMi(FeatureGenerator* fGen, Model* model, DataSequence* X,
                      int index, int seqLabel, dMatrix& Mi_YY, dVector& Ri_Y,
                      bool bUseStatePerNodes);
	void ViterbiForwardMax(dMatrix& Mi_YY, dVector& Ri_Y, dVector& alpha_Y, 
		std::vector<iVector>& viterbiBacktrace,int index);	
};

#endif //INFERENCEENGINE_H
