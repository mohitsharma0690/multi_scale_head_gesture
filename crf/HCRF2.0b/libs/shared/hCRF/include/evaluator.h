//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Evaluator
// Component
//
//	January 30, 2006

#ifndef EVALUATOR_H
#define EVALUATOR_H

//Standard Template Library includes
#include <list>

//hCRF Library includes
#include "featuregenerator.h"
#include "inferenceengine.h"
#include "dataset.h"
#include "model.h"
#include "hcrfExcep.h"

#ifdef _OPENMP
#include <omp.h>
#endif

class Evaluator 
{
  public:
    Evaluator();
    Evaluator(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    virtual ~Evaluator();
    Evaluator(const Evaluator& other);
    Evaluator& operator=(const Evaluator& other);
    void init(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    virtual double computeError(DataSequence* X, Model* m) = 0;
    virtual double computeError(DataSet* X, Model* m);
    virtual void computeStateLabels(DataSequence* X, Model* m, 
                                    iVector* vecStateLabels, 
                                    dMatrix * probabilities = NULL);
    virtual int computeSequenceLabel(DataSequence* X, Model* m, 
                                     dMatrix * probabilities);
    
  protected:
    InferenceEngine* pInfEngine;
    FeatureGenerator* pFeatureGen;
    void computeLabels(Beliefs& bel, iVector* vecStateLabels,
                       dMatrix * probabilities = NULL);
    friend class OptimizerLBFGS;
};


class EvaluatorCRF:public Evaluator
{
  public:
    EvaluatorCRF();
    EvaluatorCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorCRF();
    double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }

};

class EvaluatorHCRF:public Evaluator
{
  public:
    EvaluatorHCRF();
    EvaluatorHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorHCRF();
    double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }

    int computeSequenceLabel(DataSequence* X, Model* m, 
                             dMatrix * probabilities);
};

class EvaluatorLDCRF:public Evaluator
{
  public:
    EvaluatorLDCRF();
    EvaluatorLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorLDCRF();
    virtual double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }
    void computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, 
                            dMatrix * probabilities = NULL);
};

class EvaluatorLVPERCEPTRON:public Evaluator
{
public:
	EvaluatorLVPERCEPTRON();
	EvaluatorLVPERCEPTRON(InferenceEngine* infEngine, FeatureGenerator* featureGen);
	~EvaluatorLVPERCEPTRON();

	double computeError(DataSequence* X, Model* m);
	double computeError(DataSet* X, Model* m);

	void computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, dMatrix * probabilities = NULL);

};


class EvaluatorSharedLDCRF:public EvaluatorLDCRF
{
  public:
	EvaluatorSharedLDCRF(InferenceEngine* infEngine, 
                       FeatureGenerator* featureGen) 
      : EvaluatorLDCRF(infEngine, featureGen) {};   
    void computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, 
                            dMatrix * probabilities = NULL);
};


class EvaluatorDD: public Evaluator
{
public:
	EvaluatorDD(InferenceEngine* infEngine, FeatureGenerator* featureGen, 
              dVector* mu=0);
	~EvaluatorDD();
	double computeError(DataSet* X, Model* m);
	double computeError(DataSequence* X, Model* m);

private:
	dVector mu;
};

#endif
