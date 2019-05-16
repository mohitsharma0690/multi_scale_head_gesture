//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Gradient
// Component
//
//	January 30, 2006

#ifndef GRADIENT_H
#define GRADIENT_H

//Standard Template Library includes
#include <vector>

//hCRF Library includes
#include "dataset.h"
#include "model.h"
#include "inferenceengine.h"
#include "featuregenerator.h"
#include "evaluator.h"

#ifdef _OPENMP
#include <omp.h>
#endif

class Gradient {
  public:
    Gradient(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    Gradient(const Gradient&);
    Gradient& operator=(const Gradient&);
    virtual double computeGradient(dVector& vecGradrient, Model* m, 
                                 DataSequence* X) = 0;
    virtual double computeGradient(dVector& vecGradrient, Model* m, 
                                 DataSet* X);
    virtual ~Gradient();
	virtual void setMaxNumberThreads(int maxThreads);

  protected:
    InferenceEngine* pInfEngine;
    FeatureGenerator* pFeatureGen;
	featureVector *vecFeaturesMP;
	dVector *localGrads;
	int nbThreadsMP;
};

class GradientCRF : public Gradient 
{
  public:
    GradientCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradrient, Model* m, DataSequence* X);
    using Gradient::computeGradient;
};

class GradientHCRF : public Gradient 
{
  public:
    GradientHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradrient, Model* m, DataSequence* X);
    using Gradient::computeGradient;
};

class GradientLDCRF : public Gradient 
{
  public:
    GradientLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradrient, Model* m, DataSequence* X);
    using Gradient::computeGradient;
};

class GradientSharedLDCRF : public Gradient
{
  public:
  GradientSharedLDCRF(InferenceEngine* infEngine, 
                        FeatureGenerator* featureGen):
    Gradient(infEngine, featureGen){};
    double computeGradient(dVector& vecGradrient, Model* m, DataSequence* X);
    using Gradient::computeGradient;
};


class GradientDD : public Gradient 
{
  public:
    GradientDD(InferenceEngine* infEngine, FeatureGenerator* featureGen, 
               dVector* pMu = NULL);
    double computeGradient(dVector& vecGradrient, Model* m, DataSequence* X);
    double computeGradient(dVector& vecGradrient, Model* m, DataSet *X);
    
  private:
    dVector mu;
};

class GradientFD : public Gradient
{
  public:
    GradientFD(InferenceEngine* infEngine, FeatureGenerator* featureGen, 
               Evaluator* evaluator);
    GradientFD(const GradientFD&);
    GradientFD& operator=(const GradientFD&);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
    // We do the numerical deirvative directly on the sum (we do not
    // use the Gradient::computeGradient function)
    double computeGradient(dVector& vecGradient, Model* m, DataSet* X);
  private:
    Evaluator* pEvaluator;
};

class GradientPerceptron {
  public:
    GradientPerceptron(InferenceEnginePerceptron* infEngine, FeatureGenerator* featureGen);
    GradientPerceptron(const GradientPerceptron&);
    GradientPerceptron& operator=(const GradientPerceptron&);
    virtual double computeGradient(dVector& vecGradrient, Model* m, 
                                 DataSequence* X) = 0;
    virtual double computeGradient(dVector& vecGradrient, Model* m, 
                                 DataSet* X);
    virtual ~GradientPerceptron(){};

  protected:
    InferenceEnginePerceptron* pInfEngine;
    FeatureGenerator* pFeatureGen;
};

class GradientHMMPerceptron : public GradientPerceptron 
{
  public:
    GradientHMMPerceptron(InferenceEnginePerceptron* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradrient, Model* m, DataSequence* X);
    using GradientPerceptron::computeGradient;
};


#endif
