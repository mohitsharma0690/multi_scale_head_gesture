//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Trainer Component
//
//	April 5, 2006

#ifndef __TOOLBOX_H
#define __TOOLBOX_H

//Standard Template Library includes
// ...

//hCRF Library includes

#include "RawFeatures.h"
#include "FeaturesOne.h"
#include "RawFeaturesSquare.h"
#include "WindowRawFeatures.h"
#ifndef _PUBLIC
#include "WindowRawFeatures2.h"
#include "WindowRawFeaturesRealtime.h"
#include "SharedFeatures.h"
#include "EdgeObservationFeatures.h"
#endif
#include "EdgeFeatures.h"
#include "LabelEdgeFeatures.h"
#include "dataset.h"
#include "model.h"
#include "optimizer.h"
#include "gradient.h"
#include "evaluator.h"
#include "inferenceengine.h"
#include "hcrfExcep.h"
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef int PORT_NUMBER;

enum {
   INIT_ZERO,
   INIT_CONSTANT,
   INIT_RANDOM,
   INIT_MEAN,
   INIT_RANDOM_MEAN_STDDEV,
   INIT_GAUSSIAN,
   INIT_RANDOM_GAUSSIAN,
   INIT_RANDOM_GAUSSIAN2,
   INIT_PREDEFINED,
   INIT_PERCEPTRON //
};

enum{
   OPTIMIZER_CG,
   OPTIMIZER_BFGS,
   OPTIMIZER_ASA,
   OPTIMIZER_OWLQN,
   OPTIMIZER_LBFGS,
   OPTIMIZER_HMMPERCEPTRON // Discriminative training for HMM
};

class Toolbox
{
  public:
   Toolbox();
   Toolbox(const Toolbox&);
   Toolbox& operator=(const Toolbox&){
       throw std::logic_error("Toolbox should not be copied");
   };
   virtual ~Toolbox();
   virtual void train(DataSet& X, bool bInitWeights = true);
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL) = 0;
   virtual void validate(DataSet& dataTrain, DataSet& dataValidate,
                         double& optimalRegularisation,
                         char* filenameStats = NULL);   
   
   virtual void load(char* filenameModel, char* filenameFeatures);
   virtual void save(char* filenameModel, char* filenameFeatures);
   virtual double computeError(DataSet& X);

   double getRegularizationL1();
   double getRegularizationL2();
   void setRegularizationL1(double regFactorL1,
                            eFeatureTypes typeFeature = allTypes);
   void setRegularizationL2(double regFactorL2,
                            eFeatureTypes typeFeature = allTypes);

   int getMaxNbIteration();
   int getWeightInitType();
   void setMaxNbIteration(int maxit);
   void setWeightInitType(int initType);

   void setRandomSeed(long seed);
   long getRandomSeed();

   void setInitWeights(const dVector& w);
   dVector& getInitWeights();

   void setWeights(const dVector& w);

   int getDebugLevel();
   void setDebugLevel(int newDebugLevel);

   featureVector* getAllFeatures(DataSet &X);
   Model* getModel();
   FeatureGenerator* getFeatureGenerator();
   Optimizer* getOptimizer();

   void setRangeWeights(double minRange, double maxRange);
   void setMinRangeWeights(double minRange);
   void setMaxRangeWeights(double maxRange);
   double getMinRangeWeights();
   double getMaxRangeWeights();

   void initWeights(DataSet &X);

   // To be able to set the number of thread using the toolbox
   // (usefull from python). To change the default scheduling, one
   // must use the environment variable OMP_SCHEDULE
   void set_num_threads(int);

  protected:
   virtual void init(int opt, int windowSize);
   virtual void initModel(DataSet &X) = 0;
   virtual void initWeightsRandom();
   virtual void initWeightsFromMean(DataSet &X);
   virtual void initWeightsRandomFromMeanAndStd(DataSet &X);
   virtual void initWeightsGaussian(DataSet &X);

   virtual void initWeightsConstant(double value);  

   virtual void initWeightsRandomGaussian();
   virtual void initWeightsRandomGaussian2();

   virtual void initWeightsPerceptron(DataSet& X);
   virtual void calculateGlobalMean(DataSet &X, dVector&mean);
   virtual void calculateGlobalMeanAndStd(DataSet &X, dVector& mean,
                                          dVector& stdDev);

   int weightInitType;
   dVector initW;
   double minRangeWeights;
   double maxRangeWeights;
   Optimizer* pOptimizer;
   Gradient* pGradient;
   Evaluator* pEvaluator;
   Model* pModel;
   InferenceEngine* pInferenceEngine;
   FeatureGenerator* pFeatureGenerator;
   long seed;
};

class ToolboxCRF: public Toolbox
{
  public:
   ToolboxCRF();
   ToolboxCRF(int opt, int windowSize = 0);
   virtual ~ToolboxCRF();
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);

  protected:
   virtual void initModel(DataSet &X);
   virtual void init(int opt, int windowSize);
};

class ToolboxCRFRealtime: public ToolboxCRF
{	
  public:
	ToolboxCRFRealtime(int opt, int windowSize, int RawFeatureId = WINDOW_RAW_FEATURE_ID2);
	virtual ~ToolboxCRFRealtime(); // need to release all the resource allocated in openPort()	
	
	virtual void openPort(PORT_NUMBER portNumber, int bufferLength); // If port is already opened, then it's content will be cleared
	virtual void closePort(PORT_NUMBER portNumber);
	virtual int insertOneFrame(PORT_NUMBER portNumber, const dVector* const features, dVector* prob); // 0 means no value is returned.
protected:
	virtual void init(int opt, int windowSize, int RawFeatureId);
  private:
	std::map<PORT_NUMBER, DataSequenceRealtime*> portNumberMap;
	int windowSize;
};


class ToolboxHCRF: public Toolbox
{
  public:
   ToolboxHCRF();
   ToolboxHCRF(int nbHiddenStates, int opt, int windowSize = 0);
   virtual ~ToolboxHCRF();
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);

  protected:
   virtual void initModel(DataSet &X);
   virtual void init(int nbHiddenStates, int opt, int windowSize);
  //private: {-KGB}
   int numberOfHiddenStates;
};

class ToolboxLDCRF: public Toolbox
{
  public:
   ToolboxLDCRF();
   ToolboxLDCRF(int nbHiddenStatesPerLabel, int opt, int windowSize = 0);
   virtual ~ToolboxLDCRF();
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);

  protected:
   virtual void init(int nbHiddenStatesPerLabel, int opt, int windowSize = 0);
   virtual void initModel(DataSet &X);
   int numberOfHiddenStatesPerLabel;
};

class ToolboxGHCRF: public ToolboxHCRF
{
  public:
   ToolboxGHCRF();
   ToolboxGHCRF(int nbHiddenStates, int opt, int windowSize = 0);
   virtual ~ToolboxGHCRF();

  protected:
   virtual void init(int nbHiddenStates, int opt, int windowSize);
};

class ToolboxSharedLDCRF: public ToolboxLDCRF
{
  public:
   ToolboxSharedLDCRF();
   ToolboxSharedLDCRF(int nbHiddenStates, int opt, int windowSize = 0);
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);

  protected:
   int numberOfHiddenStates;
   virtual void init(int nbHiddenStates, int opt, int windowSize);
   virtual void initModel(DataSet &X);
};

class ToolboxHMMPerceptron: public Toolbox
{
  public:
   ToolboxHMMPerceptron();
   ToolboxHMMPerceptron(int opt, int windowSize = 0);
   virtual ~ToolboxHMMPerceptron();
   virtual void train(DataSet &X, bool bInitWeights);
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);   

  protected:
   virtual void initModel(DataSet &X);
   virtual void init(int opt, int windowSize);
   InferenceEnginePerceptron* pInferenceEnginePerceptron;
   GradientPerceptron* pGradientPerceptron;
};

class ToolboxLVPERCEPTRON: Toolbox
{
   public:
	ToolboxLVPERCEPTRON();
	ToolboxLVPERCEPTRON(int nbHiddenStatesPerLabel,int opt, int windowSize = 0);
	virtual ~ToolboxLVPERCEPTRON();
	virtual double test(DataSet& X, char* filenameOutput = NULL, 
						char* filenameStats = NULL);
  
   protected:	
	virtual void init(int nbHiddenStatesPerLabel, int opt, int windowSize = 0);
    virtual void initModel(DataSet &X);
    int numberOfHiddenStatesPerLabel;
};

#endif
