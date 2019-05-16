#include "toolbox.h"
#include <time.h>
#include "uncoptim.h"
#include "optimizer.h"


Toolbox::Toolbox()
	: weightInitType(INIT_RANDOM), minRangeWeights(-1.0), maxRangeWeights(1.0),
	  pOptimizer(NULL), pGradient(NULL), pEvaluator(NULL), pModel(NULL),  
	  pInferenceEngine(NULL), pFeatureGenerator(NULL), seed(0)
{}

void Toolbox::init(int opt, int windowSize)
{
	pModel = new Model();
	pFeatureGenerator = new FeatureGenerator;
	pFeatureGenerator->addFeature(new WindowRawFeatures(windowSize));
	pFeatureGenerator->addFeature(new EdgeFeatures());
	pInferenceEngine = new InferenceEngineFB();
#ifndef _PUBLIC
	if( opt == OPTIMIZER_HMMPERCEPTRON)
		pOptimizer = new OptimizerPerceptron();
	else if( opt == OPTIMIZER_BFGS)
#else
	if( opt == OPTIMIZER_BFGS)
#endif
		pOptimizer = new OptimizerUncOptim();
#ifndef _PUBLIC
	else if( opt == OPTIMIZER_ASA)
		pOptimizer = new OptimizerASA();
	// We want to be sure that if the library was not compiled with support for
	// the following optimizer, we get an error of the user try to use those
	// optimizer
#endif
#ifdef USEOWL
	else if(opt == OPTIMIZER_OWLQN){
	   pOptimizer = new OptimizerOWL();
	}
#else
	else if(opt == OPTIMIZER_OWLQN){
	   throw InvalidOptimizer("Not support for OWLQN compiled in the library");
	}
#endif

#ifdef USELBFGS
	else if(opt == OPTIMIZER_LBFGS){
		pOptimizer = new OptimizerLBFGS();
	}
#else
	else if(opt == OPTIMIZER_LBFGS){
	   throw InvalidOptimizer("Not support for LBFGS compiled in the library");
	}
#endif
	else if(opt == OPTIMIZER_CG) {
		pOptimizer = new OptimizerCG();
	}
	else {
		throw InvalidOptimizer("Invalid optimizer specified");
	}

}

Toolbox::~Toolbox()
{
	if(pOptimizer)
	{
		delete pOptimizer;
		pOptimizer = NULL;
	}
	if(pGradient)
	{
		delete pGradient;
		pGradient = NULL;
	}
	if(pEvaluator)
	{
		delete pEvaluator;
		pEvaluator = NULL;
	}
	if(pModel)
	{
		delete pModel;
		pModel = NULL;
	}
	if(pInferenceEngine)
	{
		delete pInferenceEngine;
		pInferenceEngine = NULL;
	}
	if(pFeatureGenerator)
	{
		delete pFeatureGenerator;
		pFeatureGenerator = NULL;
	}
}

void Toolbox::train(DataSet &X, bool bInitWeights)
{
	if(bInitWeights)
	{
		initModel(X);
		initWeights(X);
	}
	//Start training
	pOptimizer->optimize(pModel,&X, pEvaluator,pGradient);
}

double Toolbox::computeError(DataSet& X)
{

	return pEvaluator->computeError(&X, pModel);


	//UnconstrainedOptimizer* internalOptimizer = new UnconstrainedOptimizer;

	//internalOptimizer->currentModel = pModel;
	//internalOptimizer->currentDataset = &X;
	//internalOptimizer->currentEvaluator = pEvaluator;		
	//internalOptimizer->setDimension(pModel->getWeights()->getLength());	
	//// Set the initial weights
	//memcpy(internalOptimizer->x, pModel->getWeights()->get(), internalOptimizer->n*sizeof(double));
	//return internalOptimizer->F();
}


void Toolbox::setRangeWeights(double minRange, double maxRange)
{
	minRangeWeights = minRange;
	maxRangeWeights = maxRange;
}

double Toolbox::getMinRangeWeights()
{
	return minRangeWeights;
}

double Toolbox::getMaxRangeWeights()
{
	return maxRangeWeights;
}

void Toolbox::setMinRangeWeights(double minRange)
{
	minRangeWeights = minRange;
}

void Toolbox::setMaxRangeWeights(double maxRange)
{
	maxRangeWeights = maxRange;
}

int Toolbox::getWeightInitType()
{
	return weightInitType;
}

void Toolbox::setWeightInitType(int initType)
{
	weightInitType = initType;
}

void Toolbox::setRandomSeed(long in_seed)
{
   seed = in_seed;
}

long Toolbox::getRandomSeed()
{
   return seed;
}

void Toolbox::setInitWeights(const dVector& w)
{
	setWeightInitType(INIT_PREDEFINED);
	initW.set(w);
}

void Toolbox::setWeights(const dVector& w)
{
	pModel->setWeights(w);
	//setWeightInitType(INIT_PREDEFINED);
	//initW.set(w);
}

dVector& Toolbox::getInitWeights()
{
	return initW;
}


void Toolbox::initWeights(DataSet &X)
{
   switch(weightInitType)
   {
	  case INIT_ZERO:
		 initWeightsConstant(0.0);
		 break;
	  case INIT_CONSTANT:
		 initWeightsConstant(minRangeWeights);
		 break;
	  case INIT_MEAN:
		 initWeightsRandomFromMeanAndStd(X);
		 break;
	  case INIT_RANDOM_MEAN_STDDEV:
		 initWeightsRandomFromMeanAndStd(X);
		 break;
	  case INIT_GAUSSIAN:
		 initWeightsGaussian(X);
		 break;
	  case INIT_RANDOM_GAUSSIAN:
		 initWeightsRandomGaussian();
		 break;
	  case INIT_RANDOM_GAUSSIAN2:
		 initWeightsRandomGaussian2();
		 break;
	  case INIT_PREDEFINED:
		 pModel->setWeights(initW);
		 break;
	  case INIT_PERCEPTRON:
		 initWeightsPerceptron(X);
		 break;
	  case INIT_RANDOM:
		 initWeightsRandom();
		 break;
   }
}
void Toolbox::initWeightsConstant(double value)
{
	dVector w(pFeatureGenerator->getNumberOfFeatures());
	w.set(value);
	pModel->setWeights(w);
}

void Toolbox::initWeightsRandom()
{
   //Initialise weights. We use the seed (or clock) to initiliaze random number
   //generator
   if (seed==0){
	  srand( (unsigned)time( NULL ) );
   }
   else {
	  srand(seed);
   }
	  
	dVector w(pFeatureGenerator->getNumberOfFeatures());
	double widthRangeWeight = fabs(maxRangeWeights - minRangeWeights);
	for(int i = 0; i < w.getLength(); i++)
		w.setValue(i,(((double)rand())/(double)RAND_MAX)*widthRangeWeight+minRangeWeights);
	pModel->setWeights(w);
}


void Toolbox::initWeightsFromMean(DataSet &X)
{
	int nbRawFeatures = X.getNumberofRawFeatures();

	//Initialise weights
	dVector w(pFeatureGenerator->getNumberOfFeatures());
	double widthRangeWeight = fabs(maxRangeWeights - minRangeWeights);

	// mean and std_dev
	dVector mean(nbRawFeatures);
	calculateGlobalMean(X,mean);

	//Initialize weights with global mean and standard deviation
	// Only initialize the values especific to the HMM-like HCRF
	featureVector* vecFeatures = getAllFeatures(X);
	feature* pFeature = vecFeatures->getPtr();
	for(int j = 0; j < vecFeatures->size(); j++, pFeature++)
	{
		switch(pFeature->nodeIndex)
		{
		case SQUARE_RAW_FEATURE_ID:
			w.setValue(pFeature->globalId,mean[(int)pFeature->value] * mean[(int)pFeature->value]);
			break;
		case ONE_FEATURE_ID:
		case RAW_FEATURE_ID:
			w.setValue(pFeature->globalId,mean[(int)pFeature->value]);
			break;
		default:
			w.setValue(pFeature->globalId,(((double)rand())/(double)RAND_MAX)*widthRangeWeight+minRangeWeights);
		}
	}
	pModel->setWeights(w);
}

void Toolbox::initWeightsRandomFromMeanAndStd(DataSet &X)
{
   //Initialise weights. We use the seed (or clock) to initiliaze random number
   //generator
   if (seed==0){
	  srand( (unsigned)time( NULL ) );
   }
   else {
	  srand(seed);
   }
	int nbRawFeatures = X.getNumberofRawFeatures();

	//Initialise weights
	dVector w(pFeatureGenerator->getNumberOfFeatures());
	double widthRangeWeight = fabs(maxRangeWeights - minRangeWeights);
	double randValue;

	// mean and std_dev
	dVector mean(nbRawFeatures);
	dVector stdDev(nbRawFeatures);
	calculateGlobalMeanAndStd(X,mean,stdDev);

	//Initialize weights with global mean and standard deviation
	// Only initialize the values especific to the HMM-like HCRF
	featureVector* vecFeatures = getAllFeatures(X);
	feature* pFeature = vecFeatures->getPtr();
	for(int j = 0; j < vecFeatures->size(); j++, pFeature++)
	{
		switch(pFeature->nodeIndex)
		{
		case SQUARE_RAW_FEATURE_ID:
			randValue = (((double)rand())/(double)RAND_MAX)*2.0*stdDev[(int)pFeature->value]-stdDev[(int)pFeature->value];
			w.setValue(pFeature->globalId,mean[(int)pFeature->value] * mean[(int)pFeature->value]+randValue);
			break;
		case ONE_FEATURE_ID:
		case RAW_FEATURE_ID:
			randValue = (((double)rand())/(double)RAND_MAX)*2.0*stdDev[(int)pFeature->value]-stdDev[(int)pFeature->value];
			w.setValue(pFeature->globalId,mean[(int)pFeature->value]+randValue);
			break;
		default:
			w.setValue(pFeature->globalId,(((double)rand())/(double)RAND_MAX)*widthRangeWeight+minRangeWeights);
		}
	}
	pModel->setWeights(w);
}

void Toolbox::initWeightsGaussian(DataSet &X)
{
   //Initialise weights. We use the seed (or clock) to initiliaze random number
   //generator
   if (seed==0){
	  srand( (unsigned)time( NULL ) );
   }
   else {
	  srand(seed);
   }
	int nbRawFeatures = X.getNumberofRawFeatures();
	double specificWeight;

	//Initialise weights
	dVector w(pFeatureGenerator->getNumberOfFeatures());
	double widthRangeWeight = fabs(maxRangeWeights - minRangeWeights);

	// mean and std_dev
	dVector mean(nbRawFeatures);
	dVector stdDev(nbRawFeatures);
	calculateGlobalMeanAndStd(X,mean,stdDev);

	//Initialize weights with global mean and standard deviation
	// Only initialize the values especific to the HMM-like HCRF
	featureVector* vecFeatures = getAllFeatures(X);
	feature* pFeature = vecFeatures->getPtr();
	for(int j = 0; j < vecFeatures->size(); j++, pFeature++)
	{
		switch(pFeature->nodeIndex)
		{
		case SQUARE_RAW_FEATURE_ID:
			specificWeight = -1.0/(2.0*pow(stdDev[(int)pFeature->value],2));
			w.setValue(pFeature->globalId,specificWeight);
			break;
		case ONE_FEATURE_ID:
			specificWeight = (( (-pow( mean[(int)pFeature->value], 2))/
								(2.0*pow(mean[(int)pFeature->value],2)))
							  -(log(sqrt(3.141516*stdDev[(int)pFeature->value]))));
			w.setValue(pFeature->globalId,specificWeight);
			break;
		case RAW_FEATURE_ID:
			specificWeight = (mean[(int)pFeature->value])/(stdDev[(int)pFeature->value]);				
			w.setValue(pFeature->globalId,specificWeight);		
			break;
		default:
			w.setValue(pFeature->globalId,(((double)rand())/(double)RAND_MAX)*widthRangeWeight+minRangeWeights);
		}
	}
	pModel->setWeights(w);
}

void Toolbox::initWeightsRandomGaussian()
{
   //Initialise weights. We use the seed (or clock) to initiliaze random number
   //generator
   if (seed==0){
	  srand( (unsigned)time( NULL ) );
   }
   else {
	  srand(seed);
   }
/* This function initialize the weights with a gaussian distribution. The mean
 * is always zero, the std dev is stored in maxRangeWeights */
	unsigned int length = pFeatureGenerator->getNumberOfFeatures();
	dVector w(length);
	double x1, x2;
	for(unsigned int i =0; i<length; i+=2) 
	{
	   // We generate random normal number using Marsaglia Polar Method (2
	   // numbers are generated each iteration
	   double s;
	   do
	   {
		  x1 = ((double)rand())/(0.5 * (double)RAND_MAX) - 1.0;
		  x2 = ((double)rand())/(0.5 * (double)RAND_MAX) - 1.0;
		  s = x1*x1 + x2*x2;
	   } while (s>=1.0);
	   double coeff = sqrt( (-2.0 * log(s)) / s);
	   w[i] = x1*coeff*maxRangeWeights;
	   if (i+1 < length ) {
		  w[i+1] = x2*coeff*maxRangeWeights;
	   }
	}
	pModel->setWeights(w);
}

void Toolbox::initWeightsRandomGaussian2()
{
   //Initialise weights. We use the seed (or clock) to initiliaze random number
   //generator
   if (seed==0){
	  srand( (unsigned)time( NULL ) );
   }
   else {
	  srand(seed);
   }
/* This function initialize the weights with a gaussian distribution. The mean
 * is always zero, the std dev is stored in maxRangeWeights */
	unsigned int length = pFeatureGenerator->getNumberOfFeatures();
	dVector w(length);
	double x1, x2;
	for(unsigned int i =0; i<length; i+=2) 
	{
	   // We generate random normal number using Marsaglia Polar Method (2
	   // numbers are generated each iteration
	   double s;
	   do
	   {
		  x1 = ((double)rand())/(0.5 * (double)RAND_MAX) - 1.0;
		  x2 = ((double)rand())/(0.5 * (double)RAND_MAX) - 1.0;
		  s = x1*x1 + x2*x2;
	   } while (s>=1.0);
	   double coeff = sqrt( (-2.0 * log(s)) / s);
	   w[i] = x1*coeff*maxRangeWeights;
	   if (i+1 < length ) {
		  w[i+1] = x2*coeff*maxRangeWeights;
	   }
	}

	int nbLabels = pModel->getNumberOfStateLabels();
	int nbHiddenStates = pModel->getNumberOfStates();

	int nbHiddenStatesPerLabel = nbHiddenStates/nbLabels;
	int nbLabelEdge = nbHiddenStates*nbLabels;
	//Assigned Label edge weights
	int activeLabel = 0;
	int nbAssignedHiddenStates = 0;
	int y =0;
	for(unsigned int i = length-nbLabelEdge; i < length; i++)
	{
		if(y==activeLabel)
		{
			w[i]=abs(minRangeWeights);
			nbAssignedHiddenStates++;
		}
		else
			w[i]=-abs(minRangeWeights);

		y++;
		if(y>=nbLabels)
		{
			y = 0;
			if(nbAssignedHiddenStates >= nbHiddenStatesPerLabel)
			{
				activeLabel++;
				nbAssignedHiddenStates= 0;
			}
		}

	}

	pModel->setWeights(w);
}

void Toolbox::initWeightsPerceptron(DataSet& )
{
	throw HcrfNotImplemented("Perceptron weight initialisation not available for all toolboxes");
}

void Toolbox::calculateGlobalMean(DataSet &X,dVector& mean)
{
	dVector seqSum;
	int nbElements = 0;

	//Calculate mean
	for(int i = 0;i < (int)X.size() ;i++)
	{
		X.at(i)->getPrecomputedFeatures()->rowSum(seqSum);
		mean.add(seqSum);
		nbElements+=X.at(i)->getPrecomputedFeatures()->getWidth();
	}
	mean.multiply(1.0/(double)nbElements);

}


void Toolbox::calculateGlobalMeanAndStd(DataSet &X,dVector& mean,dVector& stdDev)
{
	calculateGlobalMean(X,mean);
	int nbElements = 0;

	//Calculate standard deviation
	stdDev.set(0);
	for(int i = 0;i < (int)X.size() ;i++)
	{
		double* pData = X.at(i)->getPrecomputedFeatures()->get();
		int Width = X.at(i)->getPrecomputedFeatures()->getWidth();
		int Height = X.at(i)->getPrecomputedFeatures()->getHeight();

		for(int col=0; col < Width; col++)
		{
			double* pStdDev = stdDev.get();
			double* pMean = mean.get();
			for(int row = 0; row < Height;row++)
			{
				*pStdDev += (*pData-*pMean) * (*pData-*pMean);
				pStdDev++;
				pData++;
				pMean++;
			}
		}
		nbElements+=Width;
	}
	stdDev.multiply(1.0/(double)nbElements);
	stdDev.eltSqrt();
}



int Toolbox::getMaxNbIteration()
{
	if(pOptimizer)
		return pOptimizer->getMaxNumIterations();
	else
		return 0;
}

double Toolbox::getRegularizationL2()
{
	if(pModel)
		return pModel->getRegL2Sigma();
	else
		return 0.0;
}

double Toolbox::getRegularizationL1()
{
	if(pModel)
		return pModel->getRegL1Sigma();
	else
		return 0.0;
}

void Toolbox::setMaxNbIteration(int maxit)
{
	if(pOptimizer)
		pOptimizer->setMaxNumIterations(maxit);
}

void Toolbox::setRegularizationL2(double regFactorL2, eFeatureTypes typeFeature)
{
	if(pModel)
		pModel->setRegL2Sigma(regFactorL2, typeFeature);
}

void Toolbox::setRegularizationL1(double regFactorL1, eFeatureTypes typeFeature)
{
	if(pModel)
		pModel->setRegL1Sigma(regFactorL1, typeFeature);
}

int Toolbox::getDebugLevel()
{
	if(pModel)
		return pModel->getDebugLevel();
	else
		return -1;
}
void Toolbox::setDebugLevel(int newDebugLevel)
{
	if(pModel)
		pModel->setDebugLevel(newDebugLevel);
}


Model* Toolbox::getModel()
{
	return pModel;
}

FeatureGenerator* Toolbox::getFeatureGenerator()
{
	return pFeatureGenerator;
}

Optimizer* Toolbox::getOptimizer()
{
	return pOptimizer;
}

void Toolbox::load(char* filenameModel, char* filenameFeatures)
{
	pFeatureGenerator->load(filenameFeatures);
	pModel->load(filenameModel);
	
}

void Toolbox::save(char* filenameModel, char* filenameFeatures)
{
	pModel->save(filenameModel);
	pFeatureGenerator->save(filenameFeatures);
}

void Toolbox::validate(DataSet& dataTrain, DataSet& dataValidate, double& optimalRegularisation, char* filenameStats)
{
	double MaxF1value = 0.0;
	for(int r = -1; r <= 2;r ++)
	{
		double regFactor = pow(10.0,r); 
		setRegularizationL2(regFactor);
		train(dataTrain);
		double F1Value = test(dataValidate,NULL,filenameStats);
		if(F1Value > MaxF1value)
		{
			MaxF1value = F1Value;
			optimalRegularisation = regFactor;
		}
	}
}

featureVector* Toolbox::getAllFeatures(DataSet &X)
{
	if(pFeatureGenerator->getNumberOfFeatures() == 0)
		initModel(X);
	return pFeatureGenerator->getAllFeatures(pModel,X.getNumberofRawFeatures());
}





#ifdef _OPENMP
void Toolbox::set_num_threads(int nt){
	omp_set_num_threads(nt);
}
#else
void Toolbox::set_num_threads(int){
//Do nothing if not OpenMP
}
#endif
