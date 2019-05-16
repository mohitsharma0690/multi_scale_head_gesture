#include "optimizer.h"
#include "cg_descent.h"
#include <iostream>

static Model* currentModel=NULL;
static DataSet* currentDataset=NULL;
static Evaluator* currentEvaluator=NULL;
static Gradient* currentGradient=NULL;


OptimizerCG::OptimizerCG() : Optimizer()
{
}

OptimizerCG::~OptimizerCG()
{
}


void OptimizerCG::optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad)
{
	currentModel = m;
	currentDataset = X;
	currentEvaluator = eval;
	currentGradient= grad;
	double* weights;
	//weights = currentModel->getWeights()->get();

	int status = -1;
	cg_stats Stats;
	double* work;

	work = (double *) malloc(4*currentModel->getWeights()->getLength()*sizeof(double));
	weights = (double *) malloc (currentModel->getWeights()->getLength()*sizeof(double)) ;
	double step = 0.001;
	memcpy(weights,currentModel->getWeights()->get(),currentModel->getWeights()->getLength()*sizeof(double));

	// Call
	status = cg_descent(1.e-8, weights, currentModel->getWeights()->getLength(), callbackComputeError, 
						callbackComputeGradient,work, step, &Stats, maxit);

	dVector vecGradient(currentModel->getWeights()->getLength());
	memcpy(vecGradient.get(),weights,currentModel->getWeights()->getLength()*sizeof(double));
	currentModel->setWeights(vecGradient);

	dVector tmpWeights = *(currentModel->getWeights());
	tmpWeights.transpose();
	tmpWeights.multiply(*currentModel->getWeights());
	lastNormGradient = tmpWeights[0];

	lastNbIterations = Stats.iter;
	lastFunctionError = Stats.f;
	

	if(currentModel->getDebugLevel() >= 1)
	{
		std::cout << "F = " << lastFunctionError << "  |w| = " << lastNormGradient << std::endl;
		std::cout<<"  Iteration # = "<<lastNbIterations<<"   Nb error eval = "<<Stats.nfunc;
		std::cout<< "   Nb gradient eval =  " << Stats.ngrad <<std::endl << std::endl;
		if(currentModel->getDebugLevel() >= 3){
			std::cout<<"X = "<<currentModel->getWeights();
		}
	}
	free(work);
	//TODO:: Check if we want to also free weights hsalamin 2010.02.05
}

double OptimizerCG::callbackComputeError(double* weights)
{
	dVector vecGradient(currentModel->getWeights()->getLength());
	memcpy(vecGradient.get(),weights,currentModel->getWeights()->getLength()*sizeof(double));
	currentModel->setWeights(vecGradient);
//	currentModel->getWeights()->set(weights);
	if(currentModel->getDebugLevel() >= 2){
		std::cout << "Compute error... "  << std::endl;
	}
	double errorVal = currentEvaluator->computeError(currentDataset, currentModel);
	return errorVal;
	
}

void OptimizerCG::callbackComputeGradient(double* gradient, double* weights)
{
	dVector dgrad(currentModel->getWeights()->getLength());
	memcpy(dgrad.get(),weights,currentModel->getWeights()->getLength()*sizeof(double));
	currentModel->setWeights(dgrad);
	dgrad.set(0);
	if(currentModel->getDebugLevel() >= 2){
		std::cout<<"Compute gradient... "<<std::endl;
	}
	currentGradient->computeGradient(dgrad, currentModel,currentDataset);
	memcpy(gradient,dgrad.get(),dgrad.getLength()*sizeof(double));
}


