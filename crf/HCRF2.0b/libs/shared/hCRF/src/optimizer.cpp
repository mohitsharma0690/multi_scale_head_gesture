//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Optimizer
// Component
//
//	February 2, 2006

#include "optimizer.h"



Optimizer::Optimizer() : maxit(-1),
lastNbIterations(-1),
lastFunctionError(-1),
lastNormGradient(-1)
{
}

Optimizer::~Optimizer()
{
}


void Optimizer::optimize(Model* m, DataSet* X,
                         Evaluator* eval, Gradient* grad)
{	
}

void Optimizer::optimize(Model* m, DataSet* X,
                         Evaluator* eval, GradientPerceptron* grad)
{
}

void Optimizer::setMaxNumIterations(int maxiter)
{
	maxit = maxiter;
}


int Optimizer::getMaxNumIterations()
{
	return maxit;
}

int Optimizer::getLastNbIterations()
{
	return lastNbIterations;
}

double Optimizer::getLastFunctionError()
{
	return lastFunctionError;
}

double Optimizer::getLastNormGradient()
{
	return lastNormGradient;
}
