#include "optimizer.h"
#include "uncoptim.h"

// We define a helper class for the optimisation
class UnconstrainedOptimizer : public UnconstrainedOptim
{
public:
	UnconstrainedOptimizer();
	void F();
	void G();
	// We want to be able to set and retrieve those pointer easily
	dVector vecGradient;
	Model* currentModel;
	DataSet* currentDataset;
	Evaluator* currentEvaluator;
	Gradient* currentGradient;
};

// We set up our helper class
UnconstrainedOptimizer::UnconstrainedOptimizer()
: UnconstrainedOptim(0.0,0.0,RSSEARCH)
{
}

// Compute error function
void UnconstrainedOptimizer::F()
{
	memcpy(vecGradient.get(),x,n*sizeof(double));
	currentModel->setWeights(vecGradient);

	if(currentModel->getDebugLevel() >= 2)
		std::cout << "Compute error..." << std::endl;
	f = currentEvaluator->computeError(currentDataset, currentModel);
	if(currentModel->getDebugLevel() >= 3)
	{
//		printf("  Iteration # = %i   Nb error eval = %i   Nb gradient eval =  %i\n\n", cnls, cnf, cng);
//		printf("F = %-0.10lg\n",f);
		std::cout << "  Iteration # = " << cnls << "   Nb error eval = " <<cnf << "   Nb gradient eval =  " << cng <<std::endl << std::endl;
		std::cout << "F = " << f << std::endl;
		std::cout.flush();
	}
	
}

//Compute gradient
void UnconstrainedOptimizer::G()
{
	dVector dgrad(n);
	memcpy(vecGradient.get(),x,n*sizeof(double));
	currentModel->setWeights(vecGradient);
	if(currentModel->getDebugLevel() >= 2)
		std::cout << "Compute gradient..." << std::endl;
	currentGradient->computeGradient(dgrad, currentModel,currentDataset);
	memcpy(g,dgrad.get(),n*sizeof(double));
}

// We can now implement the optimizer method of our optimizer
OptimizerUncOptim::OptimizerUncOptim(typeOptimizer defaultOptimizer)
: Optimizer()
, internalOptimizer(NULL)
, optimizer(defaultOptimizer)
{
}

OptimizerUncOptim::OptimizerUncOptim(const OptimizerUncOptim& other)
: Optimizer()
, internalOptimizer(NULL)
, optimizer(other.optimizer)
{
	throw std::logic_error("Optimizer should not be copied");
}

OptimizerUncOptim::~OptimizerUncOptim()
{
}

void OptimizerUncOptim::optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad)
{
	if (internalOptimizer == NULL)
		internalOptimizer = new UnconstrainedOptimizer;

	internalOptimizer->currentModel = m;
	internalOptimizer->currentDataset = X;
	internalOptimizer->currentEvaluator = eval;
	internalOptimizer->currentGradient= grad;
	

	// Initialize optimizer
	internalOptimizer->setDimension(m->getWeights()->getLength());	
	internalOptimizer->vecGradient.create(m->getWeights()->getLength());
	internalOptimizer->setMaxIterations(maxit);

	// Set the initial weights
	memcpy(internalOptimizer->x, m->getWeights()->get(), internalOptimizer->n*sizeof(double));

	// Call the right optimizer
	int opt = 0;
	switch(optimizer)
	{
	case optimBFGS:
		opt = internalOptimizer->BFGSoptimize();
		break;

	case optimDFP:
        opt = internalOptimizer->DFPoptimize();
		break;

	case optimFR:
		opt =  internalOptimizer->FRoptimize(0);
		break;

	case optimFRwithReset:
		opt = internalOptimizer->FRoptimize(1);
		break;

	case optimPR:
		opt = internalOptimizer->PRoptimize(0);
		break;

	case optimPRwithReset:
		opt = internalOptimizer->PRoptimize(1);
		break;
	}
	// Save the optimal weights
	memcpy(internalOptimizer->vecGradient.get(), internalOptimizer->x, internalOptimizer->n*sizeof(double));
	m->setWeights(internalOptimizer->vecGradient);

	dVector tmpWeights = *(m->getWeights());
	tmpWeights.transpose();
	tmpWeights.multiply(*m->getWeights());
	lastNormGradient = tmpWeights[0];

	lastNbIterations = internalOptimizer->cnls;
	lastFunctionError = internalOptimizer->f;

	if(m->getDebugLevel() >= 1)
	{
		std::cout << "F = " << lastFunctionError << "  |w| = " << lastNormGradient << std::endl;
		std::cout << "  Iteration # = " << lastNbIterations << "   Nb error eval = " << internalOptimizer->cnf << "   Nb gradient eval =  " << internalOptimizer->cng <<std::endl << std::endl;
	}
}


