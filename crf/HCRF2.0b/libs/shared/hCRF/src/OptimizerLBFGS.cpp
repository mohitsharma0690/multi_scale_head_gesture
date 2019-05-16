#include "optimizer.h"

#ifdef USELBFGS

#include "lbfgs.h"


OptimizerLBFGS::~OptimizerLBFGS()
{
	// Do nothing as the Optimizer does not own any of the pointer
}

OptimizerLBFGS::OptimizerLBFGS() 
: Optimizer()
, currentModel(NULL)
, currentDataset(NULL)
, currentEvaluator(NULL)
, currentGradient(NULL)
{
}

// We simply copy the pointer because Optimizer does not own them
OptimizerLBFGS::OptimizerLBFGS(const OptimizerLBFGS& other) 
: Optimizer()
, currentModel(other.currentModel)
, currentDataset(other.currentDataset)
, currentEvaluator(other.currentEvaluator)
, currentGradient(other.currentGradient)
{
   throw std::logic_error("Optimizer should not be copied");
}





void OptimizerLBFGS::optimize(Model* m, DataSet* X,Evaluator* eval, 
							  Gradient* grad)
{
	currentModel = m;
	currentDataset = X;
	currentEvaluator = eval;
	currentGradient= grad;
	int nbWeights = currentModel->getWeights()->getLength();
	vecGradient.create(nbWeights);

	lbfgsfloatval_t fx;
	lbfgsfloatval_t *m_x = lbfgs_malloc(nbWeights);
	lbfgs_parameter_t param;

	if (m_x == NULL) {
		throw std::bad_alloc();
	}

	memcpy(m_x,currentModel->getWeights()->get(),nbWeights*sizeof(double));

	/* Initialize the parameters for the L-BFGS optimization. */
	lbfgs_parameter_init(&param);
	if(m->getRegL1Sigma() != 0.0)
	{
		param.orthantwise_c = m->getRegL1Sigma();
		param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
		if(m->getRegL1FeatureTypes() != allTypes)
		{
			int indexFirstFeature = -1;
			int indexLastFeature = -1;
			// IMPORTANT: Edge features should be contiguous; otherwise
			//    indexLastFeatures will include nodeFeatures Same thing is true
			//    for node Features.
			for( std::list<FeatureType*>::iterator itFeature = eval->pFeatureGen->getListFeatureTypes().begin();
				itFeature != eval->pFeatureGen->getListFeatureTypes().end(); itFeature++)
			{
				if( (m->getRegL1FeatureTypes() == edgeFeaturesOnly && (*itFeature)->isEdgeFeatureType() ) ||
					(m->getRegL1FeatureTypes() == nodeFeaturesOnly && !(*itFeature)->isEdgeFeatureType())    )
				{
					if (indexFirstFeature == -1)
						indexFirstFeature = (*itFeature)->getIdOffset();
					indexLastFeature = (*itFeature)->getIdOffset() + (*itFeature)->getNumberOfFeatures() - 1;
				}
			}
			param.orthantwise_start = indexFirstFeature;
			param.orthantwise_end = indexLastFeature;
		}
	}
	if (maxit >= 0)
		param.max_iterations = maxit;
	param.delta = param.epsilon;
	param.past = 10;
	lastNbIterations = 0;

	/*
		Start the L-BFGS optimization; this will invoke the callback functions
		evaluate() and progress() when necessary.
	 */
	int ret = lbfgs(nbWeights, m_x, &fx, _evaluate, _progress, this, &param);

	/* Report the result. */
	if(currentModel->getDebugLevel() >= 1)
	{
		std::cout << "L-BFGS optimization terminated with status code = " << ret <<std::endl;
		std::cout << "  fx = " << fx << std::endl;
	}

	// Save the optimal weights
	memcpy(vecGradient.get(),&m_x[0],nbWeights*sizeof(double));
	currentModel->setWeights(vecGradient);

	dVector tmpWeights = *(currentModel->getWeights());
	tmpWeights.transpose();
	tmpWeights.multiply(*currentModel->getWeights());
	lastNormGradient = tmpWeights[0];
	lbfgs_free(m_x);
}

// Compute gradient and evaluate error function
double OptimizerLBFGS::Eval(const lbfgsfloatval_t *x, lbfgsfloatval_t *g, 
							const int n)
{
	dVector dgrad(n);
	// Copy current weights
	memcpy(vecGradient.get(),&x[0],n*sizeof(double));
	currentModel->setWeights(vecGradient);
	//Compute gradient
	if(currentModel->getDebugLevel() >= 2){
		std::cout << "Compute gradient and error..." << std::endl;
	}
	double f = currentGradient->computeGradient(dgrad, currentModel,
												currentDataset);
	//Copy gradient
	memcpy(g,dgrad.get(),dgrad.getLength()*sizeof(double));
	return f;
}
int OptimizerLBFGS::progress( const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
							  const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, 
							  const lbfgsfloatval_t step, int n, int k, int ls)
{
	if(currentModel->getDebugLevel() >= 1) {
		std::cout << "Iteration " << k << std::endl;
		std::cout << "  fx = " << fx << std::endl;
		std::cout << "  xnorm " << xnorm << ", gnorm = " << gnorm ;
		std::cout << ", step = " << step << "line search = "<<ls<<std::endl;
		if(currentModel->getDebugLevel() >= 3) {
			std::cout<<"x = [";
			for (int i = 0; i<n; i++) {
				if(i) std::cout<<", ";
				std::cout<<x[i];
			}
			std::cout<<"]"<<std::endl;
			std::cout<<"g = [";
			for (int i = 0; i<n; i++) {
				if(i) std::cout<<", ";
				std::cout<<g[i];
			}
			std::cout<<"]"<<std::endl;
		}
		std::cout<<std::endl;
	}
	lastNbIterations++;
	lastFunctionError = fx;
	lastNormGradient = gnorm;
	return 0;
}


#endif
