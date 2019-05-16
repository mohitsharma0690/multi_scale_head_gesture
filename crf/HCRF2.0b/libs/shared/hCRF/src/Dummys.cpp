#include "evaluator.h"
#include "gradient.h"

/////////////////////////////////////////////////////////////////////
// EvaluatorDD Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

EvaluatorDD::EvaluatorDD(InferenceEngine *infEngine,
						 FeatureGenerator *featureGen,
						 dVector* pMu)
:Evaluator(infEngine, featureGen)
, mu(2, COLVECTOR)
{
	mu[0] = 1;
	mu[1] = 3;

	if(pMu) {
		mu = *pMu;
	}

	mu.negate();
}

EvaluatorDD::~EvaluatorDD()
{
	//does nothing
}

// *
// Public Methods
// *

double EvaluatorDD::computeError(DataSet*, Model *m)
{
	dVector x, y;
	x = *(m->getWeights());
	x.add(mu);
	y = x;
	x.transpose();
	x.multiply(y);
	return -exp(-0.5*x[0]);
}

double EvaluatorDD::computeError(DataSequence*, Model *m)
{
	dVector x, y;
	x = *(m->getWeights());
	x.add(mu);
	y = x;
	x.transpose();
	x.multiply(y);
	return -exp(-0.5*x[0]);
}

/////////////////////////////////////////////////////////////////////
// EvaluatorDD Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

GradientDD::GradientDD(InferenceEngine* infEngine, 
					   FeatureGenerator* featureGen, dVector* pMu)
: Gradient(infEngine, featureGen)
, mu(2, COLVECTOR)
{
	mu[0] = 1;
	mu[1] = 3;

	if(pMu) {
		mu = *pMu;
	}
	mu.negate();
}

// *
// Public Methods
// *

double GradientDD::computeGradient(dVector& vecGradrient, Model* m,DataSet *)
{
	dVector tmpVec;
	vecGradrient = *(m->getWeights());
	vecGradrient.add(mu);
	tmpVec = vecGradrient;
	tmpVec.transpose();
	tmpVec.multiply(vecGradrient);
	double f = exp(-0.5*tmpVec[0]);
	vecGradrient.multiply(f);
	return f;
}

double GradientDD::computeGradient(dVector& vecGradrient, Model* m,DataSequence*)
{
	dVector tmpVec;
	vecGradrient = *(m->getWeights());
	vecGradrient.add(mu);
	tmpVec = vecGradrient;
	tmpVec.transpose();
	tmpVec.multiply(vecGradrient);
	double f = exp(-0.5*tmpVec[0]);
	vecGradrient.multiply(f);
	return f;
}
