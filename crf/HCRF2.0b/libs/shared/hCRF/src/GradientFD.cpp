//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Finite Difference
// Gradient Component
//
//	February 2, 2006

#include "gradient.h"
#include <cmath>

//-------------------------------------------------------------
// Gradient Class
//-------------------------------------------------------------

/*
	Constructor/Destructor
*/

GradientFD::GradientFD(InferenceEngine* infEngine, FeatureGenerator* featureGen, Evaluator* evaluator)
	: Gradient(infEngine, featureGen), pEvaluator(evaluator)
{}

GradientFD::GradientFD(const GradientFD& other)
	:Gradient(other), pEvaluator(other.pEvaluator)
{}

GradientFD& GradientFD::operator=(const GradientFD& other)
{
	Gradient::operator=(other);
	pEvaluator = other.pEvaluator;
	return *this;
}

/*
	Public Methods
*/

// Selection of h from Numerical Recipes in C++, page 392
double GradientFD::computeGradient(dVector &vecGradient, Model *m, DataSequence *X)
{
	// get pointer to weight parameter vector
	dVector oldWeights;
	oldWeights.set(*m->getWeights());
	dVector x;
	x.set(*m->getWeights());

	// compute gradient
	vecGradient.create(x.getLength());
	int i;
	double y1, y2, h, temp, EPS=1.0e-6;	// Approximate cube root of machine precision.
	for(i = 0; i < x.getLength(); i++) {
		temp = x[i];
        h = EPS*fabs(x[i]);
		if(h == 0) h=EPS;
		x[i] = temp+h;	// Trick to reduce finite precision error.
		h = x[i]-temp;

		m->setWeights(x);
		y2 = pEvaluator->computeError(X, m);

		x.setValue(i, x.getValue(i) - 2*h);
		m->setWeights(x);
		y1 = pEvaluator->computeError(X, m);


		vecGradient[i] = (y2-y1)/(2*h);

		x[i] = temp;
	}
	m->setWeights(oldWeights);
	return pEvaluator->computeError(X, m);
}

// Selection of h from Numerical Recipes in C++, page 392
double GradientFD::computeGradient(dVector &vecGradient, Model *m, DataSet *X)
{
	// get pointer to weight parameter vector
	dVector oldWeights;
	oldWeights.set(*m->getWeights());
	dVector x;
	x.set(*m->getWeights());

	// compute gradient
	vecGradient.create(x.getLength());
	int i;
	double y1, y2, h, temp, EPS=1.0e-6;	// Approximate cube root of machine precision.
	for(i = 0; i < x.getLength(); i++) {
		temp = x[i];
        h = EPS*fabs(x[i]);
		if(h == 0) h=EPS;
		x[i] = temp+h;	// Trick to reduce finite precision error.
		h = x[i]-temp;

		m->setWeights(x);
		y2 = pEvaluator->computeError(X, m);

		x.setValue(i, x.getValue(i) - 2*h);
		m->setWeights(x);
		y1 = pEvaluator->computeError(X, m);

		vecGradient[i] = (y2-y1)/(2*h);

		x[i] = temp;
	}
	m->setWeights(oldWeights);
	double sigmaL2Square = m->getRegL2Sigma()*m->getRegL2Sigma();
	if(sigmaL2Square != 0.0)
	{
		for(int f = 0; f < x.getLength(); f++)
		{
			vecGradient[f] += (*m->getWeights())[f]/sigmaL2Square;
		}
	}
	return pEvaluator->computeError(X, m);
}
