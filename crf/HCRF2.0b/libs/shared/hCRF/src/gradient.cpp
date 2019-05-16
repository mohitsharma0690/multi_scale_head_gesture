//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Gradient
// Component
//
//	February 2, 2006

#include "gradient.h"
#include <exception>

Gradient::Gradient(InferenceEngine* infEngine, FeatureGenerator* featureGen)
	:pInfEngine(infEngine), pFeatureGen(featureGen)
{
	nbThreadsMP = 1;
	vecFeaturesMP = new featureVector[1];
	localGrads = new dVector[1];
}

Gradient::Gradient(const Gradient& other)
	:pInfEngine(other.pInfEngine), pFeatureGen(other.pFeatureGen)
{
	nbThreadsMP = other.nbThreadsMP;
	if (nbThreadsMP)
	{
		vecFeaturesMP = new featureVector[nbThreadsMP];
		localGrads = new dVector[nbThreadsMP];
	}
}

Gradient& Gradient::operator=(const Gradient& other)
{
	pInfEngine = other.pInfEngine;
	pFeatureGen = other.pFeatureGen;
	nbThreadsMP = other.nbThreadsMP;
	if (vecFeaturesMP)
		delete [] vecFeaturesMP;
	if(localGrads)
		delete [] localGrads;
	if(nbThreadsMP)
	{
		vecFeaturesMP = new featureVector[nbThreadsMP];
		localGrads = new dVector[nbThreadsMP];

	}
	return *this;
}

Gradient::~Gradient()
{
	if(vecFeaturesMP)
	{
		delete [] vecFeaturesMP;
		vecFeaturesMP = 0;
	}
	if(localGrads)
	{
		delete [] localGrads;
		localGrads = 0;
	}
	nbThreadsMP = 0;
}

void Gradient::setMaxNumberThreads(int maxThreads)
{
	if (nbThreadsMP < maxThreads)
	{
		nbThreadsMP = maxThreads;

		if (vecFeaturesMP)
			delete []vecFeaturesMP;
		vecFeaturesMP = new featureVector[nbThreadsMP];

		if(localGrads)
			delete[]localGrads;
		localGrads = new dVector[nbThreadsMP] ;
	}

}

double Gradient::computeGradient(dVector& vecGradient, Model* m, DataSet* X)
{
	//Check the size of vecGradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	double ans = 0.0;
	int TID = 0;
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);
	else
		vecGradient.set(0);
	// Initialize the buffers (vecFeaturesMP) for each thread
#ifdef _OPENMP
	setMaxNumberThreads(omp_get_max_threads());
	pInfEngine->setMaxNumberThreads(omp_get_max_threads());
	pFeatureGen->setMaxNumberThreads(omp_get_max_threads());
#endif
	for(int t=0;t<nbThreadsMP;t++)
	{
		if(localGrads[t].getLength() != nbFeatures)
			localGrads[t].resize(1,nbFeatures,0);
		else
			localGrads[t].set(0);
	}

////////////////////////////////////////////////////////////
// Start of parallel Region

	// Some weird stuff in gcc 4.1, with openmp 2.5 support
#if ((_OPENMP == 200505) && __GNUG__)
#pragma omp parallel \
	shared(X, m, ans, nbFeatures, std::cout)	\
	private(TID) \
	default(none)
#else
#pragma omp parallel \
	shared(vecGradient, X, m, ans, nbFeatures, std::cout)	\
	private(TID) \
	default(none)
#endif
	{
#ifdef _OPENMP 
		TID = omp_get_thread_num();
#endif
		// Create a temporary gradient
		double localSum = 0;

#ifdef WITH_SEQUENCE_WEIGHTS
		dVector tmpVecGradient(nbFeatures);
#endif

#pragma omp for
		// we can use unsigned if we have openmp 3.0 support (_OPENMP>=200805).
#ifdef _OPENMP 
    #if _OPENMP >= 200805
		for(unsigned int i = 0; i< X->size(); i++){
    #else
	    for(int i = 0; i< X->size(); i++){
    #endif
#else
		for(unsigned int i = 0; i< X->size(); i++){
#endif
			if (m->getDebugLevel() >=2){
#pragma omp critical(output)
				std::cout << "Thread "<<TID<<" computes gradient for sequence " 
						  << i <<" out of " << (int)X->size() 
						  << " (Size: " <<  X->at(i)->length() << ")" << std::endl;
			}
			DataSequence* x = X->at(i);
#ifdef WITH_SEQUENCE_WEIGHTS
			tmpVecGradient.set(0);
			localSum += computeGradient(tmpVecGradient, m, x) * x->getWeightSequence();
			if(x->getWeightSequence() != 1.0)
				tmpVecGradient.multiply(x->getWeightSequence());
			localGrads[TID].add(tmpVecGradient);
#else
			localSum += computeGradient(localGrads[TID], m, x);// * x->getWeightSequence();
#endif
		}
#pragma omp critical (reduce_sum)
		// We now put togheter the sums
		{
			if( m->getDebugLevel() >= 2){
				std::cout<<"Thread "<<TID<<" update sums"<<std::endl;
			}
			ans += localSum;
			vecGradient.add(localGrads[TID]);
		}
	} 
	
// End of parallel Region
////////////////////////////////////////////////////////////

	// because we are minimizing -LogP
	vecGradient.negate();

	// Add the regularization term
	double sigmaL2Square = m->getRegL2Sigma()*m->getRegL2Sigma();
	if(sigmaL2Square != 0.0f) {
		if (m->getDebugLevel() >= 2){
			std::cout << "Adding L2 norm gradient\n";
		}
		for(int f = 0; f < nbFeatures; f++) {
			vecGradient[f] += (*m->getWeights())[f]/sigmaL2Square;
		}
		double weightNorm = m->getWeights()->l2Norm(false);
		ans += weightNorm / (2.0*m->getRegL2Sigma()*m->getRegL2Sigma());	
	}
	return ans;
}


