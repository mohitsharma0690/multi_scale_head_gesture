//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Feature Generator
// Component
//
//	February 2, 2006

#include "featuregenerator.h"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

featureVector::featureVector(): realSize(8), capacity(8), pFeatures(NULL)
{
	pFeatures = new feature[capacity];
	memset(pFeatures, 0, capacity*sizeof(feature));
}

featureVector::featureVector(const featureVector& source): 
	realSize(source.realSize), capacity(source.capacity), pFeatures(NULL)
{
	/* A good copy constructor is important to ensure that the
	memory is only freed once (You dont want a copy to point to the same
	area of memory */
	pFeatures = new feature[capacity];
	memcpy(pFeatures, source.pFeatures, capacity*sizeof(feature));
}

featureVector& featureVector::operator = (const featureVector& source)
{
	if(pFeatures) {
		delete[] pFeatures;
	}
	realSize = source.realSize;
	capacity = source.capacity;
	pFeatures = new feature[capacity];
	memcpy(pFeatures, source.pFeatures, capacity*sizeof(feature));
	return *this;
}

featureVector::~featureVector()
{
	if(pFeatures) {
		delete[] pFeatures;
		pFeatures = NULL;
	}
}

void featureVector::resize(int newSize)
/** This function is used to resize the vector. To step,
 first we check if need to grow the vector (doubling size each time).
 Next we initilise the new space.
**/
{
	// Easy case: We dont need to allocate any new memory. 
	if (newSize <= capacity)
	{
		realSize = newSize;
		return;
	}
	//While the capacity is too small (*2 is cheap)
	while (newSize > capacity)
		capacity *=2;
	feature* tmpNewPointer = new feature[capacity];
	memcpy(tmpNewPointer,pFeatures,realSize*sizeof(feature));
	//We also initialise to zero the extra memory
	memset(tmpNewPointer+realSize, 0, (capacity-realSize)*sizeof(feature));
	delete[] pFeatures;
	pFeatures = tmpNewPointer;
	realSize = newSize;
}

int featureVector::size()
{
	return realSize;
}

void featureVector::clear()
{
	resize(0);
}

feature* featureVector::getPtr()
{
	return pFeatures;
}

feature* featureVector::addElement()
{
	resize(realSize+1);
	return pFeatures+realSize-1;
}


FeatureType::FeatureType():idOffset(0), nbFeatures(0), 
						   idOffsetPerLabel(0), nbFeaturesPerLabel(0),
						   strFeatureTypeName(), featureTypeId(0)
{
}

FeatureType::~FeatureType()
{
}

void FeatureType::setIdOffset(int offset, int seqLabel)
{
	if(seqLabel == -1 || idOffsetPerLabel.getLength() == 0)
		idOffset = offset;
	else
		idOffsetPerLabel[seqLabel] = offset;
}

void FeatureType::init(const DataSet&, const Model& m)
/* Some of the derived class need access to the DataSet for initilisation */
{
	if (m.getNumberOfSequenceLabels()>0) {
		idOffsetPerLabel.create(m.getNumberOfSequenceLabels());
		nbFeaturesPerLabel.create(m.getNumberOfSequenceLabels());
	}
}

void FeatureType::computeFeatureMask(iMatrix& matFeatureMask, const Model& m)
{
	int firstOffset = idOffset;
	int lastOffset = idOffset + nbFeatures;
	int nbLabels = m.getNumberOfSequenceLabels();

	for(int i = firstOffset; i < lastOffset; i++)
		for(int j = 0; j < nbLabels; j++)
			matFeatureMask(i,j) = 1;
}


int FeatureType::getNumberOfFeatures(int seqLabel)
{
	if (seqLabel == -1 || nbFeaturesPerLabel.getLength() == 0)
		return nbFeatures;
	else
		return nbFeaturesPerLabel[seqLabel];
}

bool FeatureType::isEdgeFeatureType()
{
	return false;
}

iVector& FeatureType::getOffsetPerLabel()
{
	return idOffsetPerLabel;
}

iVector& FeatureType::getNbFeaturePerLabel()
{
	return nbFeaturesPerLabel;
}

char* FeatureType::getFeatureTypeName()
{
	return (char*)strFeatureTypeName.c_str();
}

int FeatureType::getFeatureTypeId()
{
	return featureTypeId;
}

void FeatureType::setNumberOfFeatures(int nFeatures)
{
	nbFeatures = nFeatures;
}

void FeatureType::setOffsetPerLabel(const iVector& newOffsetPerLabel)
{
	idOffsetPerLabel = newOffsetPerLabel;
}

void FeatureType::setNbFeaturePerLabel(const iVector& newNbFeaturePerLabel)
{
	nbFeaturesPerLabel = newNbFeaturePerLabel;
}


void FeatureType::read(std::istream& is)
{
	is >> idOffset >> nbFeatures;
	idOffsetPerLabel.read(&is);
	nbFeaturesPerLabel.read(&is);
}

void FeatureType::write(std::ostream& os)
{
	os << idOffset << " " << nbFeatures << std::endl;
	idOffsetPerLabel.write(&os);
	nbFeaturesPerLabel.write(&os);
}


FeatureGenerator::FeatureGenerator():listFeatureTypes(), vecFeatures()
{
	nbThreadsMP = 1;
	vecFeaturesMP = new featureVector[1];

}

FeatureGenerator::~FeatureGenerator()
{
	clearFeatureList();
	if(vecFeaturesMP)
	{
		delete []vecFeaturesMP;
		vecFeaturesMP = 0;
		nbThreadsMP = 0;
	}
}

void FeatureGenerator::setMaxNumberThreads(int maxThreads)
{
	if (nbThreadsMP < maxThreads)
	{
		if (vecFeaturesMP)
			delete []vecFeaturesMP;
		nbThreadsMP = maxThreads;
		vecFeaturesMP = new featureVector[nbThreadsMP];
	}

}


void FeatureGenerator::addFeature(FeatureType* featureGen)
{
	if(featureGen->isEdgeFeatureType())
		listFeatureTypes.insert(listFeatureTypes.end(),featureGen);
	else
		listFeatureTypes.insert(listFeatureTypes.begin(),featureGen);
}

void FeatureGenerator::initFeatures(const DataSet& dataset, Model& m)
{
	int nbLabels = m.getNumberOfSequenceLabels();
	int offset = 0;
	iVector offsetPerLabel;
	if (nbLabels>0)
		offsetPerLabel.create(nbLabels);

	for(std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
		itFeature != listFeatureTypes.end(); itFeature++) {
		// Initialize the featureType
		(*itFeature)->init(dataset, m);
		// Set idOffset for the specific featureType
		(*itFeature)->setIdOffset(offset);
		for(int i = 0; i < nbLabels; i++)
			(*itFeature)->setIdOffset(offsetPerLabel[i],i);
		// compute offset for next featureType
		offset += (*itFeature)->getNumberOfFeatures();
		for(int i = 0; i < nbLabels; i++)
			offsetPerLabel[i] += (*itFeature)->getNumberOfFeatures(i);
	}
	//vecFeatures.resize(offset);
	if(nbLabels > 0) {
		// Compute FeatureMask
		iMatrix matFeatureMask(nbLabels, offset);
		for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
			itFeature != listFeatureTypes.end(); itFeature++) {
			(*itFeature)->computeFeatureMask(matFeatureMask,m);
		}
		m.setFeatureMask(matFeatureMask);
	}
}


void FeatureGenerator::clearFeatureList()
{
	for(std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
		itFeature != listFeatureTypes.end(); itFeature++)
	{
		delete *itFeature;
		*itFeature = NULL;
	}
	listFeatureTypes.clear();

}

#if defined(_VEC_FEATURES) || defined(_OPENMP)
void FeatureGenerator::getFeatures(featureVector& vecFeatures, DataSequence* X,
								   Model* m, int nodeIndex, int prevNodeIndex, 
								   int seqLabel)
{
	vecFeatures.clear();
	for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
		itFeature != listFeatureTypes.end(); itFeature++) {
		(*itFeature)->getFeatures(vecFeatures, X, m, nodeIndex, 
								  prevNodeIndex, seqLabel);
	}
}
#else
featureVector* FeatureGenerator::getFeatures(DataSequence* X, Model* m, 
											 int nodeIndex, int prevNodeIndex, 
											 int seqLabel)
{
	// The ptr returned is owned by the featureGenerator object. A call to
	// getAllFeatures will change the value stored in te destination of the
	// pointer. This is dangerous.
	vecFeatures.clear();
	for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
		itFeature != listFeatureTypes.end(); itFeature++) {
		(*itFeature)->getFeatures(vecFeatures, X, m, nodeIndex, 
								  prevNodeIndex, seqLabel);
	}
	return &vecFeatures;
}
#endif


featureVector* FeatureGenerator::getAllFeatures(Model* m, int nbRawFeatures)
{
	vecFeatures.clear();
	for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
		itFeature != listFeatureTypes.end(); itFeature++)
	{
		(*itFeature)->getAllFeatures(vecFeatures, m, nbRawFeatures);
	}
	return &vecFeatures;
}

int FeatureGenerator::getNumberOfFeatures(eFeatureTypes typeFeature, 
										  int seqLabel)
{
	int totalFeatures = 0;
	for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
		itFeature != listFeatureTypes.end(); itFeature++)
	{
		if( (typeFeature == allTypes) ||
			(typeFeature == edgeFeaturesOnly && 
			 (*itFeature)->isEdgeFeatureType() ) ||
			(typeFeature == nodeFeaturesOnly && 
			 !(*itFeature)->isEdgeFeatureType())) 
		{
			totalFeatures += (*itFeature)->getNumberOfFeatures(seqLabel);
		}
	}
	return totalFeatures;

}


double FeatureGenerator::evaluateLabels(DataSequence* X, Model* m, int seqLabel)
{
	dVector *w = m->getWeights(seqLabel);
	double phi = 0;
	//add node features
	int n = X->length(), i;
	feature* pFeature;
	iVector * s = X->getStateLabels();
#if !defined(_VEC_FEATURES) && !defined(_OPENMP)
	featureVector* vecFeaturesLocal;
#endif
#if defined(_OPENMP)
	int ThreadID = omp_get_thread_num();
	if (ThreadID >= nbThreadsMP)
		ThreadID = 0;
#else
	int ThreadID = 0;
#endif

	for(i = 0; i<n; i++) {
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		getFeatures(vecFeaturesMP[ThreadID], X, m, i, -1,seqLabel);
		pFeature = vecFeaturesMP[ThreadID].getPtr();
		for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++,pFeature++){
#else
		vecFeaturesLocal = getFeatures(X, m, i, -1,seqLabel);
		pFeature = vecFeaturesLocal->getPtr();
		for(int j = 0; j < vecFeaturesLocal->size(); j++,pFeature++){
#endif
			if(pFeature->nodeState==s->getValue(i)) {
				phi += w->getValue(pFeature->id) * pFeature->value;
			}
		}
	}
	//add edge features
	uMatrix adjMat; 
	m->getAdjacencyMatrix(adjMat, X);
	int row,col;
	for(col=0; col<n; col++) { // current node index
		for(row=0; row<=col; row++) { // previous node index
			if(adjMat.getValue(row,col)==0) {
				continue;
			}
#if defined(_VEC_FEATURES) || defined(_OPENMP)
			getFeatures(vecFeaturesMP[ThreadID], X, m, col, row,seqLabel);
			pFeature = vecFeaturesMP[ThreadID].getPtr();
			for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++, pFeature++) {
#else
			vecFeaturesLocal = getFeatures(X, m, col, row,seqLabel);
			pFeature = vecFeaturesLocal->getPtr();
			for(int j = 0; j < vecFeaturesLocal->size(); j++,pFeature++){
#endif
				if(pFeature->nodeState==s->getValue(col) && pFeature->prevNodeState==s->getValue(row)) {
					phi += w->getValue(pFeature->id) * pFeature->value;
				}
			}
		}
	}
	return phi;
}

std::list<FeatureType*>& FeatureGenerator::getListFeatureTypes()
{
	return listFeatureTypes;
}


void FeatureGenerator::load(char* pFilename)
{
	std::ifstream fileInput(pFilename);
	
	if (!fileInput.is_open())
	{
		std::cerr << "Can't find features definition file: " << pFilename << std::endl;
		return;
	}

	for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
		itFeature != listFeatureTypes.end(); itFeature++)
	{
		(*itFeature)->read(fileInput);
	}
	
	fileInput.close();
}

void FeatureGenerator::save(char* pFilename)
{
	std::ofstream fileOutput(pFilename);
	
	if (!fileOutput.is_open())
		return;

	for(std:: list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
		itFeature != listFeatureTypes.end(); itFeature++)
	{
		(*itFeature)->write(fileOutput);
	}
	fileOutput.close();
}

std::ostream& operator <<(std::ostream& out, const feature& f)
{
	out << "f[" << f.id <<"] : (i=" << f.prevNodeIndex << ", j=" << f.nodeIndex ;
	out << ", yi=" << f.prevNodeState << ", yj=" << f.nodeState << ", Y=" ;
	out << f.sequenceLabel <<") = " << f.value << std::endl;
	return out;
}

std::ostream& operator <<(std::ostream& out, featureVector& v)
{
	feature* data = v.getPtr();
	for (int i = 0; i< v.size();i++) {
		out<<(data[i]);
	}
	return out;
}

