//-------------------------------------------------------------
// Hidden Conditional Random Field Library - DataSet Component
//
//	February 02, 2006

#include "dataset.h"
using namespace std;


//-------------------------------------------------------------
// DataSequence Class
//-------------------------------------------------------------

//*
// Constructors and Deconstructor
//*

DataSequenceRealtime::DataSequenceRealtime()
{
	precompFeatures = 0;
	windowSize = 0;
	width = 0;
	height = 0;
	pos = 0;
	alpha = 0;
	ready = false;
	
}

DataSequenceRealtime::DataSequenceRealtime(int windowSize, int bufferLength, int height, int numberofLabels)
{
	init(windowSize, bufferLength,height, numberofLabels);
}

DataSequenceRealtime::~DataSequenceRealtime()
{
	if(alpha != NULL)
		delete alpha;
}

int DataSequenceRealtime::init(int windowSize, int bufferLength, int height, int numberofLabels)
{
	precompFeatures = new dMatrix(windowSize+bufferLength,height);
	this->windowSize = windowSize;
	this->width = windowSize+bufferLength;
	this->height = height;
	pos = 0;
	ready = false;
	alpha = 0;
	//alpha->create(numberofLabels);
	//alpha->set(0);
	return 1;
}

void DataSequenceRealtime::push_back(const dVector* const featureVector)
{
	for(int row=0; row<height; row++)	
		precompFeatures->setValue(row, pos, featureVector->getValue(row));	
	pos++;
	if(pos == width)
	{
		pos = 0;
		ready = true;
	}
}

int DataSequenceRealtime::getWindowSize()
{
	return windowSize;
}

dVector* DataSequenceRealtime::getAlpha()
{
	return alpha;
}

void DataSequenceRealtime::initializeAlpha(int height)
{
	alpha = new dVector(height);
	alpha->set(0);
}

int DataSequenceRealtime:: getPosition()
{
	return pos;
}

bool DataSequenceRealtime::isReady()
{
	return ready;
}

DataSequence::DataSequence()
{
	init();
}

DataSequence::DataSequence(const DataSequence& other)
{
	init();
	operator=(other);
}

DataSequence& DataSequence::operator=(const DataSequence&)
{
	throw HcrfNotImplemented("Copying DataSequence is not supported") ;
}

DataSequence::DataSequence(dMatrix* precomputedFeatures, iVector* stateLabels_, int seqLabel)
{
	init();
	precompFeatures = precomputedFeatures;
	stateLabels = stateLabels_;
	sequenceLabel = seqLabel;
}

void DataSequence::init()
{
	stateLabels = 0;
	statesPerNode = 0;
	sequenceLabel = 0;
	adjMat = 0;
	precompFeatures = 0;
	precompFeaturesSparse = 0;
	weightSequence = 1.0;
	estimatedSequenceLabel = 0;
	estimatedStateLabels = 0;
	estimatedProbabilitiesPerStates = 0;
}

DataSequence::~DataSequence()
{
	if(adjMat != NULL)
	{
		delete adjMat;
		adjMat = NULL;
	}
	if(stateLabels != NULL)
	{
		delete stateLabels;
		stateLabels = NULL;
	}
	if(statesPerNode != NULL)
	{
		delete statesPerNode;
		statesPerNode = NULL;
	}
	if (precompFeatures != NULL)
	{
		delete precompFeatures;
		precompFeatures = NULL;
	}
	if (estimatedStateLabels != NULL)
	{
		delete estimatedStateLabels;
		estimatedStateLabels = NULL;
	}
	if (estimatedProbabilitiesPerStates != NULL)
	{
		delete estimatedProbabilitiesPerStates;
		estimatedProbabilitiesPerStates = NULL;
	}
}

//*
// Public Methods
//*

int DataSequence::load(istream* isData, istream* isLabels, istream* isAdjMat, istream* isStatesPerNodes, istream* isDataSparse)
{ 
	if(isData == NULL && isLabels == NULL && isAdjMat == NULL && isStatesPerNodes == NULL)
		return 1;

	if(isData)
	{
		dMatrix* pNewMat = new dMatrix;
		if(pNewMat->read(isData)==0)
			precompFeatures = pNewMat;
		else
		{
			delete pNewMat;
			return 1;
		}
	}
	if(isLabels)
	{
		iVector* pNewVec = new iVector;
		if(pNewVec->read(isLabels)==0)
			stateLabels = pNewVec;
		else
		{
			delete pNewVec;
			return 1;
		}
	}
	if(isAdjMat)
	{
		uMatrix* pNewMat = new uMatrix;
		if(pNewMat->read(isAdjMat)==0)
			adjMat = pNewMat;
		else
		{
			delete pNewMat;
			return 1;
		}
	}
	if(isStatesPerNodes)
	{
		iMatrix* pNewMat = new iMatrix;
		if(pNewMat->read(isStatesPerNodes)==0)
			statesPerNode = pNewMat;
		else
		{
			delete pNewMat;
			return 1;
		}
	}
	if(isDataSparse)
	{
		dMatrixSparse* pNewMat = new dMatrixSparse;
		if(pNewMat->read(isDataSparse)==0)
			precompFeaturesSparse = pNewMat;
		else
		{
			delete pNewMat;
			return 1;
		}
	}
	return 0;
}


int	DataSequence::length() const
{
	if (precompFeatures != NULL)
		return precompFeatures->getWidth();
	if (precompFeaturesSparse != NULL)
		return (int)precompFeaturesSparse->getWidth();
	else
		return 0;
}


void DataSequence::equal(const DataSequence&)
{
	throw HcrfNotImplemented("Comparing DataSequence is not supported") ;
}

void DataSequence::setSequenceLabel(int seqLabel)
{
	sequenceLabel = seqLabel;
}

int DataSequence::getSequenceLabel() const
{
	return sequenceLabel;
}

void DataSequence::setStateLabels(iVector *v)
{
	if(stateLabels != NULL)
	{
		delete stateLabels;
		stateLabels = NULL;
	}
	stateLabels = v;
}

int DataSequence::getStateLabels(int nodeIndex) const
{
	return (*stateLabels)[nodeIndex];
}


iVector* DataSequence::getStateLabels() const
{
	return stateLabels;
}

void DataSequence::setAdjacencyMatrix(uMatrix* m)
{
	if(adjMat != NULL)
	{
		delete adjMat;
		adjMat = NULL;
	}
	adjMat = m;
}

void DataSequence::getAdjacencyMatrix(uMatrix& outMat) const
{
	if (adjMat == NULL) {
		outMat.resize(0,0);
	} else{
		outMat = *adjMat;
	}
}

void DataSequence::setPrecomputedFeatures(dMatrix* m)
{
	if(precompFeatures != NULL)
	{
		delete precompFeatures;
		precompFeatures = NULL;
	}
	precompFeatures = m;
}

dMatrix* DataSequence::getPrecomputedFeatures() const
{
	return precompFeatures;
}

void DataSequence::setPrecomputedFeaturesSparse(dMatrixSparse* m)
{
	if(precompFeaturesSparse != NULL)
	{
		delete precompFeaturesSparse;
		precompFeaturesSparse = NULL;
	}
	precompFeaturesSparse = m;
}

dMatrixSparse* DataSequence::getPrecomputedFeaturesSparse() const
{
	return precompFeaturesSparse;
}

void DataSequence::setStatesPerNode(iMatrix* spn)
// Takes ownership of spn
{
	if(statesPerNode != NULL)
	{
		delete statesPerNode;
		statesPerNode = NULL;
	}
	statesPerNode = spn;
}

iMatrix* DataSequence::getStatesPerNode() const
{
	return statesPerNode;
}

void DataSequence::setEstimatedStateLabels(iVector *v)
{
	if(estimatedStateLabels != NULL)
	{
		delete estimatedStateLabels;
		estimatedStateLabels = NULL;
	}
	estimatedStateLabels = v;
}

iVector* DataSequence::getEstimatedStateLabels() const
{
	return estimatedStateLabels;
}

void DataSequence::setEstimatedSequenceLabel(int seqLabel)
{
	estimatedSequenceLabel = seqLabel;
}

int DataSequence::getEstimatedSequenceLabel() const
{
	return estimatedSequenceLabel;
}

void DataSequence::setEstimatedProbabilitiesPerStates(dMatrix *m)
{
	if(estimatedProbabilitiesPerStates != NULL)
	{
		delete estimatedProbabilitiesPerStates;
		estimatedProbabilitiesPerStates = NULL;
	}
	estimatedProbabilitiesPerStates= m;
}

dMatrix* DataSequence::getEstimatedProbabilitiesPerStates() const
{
	return estimatedProbabilitiesPerStates;
}

void DataSequence::setWeightSequence(double w)
{
	weightSequence = w;
}

double DataSequence::getWeightSequence() const
{
	return weightSequence;
}




//-------------------------------------------------------------
// DataSet Class
//-------------------------------------------------------------

//*
// Constructors and Deconstructor
//*

DataSet::DataSet()
   : container(std::vector<DataSequence*>())
{
	//does nothing
}

DataSet::DataSet(const char *fileData, const char *fileStateLabels,
				 const char *fileSeqLabels, const char * fileAdjMat ,
				 const char * fileStatesPerNodes,const char * fileDataSparse)
   : container(std::vector<DataSequence*>())
{
	load(fileData, fileStateLabels, fileSeqLabels, fileAdjMat ,
		 fileStatesPerNodes,fileDataSparse);
}

DataSet::~DataSet()
{
	clearSequence();
}

//*
// Public Methods
//*

void DataSet::clearSequence()
{
	for(vector<DataSequence*>::iterator itSeq = container.begin();
		itSeq != container.end(); itSeq++)
	{
		delete (*itSeq);
		(*itSeq) = NULL;
	}
	container.clear();
}

int DataSet::load(const char *fileData, const char *fileStateLabels,
				  const char *fileSeqLabels, const char * fileAdjMat,
				  const char * fileStatesPerNodes,const char * fileDataSparse)
{
	istream* isData = NULL;
	istream* isDataSparse = NULL;
	istream* isStateLabels = NULL;
	istream* isSeqLabels = NULL;
	istream* isAdjMat = NULL;
	istream* isStatesPerNodes = NULL;

	if(fileData != NULL)
	{
		isData = new ifstream(fileData);
		if(!((ifstream*)isData)->is_open())
		{
			cerr << "Can't find data file: " << fileData << endl;
			delete isData;
			isData = NULL;
			throw BadFileName("Can't find data files");
		}
	}

	if(fileStateLabels != NULL)
	{
		isStateLabels = new ifstream(fileStateLabels);
		if(!((ifstream*)isStateLabels)->is_open())
		{
			cerr << "Can't find state labels file: " << fileStateLabels << endl;
			delete isStateLabels;
			isStateLabels = NULL;
			throw BadFileName("Can't find state labels file");
		}
	}
	if(fileSeqLabels != NULL)
	{
		isSeqLabels = new ifstream(fileSeqLabels);
		if(!((ifstream*)isSeqLabels)->is_open())
		{
			cerr << "Can't find sequence labels file: " << fileSeqLabels << endl;
			delete isSeqLabels;
			isSeqLabels = NULL;
		}
	}
	if(fileAdjMat != NULL)
	{
		isAdjMat = new ifstream(fileAdjMat);
		if(!((ifstream*)isAdjMat)->is_open())
		{
			cerr << "Can't find adjency matrices file: " << fileAdjMat << endl;
			delete isAdjMat;
			isAdjMat = NULL;
		}
	}
	if(fileStatesPerNodes != NULL)
	{
		isStatesPerNodes = new ifstream(fileStatesPerNodes);
		if(!((ifstream*)isStatesPerNodes)->is_open())
		{
			cerr << "Can't find states per nodes file: " << fileStatesPerNodes << endl;
			delete isStatesPerNodes;
			isStatesPerNodes = NULL;
		}
	}
	
	if(fileDataSparse != NULL)
	{
		isDataSparse = new ifstream(fileDataSparse);
		if(!((ifstream*)isDataSparse)->is_open())
		{
			cerr << "Can't find sparse data file: " << fileDataSparse << endl;
			delete isDataSparse;
			isDataSparse = NULL;
			throw BadFileName("Can't find sparse data files");
		}
	}

	DataSequence* seq = new DataSequence;
	int seqLabel;

	while(seq->load(isData,isStateLabels,isAdjMat,isStatesPerNodes,isDataSparse) == 0)
	{
		if(isSeqLabels)
		{
			*isSeqLabels >> seqLabel;
			seq->setSequenceLabel(seqLabel);
		}
		container.insert(container.end(),seq);
		seq = new DataSequence;				
	}
	delete seq;
	if(isData)
        delete isData;
	if(isStateLabels)
		delete isStateLabels;
	if(isSeqLabels)
		delete isSeqLabels;
	if(isAdjMat)
		delete isAdjMat;
	if(isStatesPerNodes)
		delete isStatesPerNodes;
	if(isDataSparse)
		delete isDataSparse;

	return 0;
}


int DataSet::searchNumberOfStates()
{
	int MaxLabel = -1;

	for(vector<DataSequence*>::iterator itSeq = container.begin(); 
		itSeq != container.end(); itSeq++)
	{
		if((*itSeq)->getStateLabels())
		{
			int seqMaxLabel = (*itSeq)->getStateLabels()->getMaxValue();
			if(seqMaxLabel > MaxLabel)
				MaxLabel = seqMaxLabel;
		}
	}
	return MaxLabel + 1;
}

int DataSet::searchNumberOfSequenceLabels()
{
	int MaxLabel = -1;

	for(vector<DataSequence*>::iterator itSeq = container.begin(); 
		itSeq != container.end(); itSeq++)
	{
		if((*itSeq)->getSequenceLabel() > MaxLabel)
			MaxLabel = (*itSeq)->getSequenceLabel();
	}
	return MaxLabel + 1;
}
int DataSet::getNumberofRawFeatures()
{
   if(size() > 0 && (*(container.begin()))->getPrecomputedFeatures() != NULL)
	  return (*(container.begin()))->getPrecomputedFeatures()->getHeight();
	else
		return 0;
}


std::ostream& operator <<(std::ostream& out, const DataSequence& seq)
{
	for(int i = 0; i < seq.length(); i++)
	{
		if(seq.getPrecomputedFeatures())
		{
			out << "f(:," << i << ") = [";
			for(int j=0; j < seq.getPrecomputedFeatures()->getHeight(); j++)
				out << (*seq.getPrecomputedFeatures())(j,i) << " ";
			out << endl;
		}
		if(seq.getStateLabels())
		{
			out << "y(" << i << ") = " << (*seq.getStateLabels())[i] << endl;
		}
	}
	return out;
}

std::ostream& operator <<(std::ostream& out, const DataSet& data)
{

   for(size_t indexSeq = 0; indexSeq < data.size(); indexSeq++)
	{
		out << "Sequence " << indexSeq << endl;
		out << *(data.at(indexSeq));
	}
	return out;
}

