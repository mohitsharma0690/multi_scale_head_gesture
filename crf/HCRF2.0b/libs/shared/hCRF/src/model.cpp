//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Model Component
//
//	February 02, 2006

#include "model.h"
#include "featuregenerator.h"
#include <math.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <sstream>
using namespace std;

//-------------------------------------------------------------
// Model Class
//-------------------------------------------------------------

//*
// Constructor and Destructor
//*

Model::Model(int nos, int nosl, int nostatel):
weights(0)
{
	numberOfStates = 0;
	numberOfSequenceLabels = 0;
	numberOfStateLabels = 0;
	numberOfFeaturesPerLabel = 0;
	numberOfRawFeaturesPerFrame = 0;
	debugLevel = 0;

	regL1Sigma = 0.0;
	regL2Sigma = 0.0;
	regL1FeatureType = allTypes;
	regL2FeatureType = allTypes;

	if(nos > 0) {
		setNumberOfStates(nos);
	}
	if(nosl > 0) {
		setNumberOfSequenceLabels(nosl);
	}
	if(nostatel > 0) {
		setNumberOfStateLabels(nostatel);
	}

	setAdjacencyMatType(CHAIN);
	setStateMatType(ALLSTATES);
}

Model::~Model()
{
}

//*
// Public Methods
//*

void Model::setAdjacencyMatType(eGraphTypes atype, ...)
{
	va_list ap;
	va_start(ap,atype);
	if (atype==ADJMAT_PREDEFINED){
		char *filename = va_arg(ap,char*);
		if(!filename || loadAdjacencyMatrix(filename)) {
			throw BadFileName("Impossible to load the given adjacency matrix");
		}
	}
	adjMatType = atype;
	va_end(ap);
}

eGraphTypes Model::getAdjacencyMatType()
{
	return adjMatType;
}

int Model::setStateMatType(int stype, ...)
{
	/* This function is used to set the type of matrix used for
	the hidden state. If stype is STATEMAT_PREDFINED, then
	the second argument is expected to be a pointer to an iMatrix
	*/
	va_list ap;
	va_start(ap,stype);
	if(stype==STATEMAT_PREDEFINED)
	{
		iMatrix* userMatrix = va_arg(ap, iMatrix*);
		statesPerLabel = *userMatrix;
	}
	stateMatType = stype;
	va_end(ap);
	updateStatesPerLabel();
	return 0;
}

int Model::getStateMatType()
{
	return stateMatType;
}

void Model::getAdjacencyMatrix(uMatrix& adjMat, DataSequence* seq)
{
	int n = seq->length();
	seq->getAdjacencyMatrix(adjMat);
	if(adjMat.getWidth() == 0) {
		switch(adjMatType) {
			case CHAIN:
			case DANGLING_CHAIN:
				makeChain(adjMat, n);
				break;
			default:
				predefAdjMat(adjMat, n);
		};
	}
}

iMatrix* Model::getStateMatrix(DataSequence* seq)
{
	int n = seq->length();
	iMatrix * seqStateMat, * modelStateMat;
	seqStateMat = seq->getStatesPerNode();
	if(!seqStateMat) {
		switch(stateMatType) {
			case ALLSTATES:
				modelStateMat = makeFullStateMat(n);
				break;
			case STATES_BASED_ON_LABELS:
			case STATEMAT_PREDEFINED:
				// We generate based on the data sequence all the possible
				// states for the hiddent states
				modelStateMat = makeLabelsBasedStateMat(seq);
				break;
			default:
				throw std::runtime_error("Unkown state matrix type");
		}
		seqStateMat = modelStateMat;
	}

	return seqStateMat;
}

iVector* Model::getStateMatrix(DataSequence* seq, int nodeIndex)
{
	iMatrix * seqStateMat = getStateMatrix(seq);
	if(!seqStateMat) {
		return 0;
	}

	if(stateVec.getLength()==0) {
		stateVec.create(numberOfStates,COLVECTOR);
	}

	int row;
	for(row=0;row<seqStateMat->getHeight();row++) {
		stateVec[row] = seqStateMat->getValue(row,nodeIndex);
	}

	return &stateVec;
}

void Model::setWeights(const dVector& w)
{
	// set global parameter vector
	weights.set(w);
	refreshWeights();
}

void Model::refreshWeights()
{
	int i, j, total_ftrs, *p_fMask;
	double *p_yData, *p_wData;

	// set parameter vectors for each sequence label in the HCRF
	if(numberOfSequenceLabels>0) {

		// the feature mask must be set to perform this operation
#ifdef _DEBUG
		assert(featureMask.getWidth()!=0);
#endif

		// for each sequence label set its weight vector according
		// to the feature mask
		p_fMask = featureMask.get();
		total_ftrs = featureMask.getHeight();
		for(i = 0; i < numberOfSequenceLabels; i++)
		{
			if(weights_y[i].getLength()!=numberOfFeaturesPerLabel)
				weights_y[i].create(numberOfFeaturesPerLabel);
			p_wData = weights.get();
			p_yData = weights_y[i].get();
			for(j = 0; j < total_ftrs; j++, p_wData++, p_fMask++)
			{
				if(*p_fMask)
				{
					*p_yData = *p_wData;
					p_yData++;
				}
			}

			// make sure that featureMask was set correctly
#ifdef _DEBUG
			assert(p_yData==weights_y[i].get()+weights_y[i].getLength());
#endif
		}
	}
}

iMatrix* Model::getFeatureMask()
{
	return &featureMask;
}

uMatrix* Model::getInternalAdjencyMatrix()
{
	return &adjMat;
}

iMatrix* Model::getInternalStateMatrix()
{
	return &stateMat;
}


dVector* Model::getWeights(int seqLabel)
{
#ifdef _DEBUG
	assert(seqLabel<numberOfSequenceLabels);
#endif
	if(seqLabel==-1) {
		return &weights;
	} else {
		return &weights_y[seqLabel];
	}
}

int Model::getNumberOfStates() const
{
	return numberOfStates;
}

void Model::setNumberOfStates(int nos)
{
//	assert(nos>0);
	numberOfStates = nos;
	updateStatesPerLabel();
}

int Model::getNumberOfSequenceLabels() const
{
	return numberOfSequenceLabels;
}

void Model::setNumberOfSequenceLabels(int nosl)
{
//	assert(nosl>0);
	numberOfSequenceLabels = nosl;
	weights_y.resize(nosl);
}

int Model::getNumberOfStateLabels() const
{
	return numberOfStateLabels;
}

int Model::getNumberOfRawFeaturesPerFrame()
{
	return numberOfRawFeaturesPerFrame;
}

void Model::setNumberOfRawFeaturesPerFrame(int numberOfRawFeaturesPerFrame)
{
	this->numberOfRawFeaturesPerFrame = numberOfRawFeaturesPerFrame;
}


void Model::setNumberOfStateLabels(int nbStateLabels)
{
//	assert(nbStateLabels>0);
	numberOfStateLabels = nbStateLabels;
	updateStatesPerLabel();
}

int Model::getDebugLevel()
{
	return debugLevel;
}

void Model::setDebugLevel(int newDebugLevel)
{
	debugLevel = newDebugLevel;
}


void Model::setFeatureMask(iMatrix &ftrMask)
{
	int i, *p_fMask;

#ifdef _DEBUG
	assert(ftrMask.getWidth()==numberOfSequenceLabels);
#endif

	featureMask = ftrMask;
	p_fMask = featureMask.get();
	numberOfFeaturesPerLabel = 0;
	for(i = 0; i < featureMask.getHeight(); i++, p_fMask++) {
		if(*p_fMask) {
			numberOfFeaturesPerLabel++;
		}
	}
}

int Model::getNumberOfFeaturesPerLabel()
{
	return numberOfFeaturesPerLabel;
}

void Model::load(const char* pFilename)
{
	ifstream fileInput(pFilename);
	if (!fileInput.is_open())
	{
		std::stringstream error("Can't find model definition file: ");
		error<< pFilename;
		throw BadFileName(error.str());
		return;
	}

	read(&fileInput);

	fileInput.close();
}

void Model::save(const char* pFilename) const
{
	ofstream fileOutput(pFilename);
	if (!fileOutput.is_open())
	{
		std::stringstream error("Can't open model definition for saving: ");
		error<< pFilename;
		throw BadFileName(error.str());
		return;
	}
	write(&fileOutput);
	
	fileOutput.close();
}

int Model::read(std::istream *stream)
{

	(*stream) >> numberOfStates;
	if(stream->fail()) return 1;
	this->setNumberOfStates(numberOfStates);

	(*stream) >> numberOfRawFeaturesPerFrame;

	(*stream) >> numberOfSequenceLabels;
	if(stream->fail()) return 1;
	this->setNumberOfSequenceLabels(numberOfSequenceLabels);

	(*stream) >> numberOfStateLabels;
	if(stream->fail()) return 1;
	this->setNumberOfStateLabels(numberOfStateLabels);

	(*stream) >> numberOfFeaturesPerLabel;
	if(stream->fail()) return 1;
	
	int temp;
	(*stream) >> temp;
	adjMatType = (eGraphTypes) temp;
	if(stream->fail()) return 1;

	(*stream) >> stateMatType;
	if(stream->fail()) return 1;

	(*stream) >> weights;
	if(stream->fail()) return 1;

//	(*stream) >> adjMat;
//	if(stream->fail()) return 1;

//	(*stream) >> stateMat;
//	if(stream->fail()) return 1;

	(*stream) >> featureMask;
	if(stream->fail()) return 1;

//	(*stream) >> statesPerLabel;
//	if(stream->fail()) return 1;

//	(*stream) >> labelPerState;
//	if(stream->fail()) return 1;

	updateStatesPerLabel();
	refreshWeights();

	return 0;
}

int Model::write(std::ostream *stream) const
{
	(*stream) << numberOfStates << "\n";
	if(stream->fail()) return 1;

	(*stream) << numberOfRawFeaturesPerFrame << "\n";  
	if(stream->fail()) return 1;
	
	(*stream) << numberOfSequenceLabels << "\n";
	if(stream->fail()) return 1;

	(*stream) << numberOfStateLabels << "\n";
	if(stream->fail()) return 1;
	
	

	(*stream) << numberOfFeaturesPerLabel << "\n";
	if(stream->fail()) return 1;

	(*stream) << adjMatType << "\n";
	if(stream->fail()) return 1;

	(*stream) << stateMatType << "\n";
	if(stream->fail()) return 1;

	if(weights.write(stream)) return 1;

	//if(adjMat.write(stream)) return 1;

	//if(stateMat.write(stream)) return 1;

	if(featureMask.write(stream)) return 1;

	//if(statesPerLabel.write(stream)) return 1;

	//if(labelPerState.write(stream)) return 1;
	
	

	return 0;
}

iMatrix& Model::getStatesPerLabel()
{
	return statesPerLabel;
}

iVector& Model::getLabelPerState()
{
	return labelPerState;
}


//*
// Private Methods
//*

int Model::loadAdjacencyMatrix(const char *pFilename)
{
	if(!pFilename)
		throw BadFileName("Null pointer");
	ifstream infile;
	infile.open(pFilename, ifstream::in);
	if(adjMat.read(&infile)) {
		return 1;
	}
	infile.close();
	return 0;
}

int Model::loadStateMatrix(const char *pFilename)
{
	if(!pFilename) throw BadFileName("Null pointer");
	ifstream infile;
	infile.open(pFilename, ifstream::in);
	if(stateMat.read(&infile)) {
		return 1;
	}
	infile.close();
	return 0;
}

void Model::makeChain(uMatrix& outMat, int n)
{
	int prevN = outMat.getHeight();
	int row, col;
	// We assume that adjMat is already a chain. This can be dangerous
	if(prevN<n) {
		outMat.resize(n,n);
		if(prevN>0)
			prevN--;
		for(col=prevN,row=prevN; col<n; col++,row++) {
			if(col>0) {
				outMat(col-1,row) = 1;
			}
			outMat(col,row) = 0;
			if(col<(n-1)) {
				outMat(col+1,row) = 1;
			}
		}
	}
}

void Model::predefAdjMat(uMatrix& outMat, int n)
{
	if(adjMat.getHeight()<n) {
		outMat.resize(0,0);
	}
	outMat = adjMat;
}

iMatrix * Model::makeLabelsBasedStateMat(DataSequence* seq)
{
	if(!seq->getStatesPerNode())
	{
#ifdef _DEBUG
		assert(statesPerLabel.getWidth() > 0);
		assert(seq->getStateLabels());
#endif

		iMatrix* tmpStateMat = new iMatrix(seq->getStateLabels()->getLength(),numberOfStates,0);
		for(int i = 0 ; i < seq->getStateLabels()->getLength(); i++)
			for(int j = 0 ; j < numberOfStates; j++)
				tmpStateMat->setValue(j,i, statesPerLabel.getValue(j,seq->getStateLabels()->getValue(i)));

		seq->setStatesPerNode( tmpStateMat );
	}
	return seq->getStatesPerNode();
}

iMatrix * Model::makeFullStateMat(int n)
{
	if(numberOfStates==0) {
		return 0;
	}
	stateMat.create(n,numberOfStates,1);
	return &stateMat;
}

iMatrix * Model::predefStateMat(int n)
{
	if(stateMat.getHeight()<n || numberOfStates==0) {
		return 0;
	}

	return &stateMat;
}

void Model::updateStatesPerLabel()
{
	// We update statesPerLabel only if the stateMatType is SATES_BASED_ON_LABELS
	if(numberOfStateLabels > 0 && numberOfStates > 0 && stateMatType == STATES_BASED_ON_LABELS)
	{
		statesPerLabel.create(numberOfStateLabels,numberOfStates);
		labelPerState.create(numberOfStates);
		int numberOfStatesPerLabels = (int)((float)numberOfStates / (float)numberOfStateLabels + 0.5f);
		int l = 0;
		int li = 0;
		for( int s = 0; s < numberOfStates; s++)
		{
			statesPerLabel(s,l) = 1;
			labelPerState[s] = l;
			li++;
			if(li >= numberOfStatesPerLabels && l < (numberOfStateLabels - 1))
			{
				li = 0;
				l ++;
			}
		}
	}
}

double Model::getRegL1Sigma()
{
	return regL1Sigma;
}

double Model::getRegL2Sigma()
{
	return regL2Sigma;
}

void Model::setRegL1Sigma(double sigma, eFeatureTypes typeFeature)
{
	regL1Sigma = sigma;
	regL1FeatureType = typeFeature;
}

void Model::setRegL2Sigma(double sigma, eFeatureTypes typeFeature)
{
	regL2Sigma = sigma;
	regL2FeatureType = typeFeature;
}

eFeatureTypes Model::getRegL1FeatureTypes()
{
	return regL1FeatureType;
}

eFeatureTypes Model::getRegL2FeatureTypes()
{
	return regL2FeatureType;
}



//-------------------------------------------------------------
// stream io routines
//-------------------------------------------------------------

std::istream& operator >>(std::istream& in, Model& m)
{
	m.read(&in);
	return in;
}

std::ostream& operator <<(std::ostream& out, const Model& m)
{
	m.write(&out);
	return out;
}
