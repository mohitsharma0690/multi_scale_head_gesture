#include "toolbox.h"

ToolboxHCRF::ToolboxHCRF():Toolbox()
{

}

ToolboxHCRF::ToolboxHCRF(int nbHiddenStates, int opt, int windowSize)
: Toolbox()
, numberOfHiddenStates(nbHiddenStates)
{
	init(nbHiddenStates,opt, windowSize);
}

ToolboxHCRF::~ToolboxHCRF()
{

}

void ToolboxHCRF::init(int nbHiddenStates, int opt, int windowSize)
{
	Toolbox::init(opt, windowSize);
	numberOfHiddenStates = nbHiddenStates;
	pFeatureGenerator->addFeature(new LabelEdgeFeatures());
	pGradient = new GradientHCRF (pInferenceEngine, pFeatureGenerator);
	pEvaluator = new EvaluatorHCRF (pInferenceEngine, pFeatureGenerator);

}

void ToolboxHCRF::initModel(DataSet &X)
{
	// Find number of states and initialize Model
	
	pModel->setNumberOfStates(numberOfHiddenStates);
	pModel->setNumberOfSequenceLabels(X.searchNumberOfSequenceLabels());

	// Initialize feature generator
	pFeatureGenerator->initFeatures(X,*pModel);
}

double ToolboxHCRF::test(DataSet& X, char* filenameOutput, char* filenameStats)
{
	double returnedF1value = 0.0;
	std::ofstream* fileOutput = NULL;
	if(filenameOutput)
	{
		fileOutput = new std::ofstream(filenameOutput);
		if (!fileOutput->is_open())
		{
			delete fileOutput;
			fileOutput = NULL;
		}
	}
	std::ostream* fileStats = NULL;
	if(filenameStats)
	{
		fileStats = new std::ofstream(filenameStats, std::ios_base::out | std::ios_base::app);
		if (!((std::ofstream*)fileStats)->is_open())
		{
			delete fileStats;
			fileStats = NULL;
		}
	}
	if(fileStats == NULL && pModel->getDebugLevel() >= 1)
		fileStats = &std::cout;


	DataSet::iterator it;
	int nbSeqLabels = pModel->getNumberOfSequenceLabels();
	iVector seqTruePos(nbSeqLabels);
	iVector seqTotalPos(nbSeqLabels);
	iVector seqTotalPosDetected(nbSeqLabels);

	for(it = X.begin(); it != X.end(); it++) 
	{
		//  Compute detected label
		dMatrix* matProbabilities = new dMatrix;
		int labelDetected = pEvaluator->computeSequenceLabel(*it,pModel,matProbabilities);
		(*it)->setEstimatedProbabilitiesPerStates(matProbabilities);
		(*it)->setEstimatedSequenceLabel(labelDetected);
		// Read ground truth label
		int label = (*it)->getSequenceLabel();

		// optionally writes results in file
		if( fileOutput)
			(*fileOutput) << labelDetected << std::endl;

		// Update total of positive detections
		seqTotalPos[label]++;
		seqTotalPosDetected[labelDetected]++;
		if( label == labelDetected)
			seqTruePos[label]++;
	}
	// Print results
	if(fileStats)
	{
		(*fileStats) << std::endl << "Calculations per sequences:" << std::endl;
		(*fileStats) << "Label\tTrue+\tMarked+\tDetect+\tPrec.\tRecall\tF1" << std::endl;
	}
	double prec,recall;
	int SumTruePos = 0, SumTotalPos = 0, SumTotalPosDetected = 0;
	for(int i=0 ; i<nbSeqLabels ; i++) 
	{
		SumTruePos += seqTruePos[i]; SumTotalPos += seqTotalPos[i]; SumTotalPosDetected += seqTotalPosDetected[i];
		prec=(seqTotalPos[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPos[i]))/1000;
		recall=(seqTotalPosDetected[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPosDetected[i]))/1000;
		if(fileStats)
			(*fileStats) << i << ":\t" << seqTruePos[i] << "\t" << seqTotalPos[i] << "\t" << seqTotalPosDetected[i] << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
	}
	prec=(SumTotalPos==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPos);
	recall=(SumTotalPosDetected==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPosDetected);
	if(fileStats)
	{
		(*fileStats) << "-----------------------------------------------------------------------" << std::endl;
		(*fileStats) << "Ov:\t" << SumTruePos << "\t" << SumTotalPos << "\t" << SumTotalPosDetected << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
	}
	returnedF1value = 2*prec*recall/(prec+recall);

	if( fileOutput )
	{
		fileOutput->close();
		delete fileOutput;
	}
	if(fileStats != &std::cout && fileStats != NULL)
	{
		((std::ofstream*)fileStats)->close();
		delete fileStats;
	}
	return returnedF1value;
}
