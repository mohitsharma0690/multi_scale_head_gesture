#include <string>
#include <map>
#include "mex.h"
#include "hCRF.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
typedef void (*matlab_handler)(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[]);

typedef map<string, matlab_handler> DISPATCHTAB;
static DISPATCHTAB *func_lookup = 0;


// Static variables
static Toolbox* toolbox = NULL;
static DataSet* data = NULL;
// We are by default not in object mode (pass an array of pointer as 
// first argument to init toolbox and data). The first call put the
// the toolbox either in default mode or object mode. After that, the
// same mode must always be used.
static bool objectMode = false;

template <class elType>
void copyMatToCpp(Matrix<elType>* matData, mxArray* mxData)
/*
This function copy to data contained in the mxArray in the 
Matrix data.
*/
{
	matData->create((int)mxGetN(mxData),(int)mxGetM(mxData));
	memcpy(matData->get(), (void*)mxGetData(mxData), (int)mxGetNumberOfElements(mxData) * sizeof(elType));
}

template <class elType>
void copyMatToCpp(MatrixSparse<elType>* matDataSparse, mxArray* mxData)
/*
This function copy to data contained in the mxArray in the 
Matrix data.
*/
{
	matDataSparse->createJc((int)mxGetN(mxData));
	memcpy(matDataSparse->getJc()->get(), (void*)mxGetJc(mxData), ((int)mxGetN(mxData)+1) * sizeof(size_t));
	
	matDataSparse->setHeight((int)mxGetM(mxData));
	
	if(matDataSparse->getNumOfElements()>0)
	{
		matDataSparse->createPrIr(matDataSparse->getNumOfElements());
		memcpy(matDataSparse->getIr()->get(), (void*)mxGetIr(mxData), matDataSparse->getNumOfElements() * sizeof(size_t));
		memcpy(matDataSparse->getPr()->get(), (void*)mxGetPr(mxData), matDataSparse->getNumOfElements() * sizeof(elType));
	}
}




template <class elType>
void copyMatToCpp(Vector<elType>* matData, mxArray* mxData)
/*
This function copy to data contained in the mxArray in the 
Vector data.
*/
{
	matData->create((int)mxGetNumberOfElements(mxData));
	memcpy(matData->get(), (void*)mxGetData(mxData), (int)mxGetNumberOfElements(mxData) * sizeof(elType));
}

static void createToolbox(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(nrhs<1)
		mexErrMsgTxt("hCRF: Must specify model type (crf, ghcrf, hcrf, ldcrf or sharedLdcrf).\n");

	if(toolbox != NULL)
	{
		delete toolbox;
		toolbox = NULL;
	}

	char str[100];
	int r = mxGetString(prhs[0], str, sizeof(str));
	if(r!=0)
		mexErrMsgTxt("hCRF: Model type must be a string.\n");

	int nbHiddenStates = 3;
	int windowSize = 0;
	int opt = OPTIMIZER_BFGS;

	if(nrhs>1)
	{
		char strOptimizer[100];
		r = mxGetString(prhs[1], strOptimizer, sizeof(strOptimizer));
		if(r == 0)
		{
			if(!strcmp(strOptimizer,"cg"))
				opt = OPTIMIZER_CG;
			else if(!strcmp(strOptimizer,"bfgs"))
				opt = OPTIMIZER_BFGS;
#ifndef _PUBLIC
			else if(!strcmp(strOptimizer,"asa"))
				opt = OPTIMIZER_ASA;
			else if(!strcmp(strOptimizer,"owlqn"))
				opt = OPTIMIZER_OWLQN;
#endif
			else if(!strcmp(strOptimizer,"lbfgs"))
				opt = OPTIMIZER_LBFGS;
			else
				mexErrMsgTxt("hCRF: Invalid optimiser string.\n");
		}
	}
	if(nrhs>2)
		if(!mxIsEmpty(prhs[2]))
			nbHiddenStates = (int)*((double*)mxGetData(prhs[2]));

	if(nrhs>3)
		if(!mxIsEmpty(prhs[3]))
			windowSize = (int)*((double*)mxGetData(prhs[3]));


	if(!strcmp(str,"crf"))
		toolbox = new ToolboxCRF(opt, windowSize);
	else if(!strcmp(str,"hcrf"))
		toolbox = new ToolboxHCRF(nbHiddenStates, opt, windowSize);
	else if(!strcmp(str,"ghcrf"))
		toolbox = new ToolboxGHCRF(nbHiddenStates, opt, windowSize);
	else if((!strcmp(str,"ldcrf"))||(!strcmp(str,"fhcrf")))
		toolbox = new ToolboxLDCRF(nbHiddenStates, opt, windowSize);
	#ifndef _PUBLIC
	else if(!strcmp(str,"sldcrf")||!strcmp(str,"sharedLdcrf"))
		toolbox = new ToolboxSharedLDCRF(nbHiddenStates, opt, windowSize);
	#endif
	else
		mexErrMsgTxt("hCRF: Invalid model type. Should be 'crf', 'hcrf', 'ldcrf', 'sharedLdcrf'.\n");
}


static void loadData(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{	
	if(nrhs<1)
		mexErrMsgTxt("hCRF: Must specify filenames for dataset.\n");

	char strData[256];
	char strLabels[256];
	char strSeqLabels[256];
	char* pData = strData;
	char* pLabels = NULL;
	char* pSeqLabels = NULL;

	int r = mxGetString(prhs[0], strData, sizeof(strData));
	if(r!=0) 
	{
		mexErrMsgTxt("hCRF: Features filename must be a string.\n");
		pData = 0;
	}
	if(nrhs > 1)
	{
		pLabels = strLabels;
		r = mxGetString(prhs[1], strLabels, sizeof(strLabels));
		if(r!=0) 
		{
			mexErrMsgTxt("hCRF: Labels filename must be a string.\n");
			pLabels = 0;
		}
	}
	if(nrhs > 2)
	{
		pSeqLabels = strSeqLabels;
		r = mxGetString(prhs[2], strSeqLabels, sizeof(strSeqLabels));
		if(r!=0) 
		{
			mexErrMsgTxt("hCRF: Sequence labels filename must be a string.\n");
			pSeqLabels = 0;
		}
	}

	if(data != NULL)
	{
		delete data;
		data = NULL;
	}

//	mexEvalString("fprintf('test2')");
//	cerr << "test" << endl;
	data = new DataSet(pData, pLabels, pSeqLabels);
}


static void setData(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	bool bValidData1 = false;	
	bool bValidData2= false;  
	bool bValidLabels = false;
	bool bValidSeqLabel = false;
	bool bValidSeqWeights = false;	 
	int nbSeq = 0;

	if(nrhs<1)
		mexErrMsgTxt("hCRF: Raw features missing.\n");
	
	if(!mxIsEmpty(prhs[0]))
	{
		if(!mxIsCell(prhs[0]))
			mexErrMsgTxt("hCRF: First argument must be a cell array representing raw(either dense or sparse) features.\n");
		else
		{			
			nbSeq = (int)mxGetNumberOfElements(prhs[0]);
			if(nbSeq < 1 )
				mexErrMsgTxt("hCRF: First argument must be a cell array representing raw features.\n");
			for (int i = 0; i < nbSeq; i++)
			{					
				if(mxGetCell(prhs[0],i) == 0 || mxIsEmpty(mxGetCell(prhs[0],i)))
					mexErrMsgTxt("hCRF: First argument (feature sequences) contains at least one empty sequence.\n");
			}
			bValidData1 = true;
		}
	}
	
	if(nbSeq < 1)
	mexErrMsgTxt("hCRF: Raw features missing, should be passes in the first Argument\n");

	if(nrhs > 4) 
	{
		if(!mxIsEmpty(prhs[4]))
		{
			if(!mxIsCell(prhs[4]))
				mexErrMsgTxt("hCRF: If the Fifth argument is passed, it must be a cell array representing sparse raw features.\n And the first Argument should be a cell array representing dense raw features.");
			else
			{		
				if(nbSeq > 0){
					if((int)mxGetNumberOfElements(prhs[4]) != nbSeq )		
						mexErrMsgTxt("hCRF: The number of sequences in the Fifth Argument should be the same as in the first Argument.\n");													
				}
				for (int i = 0; i < nbSeq; i++)
					if(mxIsEmpty(mxGetCell(prhs[4],i)))
						mexErrMsgTxt("hCRF: Fifth argument (sparse feature sequences) contains at least one empty sequence.\n");
				bValidData2= true;
			}												
		}
	}
	
	if(nrhs>1)
	{
		if(mxIsCell(prhs[1])) // Label
		{
			if((int)mxGetNumberOfElements(prhs[1]) != nbSeq)
				mexErrMsgTxt("hCRF: The number of state-label sequences should be the same as the number of raw-feature sequences.\n");
			if((int)mxGetNumberOfElements(prhs[1]) > 0)
			{
				if(mxIsInt32(mxGetCell(prhs[1],0)))
				{
					for (int i = 0; i < nbSeq; i++)
						if(mxIsEmpty(mxGetCell(prhs[1],i)))
							mexErrMsgTxt("hCRF: Second argument (label sequences) contains at least one empty sequence.\n");
					bValidLabels = true;
				}
				else
					mexErrMsgTxt("hCRF: State labels should be of type Int32.\n");
			}
		}
	}
	if(nrhs>2)
	{
		if(!mxIsEmpty(prhs[2])) // Sequence Labels
		{
			if(mxIsInt32(prhs[2]))
			{
				if((int)mxGetNumberOfElements(prhs[2]) != nbSeq)
					mexErrMsgTxt("hCRF: The number of sequence-label sequences should be the same as the number of raw-feature sequences.\n");
				bValidSeqLabel = true;
			}
			else
				mexErrMsgTxt("hCRF: Sequence labels should be of type Int32.\n");
		}
	}	

	if(nrhs>3)
	{
		if(mxIsDouble(prhs[3])) // Sequence Weights
		{
			if((int)mxGetNumberOfElements(prhs[3]) != nbSeq)
				mexErrMsgTxt("hCRF: The number of sequence-weight sequences should be the same as the number of raw-feature sequences.\n");
			bValidSeqWeights = true;
		}
		else
			mexErrMsgTxt("hCRF: Sequence weights should be Double.\n");
	}	

	if(nbSeq > 0 && (bValidData1 || bValidData2|| bValidLabels || bValidSeqLabel || bValidSeqWeights))
	{
		if(data != NULL)
		{
			delete data;
			data = NULL;
		}
		data = new DataSet;

		for (int i = 0; i < nbSeq; i++)
		{
			DataSequence* seq = new DataSequence;
			if(bValidData1)
			{	// The first Argument can either be sparse or dense features
				if(mxIsSparse(mxGetCell(prhs[0],i)))
				{
					dMatrixSparse* matDataSparse = new dMatrixSparse();
					copyMatToCpp(matDataSparse, mxGetCell(prhs[0],i));
					seq->setPrecomputedFeaturesSparse(matDataSparse);
				}
				else
				{
					dMatrix* matData = new dMatrix();
					copyMatToCpp(matData, mxGetCell(prhs[0],i));
					seq->setPrecomputedFeatures(matData);
				}
			}
			if(bValidData2)
			{
				if(mxIsSparse(mxGetCell(prhs[0],i)))
				{
					dMatrixSparse* matDataSparse = new dMatrixSparse();
					copyMatToCpp(matDataSparse, mxGetCell(prhs[4],i));
					seq->setPrecomputedFeaturesSparse(matDataSparse);
				}
				else
				{
					mexErrMsgTxt("hCRF: If the Fifth argument is passed, it must be a cell array representing sparse raw features.\n And the first Argument should be a cell array representing dense raw features.");
				}
			}
			if(bValidLabels)
			{
				iVector* vecLabels = new iVector();
				copyMatToCpp(vecLabels, mxGetCell(prhs[1],i));
				seq->setStateLabels(vecLabels);
			}
			if(bValidSeqLabel)
			{
				int seqLabel = ((int*)mxGetData(prhs[2]))[i];
				seq->setSequenceLabel(seqLabel);
			}
			if(bValidSeqWeights)
			{
				double weightSeq = ((double*)mxGetData(prhs[3]))[i];
				seq->setWeightSequence(weightSeq);
			}
			data->insert(data->end(),seq);
		}
	}
}

static void train(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");
	if(data == NULL)
		mexErrMsgTxt("hCRF: No dataset was previously loaded. Use function 'loadData' to load a dataset.\n");

	toolbox->train(*data,true);
}

static void test(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");
	if(data == NULL)
		mexErrMsgTxt("hCRF: No dataset was previously loaded. Use function 'loadData' to load a dataset.\n");

	char strOutput[256];
	char strStats[256];
	char* pOutput = NULL;
	char* pStats = NULL;

	if(nrhs > 0)
	{
		pOutput = strOutput;
		int r = mxGetString(prhs[0], strOutput, sizeof(strOutput));
		if(r!=0) 
		{
			mexErrMsgTxt("hCRF: Ouput filename must be a string.\n");
			pOutput = 0;
		}
	}
	if(nrhs > 1)
	{
		pStats = strStats;
		int r = mxGetString(prhs[1], strStats, sizeof(strStats));
		if(r!=0) 
		{
			mexErrMsgTxt("hCRF: Stats filename must be a string.\n");
			pStats = 0;
		}
	}

	toolbox->test(*data,pOutput, pStats);
}

static void get(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");

	if(nrhs < 1)
		mexErrMsgTxt("hCRF: One parameter required: name of the parameter to read.\n");

	char strNameParameter[256];
	int r = mxGetString(prhs[0], strNameParameter, sizeof(strNameParameter));
	if(r!=0) 
		mexErrMsgTxt("hCRF: Name of the parameter must be a string.\n");

	if(!strcmp(strNameParameter,"maxIterations"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getMaxNbIteration());
	}
	else if(!strcmp(strNameParameter,"debugLevel"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getDebugLevel());
	}
	else if(!strcmp(strNameParameter,"regularization") || !strcmp(strNameParameter,"regularizationL2"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getRegularizationL2());
	}
	else if(!strcmp(strNameParameter,"regularizationL1"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getRegularizationL1());
	}
	else if(!strcmp(strNameParameter,"statsNbIterations"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getOptimizer()->getLastNbIterations());
	}
	else if(!strcmp(strNameParameter,"statsFunctionError"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getOptimizer()->getLastFunctionError());
	}
	else if(!strcmp(strNameParameter,"statsNormGradient"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getOptimizer()->getLastNormGradient());
	}
	else if(!strcmp(strNameParameter,"minRangeWeights"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getMinRangeWeights());
	}
	else if(!strcmp(strNameParameter,"maxRangeWeights"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getMaxRangeWeights());
	}
	else if(!strcmp(strNameParameter,"randomSeed"))
	{
		plhs[0] = mxCreateScalarDouble(toolbox->getRandomSeed());
	}

	else if(!strcmp(strNameParameter,"weightsInitType"))
	{
		switch(toolbox->getWeightInitType())
		{
		case INIT_ZERO:
			plhs[0] = mxCreateString("ZERO");
			break;
		case INIT_CONSTANT:
			plhs[0] = mxCreateString("CONSTANT");
			break;
		case INIT_RANDOM:
			plhs[0] = mxCreateString("RANDOM");
			break;
		case INIT_MEAN:
			plhs[0] = mxCreateString("MEAN");
			break;
		case INIT_RANDOM_MEAN_STDDEV:
			plhs[0] = mxCreateString("RANDOM_MEAN_STDDEV");
			break;
		case INIT_RANDOM_GAUSSIAN:
			plhs[0] = mxCreateString("RANDOM_GAUSSIAN");
			break;
		case INIT_RANDOM_GAUSSIAN2:
			plhs[0] = mxCreateString("RANDOM_GAUSSIAN2");
			break;
		case INIT_GAUSSIAN:
			plhs[0] = mxCreateString("GAUSSIAN");
			break;
		case INIT_PREDEFINED:
			plhs[0] = mxCreateString("PREDEFINED");
			break;
		}
	}
	else
		mexErrMsgTxt("hCRF: Unknown parameter.\n");
}

static void set(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");

	if(nrhs < 2)
		mexErrMsgTxt("hCRF: Two parameters required: name of the parameter and value to assign.\n");

	char strNameParameter[256];
	int r = mxGetString(prhs[0], strNameParameter, sizeof(strNameParameter));
	if(r!=0) 
		mexErrMsgTxt("hCRF: Name of the parameter must be a string.\n");

	if(!strcmp(strNameParameter,"maxIterations"))
	{
		if(!mxIsEmpty(prhs[1]))
			toolbox->setMaxNbIteration((int)*((double*)mxGetData(prhs[1])));
	}
	else if(!strcmp(strNameParameter,"debugLevel"))
	{
		if(!mxIsEmpty(prhs[1]))
			toolbox->setDebugLevel((int)*((double*)mxGetData(prhs[1])));
	}
	else if(!strcmp(strNameParameter,"nbThreads"))
	{
#ifdef _OPENMP
		if(!mxIsEmpty(prhs[1]))
			omp_set_num_threads((int)*((double*)mxGetData(prhs[1])));
#endif
	}
	else if(!strcmp(strNameParameter,"regularization") || !strcmp(strNameParameter,"regularizationL2")  || !strcmp(strNameParameter,"regL2Sigma"))
	{
		if(!mxIsEmpty(prhs[1]))
			toolbox->setRegularizationL2(*((double*)mxGetData(prhs[1])));
	}
	else if(!strcmp(strNameParameter,"regularizationL1") || !strcmp(strNameParameter,"regL1Sigma"))
	{
		if(!mxIsEmpty(prhs[1]))
			toolbox->setRegularizationL1(*((double*)mxGetData(prhs[1])));
	}
	else if(!strcmp(strNameParameter,"regL1FeatureTypes"))
	{
		char strType[100];
		r = mxGetString(prhs[1], strType, sizeof(strType));
		if(r == 0)
		{
			eFeatureTypes featureType = allTypes;
			if(!strcmp(strType,"ALL"))
				featureType = allTypes;
			else if(!strcmp(strType,"NODE"))
				featureType = nodeFeaturesOnly;
			else if(!strcmp(strType,"EDGE"))
				featureType = edgeFeaturesOnly;
			else
				return;
			toolbox->setRegularizationL1(toolbox->getRegularizationL1(),featureType);
		}
	}
	else if(!strcmp(strNameParameter,"regL2FeatureTypes"))
	{
		char strType[100];
		r = mxGetString(prhs[1], strType, sizeof(strType));
		if(r == 0)
		{
			eFeatureTypes featureType = allTypes;
			if(!strcmp(strType,"ALL"))
				featureType = allTypes;
			else if(!strcmp(strType,"NODE"))
				featureType = nodeFeaturesOnly;
			else if(!strcmp(strType,"EDGE"))
				featureType = edgeFeaturesOnly;
			else
				return;
			toolbox->setRegularizationL2(toolbox->getRegularizationL2(),featureType);
		}
	}
	else if(!strcmp(strNameParameter,"minRangeWeights"))
	{
		if(!mxIsEmpty(prhs[1]))
			toolbox->setMinRangeWeights(*((double*)mxGetData(prhs[1])));
	}
	else if(!strcmp(strNameParameter,"maxRangeWeights"))
	{
		if(!mxIsEmpty(prhs[1]))
			toolbox->setMaxRangeWeights(*((double*)mxGetData(prhs[1])));
	}
	else if(!strcmp(strNameParameter,"randomSeed"))
	{
		if(!mxIsEmpty(prhs[1]))
			toolbox->setRandomSeed(*((double*)mxGetData(prhs[1])));
	}
	else if(!strcmp(strNameParameter,"initWeights"))
	{
		if(!mxIsEmpty(prhs[1])&& mxIsDouble(prhs[1]))
		{
			dVector vecWeights ((int)(mxGetM(prhs[1])*mxGetN(prhs[1])));
			memcpy(vecWeights.get(),mxGetPr(prhs[1]),mxGetM(prhs[1])*mxGetN(prhs[1])*sizeof(double));
			toolbox->setInitWeights(vecWeights);
		}
	}
	else if(!strcmp(strNameParameter,"weightsInitType"))
	{
		char strType[100];
		r = mxGetString(prhs[1], strType, sizeof(strType));
		if(r == 0)
		{
			int initType = INIT_RANDOM;
			if(!strcmp(strType,"ZERO"))
				initType = INIT_ZERO;
			else if(!strcmp(strType,"CONSTANT"))
				initType = INIT_CONSTANT;
			else if(!strcmp(strType,"RANDOM"))
				initType = INIT_RANDOM;
			else if(!strcmp(strType,"MEAN"))
				initType = INIT_MEAN;
			else if(!strcmp(strType,"RANDOM_MEAN_STDDEV"))
				initType = INIT_RANDOM_MEAN_STDDEV;
			else if(!strcmp(strType,"GAUSSIAN"))
				initType = INIT_GAUSSIAN;
			else if(!strcmp(strType,"PREDEFINED"))
				initType = INIT_PREDEFINED;
			else if(!strcmp(strType, "RANDOM_GAUSSIAN"))
				initType = INIT_RANDOM_GAUSSIAN;
			else if(!strcmp(strType, "RANDOM_GAUSSIAN2"))
				initType = INIT_RANDOM_GAUSSIAN2;
			else
				return;
			toolbox->setWeightInitType(initType);

		}
	}
	else
		mexErrMsgTxt("hCRF: Unknown parameter.\n");

}

static void getAllFeatures(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");
	if(data == NULL)
		mexErrMsgTxt("hCRF: No dataset was previously loaded. Use function 'loadData' to load a dataset.\n");
	if(nlhs > 0 && data->size() > 0)
	{
		featureVector* vecFeatures = toolbox->getAllFeatures(*data);
		feature* pFeature = vecFeatures->getPtr();
		plhs[0] = mxCreateNumericMatrix(vecFeatures->size(), 1, mxINT32_CLASS,mxREAL);
		if(nlhs > 1)
			plhs[1] = mxCreateNumericMatrix(vecFeatures->size(), 1, mxINT32_CLASS,mxREAL);
		if(nlhs > 2)
			plhs[2] = mxCreateNumericMatrix(vecFeatures->size(), 1, mxINT32_CLASS,mxREAL);
		if(nlhs > 3)
			plhs[3] = mxCreateNumericMatrix(vecFeatures->size(), 1, mxINT32_CLASS,mxREAL);
		if(nlhs > 4)
			plhs[4] = mxCreateNumericMatrix(vecFeatures->size(), 1, mxINT32_CLASS,mxREAL);
		for(int i = 0; i < vecFeatures->size(); pFeature++, i++)
		{
			((int*)mxGetPr(plhs[0]))[i] = (int)pFeature->value;
			if(nlhs > 1)
				((int*)mxGetPr(plhs[1]))[i] = (int)pFeature->nodeState;
			if(nlhs > 2)
				((int*)mxGetPr(plhs[2]))[i] = (int)pFeature->prevNodeState;
			if(nlhs > 3)
				((int*)mxGetPr(plhs[3]))[i] = (int)pFeature->sequenceLabel;
			if(nlhs > 4)
				((int*)mxGetPr(plhs[4]))[i] = (int)pFeature->nodeIndex;
		}
	}
}


static void getResults(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(data == NULL)
		mexErrMsgTxt("hCRF: No dataset was previously loaded. Use function 'loadData' to load a dataset.\n");
	if (data->size() == 0)
		mexErrMsgTxt("hCRF: Empty data set\n");
	if((*data->begin())->getEstimatedProbabilitiesPerStates() == NULL)
		mexErrMsgTxt("hCRF: No results available. You must run 'test' function before to be able to access results.\n");

	if(nlhs > 0)
	{
		// We return a cell array containing for each observation, each time and each labels 
		// an estimated probability.

		plhs[0] = mxCreateCellMatrix(1,(int)data->size());
		int seqIndex = 0;
		for(vector<DataSequence*>::iterator itSeq = data->begin(); itSeq != data->end(); itSeq++)
		{
			int m = (*itSeq)->getEstimatedProbabilitiesPerStates()->getHeight();
			int n = (*itSeq)->getEstimatedProbabilitiesPerStates()->getWidth();
			mxArray* newArray = mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS,mxREAL); 
			memcpy(mxGetPr(newArray),(*itSeq)->getEstimatedProbabilitiesPerStates()->get(),m*n*sizeof(double));
			mxSetCell(plhs[0],seqIndex,newArray);
			seqIndex++;
		}
	}
	if(nlhs > 1)
	{
		// We return the sequence label for each sequence
		plhs[1] = mxCreateNumericMatrix(1,(int)data->size(), mxDOUBLE_CLASS, mxREAL);
		double * pData = mxGetPr(plhs[1]);
		for(vector<DataSequence*>::iterator itSeq = data->begin(); itSeq != data->end(); itSeq++)
		{
			(*pData) = (*itSeq)->getEstimatedSequenceLabel();
			pData ++;
		}
	}
	if(nlhs > 2)
	{
		// We can also get the estimated label per state
		plhs[2] = mxCreateCellMatrix(1,(int)data->size());
		int seqIndex = 0;
		for(vector<DataSequence*>::iterator itSeq = data->begin(); itSeq != data->end(); itSeq++)
		{
			int m = (*itSeq)->getEstimatedStateLabels()->getHeight();
			int n = (*itSeq)->getEstimatedStateLabels()->getWidth();
			mxArray* newArray = mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS,mxREAL); 
			memcpy(mxGetPr(newArray),(*itSeq)->getEstimatedStateLabels()->get(),m*n*sizeof(double));
			mxSetCell(plhs[0],seqIndex,newArray);
			seqIndex++;
		}
	}
}


static void loadModel(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");

	if(nrhs < 2)
		mexErrMsgTxt("hCRF: Model and features filenames required.\n");

	char strFeatures[256];
	char strModel[256];

	int r = mxGetString(prhs[0], strModel, sizeof(strModel));
	if(r!=0) 
		mexErrMsgTxt("hCRF: Model filename must be a string.\n");

	r = mxGetString(prhs[1], strFeatures, sizeof(strFeatures));
	if(r!=0) 
			mexErrMsgTxt("hCRF: Feature filename must be a string.\n");

	toolbox->load(strModel, strFeatures);
}

static void saveModel(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");

	if(nrhs < 2)
		mexErrMsgTxt("hCRF: Model and features filenames required.\n");

	char strFeatures[256];
	char strModel[256];

	int r = mxGetString(prhs[0], strModel, sizeof(strModel));
	if(r!=0) 
		mexErrMsgTxt("hCRF: Model filename must be a string.\n");

	r = mxGetString(prhs[1], strFeatures, sizeof(strFeatures));
	if(r!=0) 
		mexErrMsgTxt("hCRF: Feature filename must be a string.\n");

	toolbox->save(strModel, strFeatures);
}

static void setModel(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");

	if(nrhs != 2)
		mexErrMsgTxt("hCRF: Two input parameters: model and features.\n");

	if(!mxIsStruct(prhs[0]))
		mexErrMsgTxt("hCRF: First parameter should be a structure.\n");
	if(!mxIsStruct(prhs[1]))
		mexErrMsgTxt("hCRF: Second parameter should be a structure.\n");

	mxArray* matNbStates = mxGetField(prhs[0],0,"nbStates");
	if(matNbStates != NULL && !mxIsEmpty(matNbStates))
		toolbox->getModel()->setNumberOfStates((int)*mxGetPr(matNbStates));

	mxArray* matNbSeqLabels = mxGetField(prhs[0],0,"nbSeqLabels");
	if(matNbSeqLabels != NULL && !mxIsEmpty(matNbSeqLabels))
		toolbox->getModel()->setNumberOfSequenceLabels((int)*mxGetPr(matNbSeqLabels));

	mxArray* matNbStateLabel = mxGetField(prhs[0],0,"nbStateLabels");
	if(matNbStateLabel != NULL && !mxIsEmpty(matNbStateLabel))
		toolbox->getModel()->setNumberOfStateLabels((int)*mxGetPr(matNbStateLabel));

//	mxArray* matNbFeaturePerLabel = mxGetField(prhs[0],0,"nbFeaturePerLabel");
//	if(matNbFeaturePerLabel != NULL && !mxIsEmpty(matNbFeaturePerLabel))
//		toolbox->getModel()->setNumberOfFeaturePerLabel((int)*mxGetPr(matNbFeaturePerLabel));

	mxArray* matAdjencyMatrixType = mxGetField(prhs[0],0,"adjencyMatrixType");
	if(matAdjencyMatrixType != NULL && !mxIsEmpty(matAdjencyMatrixType))
		toolbox->getModel()->setAdjacencyMatType((eGraphTypes)(int)*mxGetPr(matAdjencyMatrixType));
	mxArray* matStateMatrixType = mxGetField(prhs[0],0,"stateMatrixType");
	if(matStateMatrixType != NULL && !mxIsEmpty(matStateMatrixType))
	{
		// We have to check for the different matrix type and pass aditional parameter if needed
		int stateMatrixType = (int)*mxGetPr(matStateMatrixType);
		if (stateMatrixType == STATEMAT_PREDEFINED) {
			mxArray* matStateMatrix = mxGetField(prhs[0],0,"stateMatrix");
			if(matStateMatrix == NULL || mxIsEmpty(matStateMatrix))
				mexErrMsgTxt("stateMatrix not defined but stateMatrixType==STATEMAT_PREDIFNED");
			iMatrix* matData = new iMatrix();
			copyMatToCpp(matData, matStateMatrix);
			toolbox->getModel()->setStateMatType(stateMatrixType, matData);
			delete matData;
		}
		else
			// No additional parameter needed
			toolbox->getModel()->setStateMatType(stateMatrixType);
	}
	mxArray* matFeatureMask = mxGetField(prhs[0],0,"featureMask");
	if(matFeatureMask != NULL && mxIsInt32(matFeatureMask))
	{
		iMatrix tmpFeatureMask ((int)mxGetN(matFeatureMask),(int)mxGetM(matFeatureMask));
		memcpy(tmpFeatureMask.get(),mxGetPr(matFeatureMask),mxGetM(matFeatureMask)*mxGetN(matFeatureMask)*sizeof(int));
		toolbox->getModel()->setFeatureMask(tmpFeatureMask);
	}

	int nbFeatureType = (int)mxGetN(prhs[1]);
	list<FeatureType*>::iterator itFeatureType = toolbox->getFeatureGenerator()->getListFeatureTypes().begin();
	for (int i = 0; i < nbFeatureType; i++)
	{
		mxArray* matOffset = mxGetField(prhs[1],i,"offset");
		if(matOffset != NULL && !mxIsEmpty(matOffset))
			(*itFeatureType)->setIdOffset((int)*mxGetPr(matOffset));

		mxArray* matNbFeatures = mxGetField(prhs[1],i,"nbFeatures");
		if(matNbFeatures != NULL && !mxIsEmpty(matNbFeatures))
			(*itFeatureType)->setNumberOfFeatures((int)*mxGetPr(matNbFeatures));

		mxArray* matOffsetPerLabel = mxGetField(prhs[1],i,"offsetPerLabel");
		if(matOffsetPerLabel != NULL && mxIsInt32(matOffsetPerLabel))
		{
			iVector tmpOffsetPerLabel ((int)mxGetN(matOffsetPerLabel)*(int)mxGetM(matOffsetPerLabel));
			memcpy(tmpOffsetPerLabel.get(),mxGetPr(matOffsetPerLabel),mxGetM(matOffsetPerLabel)*mxGetN(matOffsetPerLabel)*sizeof(int));
			(*itFeatureType)->setOffsetPerLabel(tmpOffsetPerLabel);
		}

		mxArray* matNbFeaturePerLabel = mxGetField(prhs[1],i,"nbFeaturePerLabel");
		if(matNbFeaturePerLabel != NULL && mxIsInt32(matNbFeaturePerLabel))
		{
			iVector tmpNbFeaturePerLabel ((int)mxGetN(matNbFeaturePerLabel)*(int)mxGetM(matNbFeaturePerLabel));
			memcpy(tmpNbFeaturePerLabel.get(),mxGetPr(matNbFeaturePerLabel),mxGetM(matNbFeaturePerLabel)*mxGetN(matNbFeaturePerLabel)*sizeof(int));
			(*itFeatureType)->setNbFeaturePerLabel(tmpNbFeaturePerLabel);
		}

		itFeatureType++;
	}
	mxArray* matWeights = mxGetField(prhs[0],0,"weights");
	if(matWeights != NULL && mxIsDouble(matWeights))
	{
		dVector* vecWeights = new dVector((int)(mxGetM(matWeights)*mxGetN(matWeights)));
		memcpy(vecWeights->get(),mxGetPr(matWeights),mxGetM(matWeights)*mxGetN(matWeights)*sizeof(double));
		toolbox->getModel()->setWeights(*vecWeights);
	}

}

static void getModel(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if(toolbox == NULL)
		mexErrMsgTxt("hCRF: No toolbox was previously created. Use function 'createToolbox' to create a toolbox.\n");

	if(nlhs > 0)
	{
		plhs[0] = mxCreateStructMatrix(1,1,0,0);
		int fieldIndex = mxAddField(plhs[0],"nbStates");
		mxSetFieldByNumber(plhs[0],0,fieldIndex,mxCreateDoubleScalar(toolbox->getModel()->getNumberOfStates()));
		fieldIndex = mxAddField(plhs[0],"nbSeqLabels");
		mxSetFieldByNumber(plhs[0],0,fieldIndex,mxCreateDoubleScalar(toolbox->getModel()->getNumberOfSequenceLabels()));
		fieldIndex = mxAddField(plhs[0],"nbStateLabels");
		mxSetFieldByNumber(plhs[0],0,fieldIndex,mxCreateDoubleScalar(toolbox->getModel()->getNumberOfStateLabels()));
//		fieldIndex = mxAddField(plhs[0],"nbFeaturePerLabel");
//		mxSetFieldByNumber(plhs[0],0,fieldIndex,mxCreateDoubleScalar(toolbox->getModel()->getNumberOfFeaturesPerLabel()));
		fieldIndex = mxAddField(plhs[0],"adjencyMatrixType");
		mxSetFieldByNumber(plhs[0],0,fieldIndex,mxCreateDoubleScalar(toolbox->getModel()->getAdjacencyMatType()));
		fieldIndex = mxAddField(plhs[0],"stateMatrixType");
		mxSetFieldByNumber(plhs[0],0,fieldIndex,mxCreateDoubleScalar(toolbox->getModel()->getStateMatType()));
		if (toolbox->getModel()->getStateMatType() == STATEMAT_PREDEFINED)
		{
#ifdef WIN64
			mxArray* matState = mxCreateNumericMatrix(toolbox->getModel()->getStatesPerLabel().getHeight(), toolbox->getModel()->getStatesPerLabel().getWidth(),mxINT32_CLASS,	mxREAL);
#else
			mxArray* matState = mxCreateNumericMatrix(toolbox->getModel()->getStatesPerLabel().getHeight(), toolbox->getModel()->getStatesPerLabel().getWidth(),mxINT32_CLASS,	mxREAL);
#endif
			memcpy(mxGetPr(matState),toolbox->getModel()->getStatesPerLabel().get(),toolbox->getModel()->getStatesPerLabel().getHeight()*toolbox->getModel()->getStatesPerLabel().getWidth()*sizeof(int));
			fieldIndex = mxAddField(plhs[0],"stateMatrix");
			mxSetFieldByNumber(plhs[0],0,fieldIndex, matState);
		}
		if(toolbox->getModel()->getWeights())
		{
			fieldIndex = mxAddField(plhs[0],"weights");
			mxArray* matWeights = mxCreateNumericMatrix(toolbox->getModel()->getWeights()->getHeight(), toolbox->getModel()->getWeights()->getWidth(),mxDOUBLE_CLASS,mxREAL);
			memcpy(mxGetPr(matWeights),toolbox->getModel()->getWeights()->get(),toolbox->getModel()->getWeights()->getHeight()*toolbox->getModel()->getWeights()->getWidth()*sizeof(double));
			mxSetFieldByNumber(plhs[0],0,fieldIndex,matWeights);
		}
//		fieldIndex = mxAddField(plhs[0],"adjencyMatrix");
//		mxArray* matAdjMatrix = mxCreateNumericMatrix(toolbox->getModel()->getInternalAdjencyMatrix()->getHeight(), toolbox->getModel()->getInternalAdjencyMatrix()->getWidth(),mxUCHAR32_CLASS,mxREAL);
//		memcpy(mxGetPr(matAdjMatrix),toolbox->getModel()->getInternalAdjencyMatrix()->get(),toolbox->getModel()->getInternalAdjencyMatrix()->getHeight()*toolbox->getModel()->getInternalAdjencyMatrix()->getWidth()*sizeof(int));
//		mxSetFieldByNumber(plhs[0],0,fieldIndex,matAdjMatrix);

//		fieldIndex = mxAddField(plhs[0],"stateMatrix");
//		mxArray* matStateMatrix = mxCreateNumericMatrix(toolbox->getModel()->getInternalStateMatrix()->getHeight(), toolbox->getModel()->getInternalStateMatrix()->getWidth(),mxINT32_CLASS,mxREAL);
//		memcpy(mxGetPr(matStateMatrix),toolbox->getModel()->getInternalStateMatrix()->get(),toolbox->getModel()->getInternalStateMatrix()->getHeight()*toolbox->getModel()->getInternalStateMatrix()->getWidth()*sizeof(int));
//		mxSetFieldByNumber(plhs[0],0,fieldIndex,matStateMatrix);

		fieldIndex = mxAddField(plhs[0],"featureMask");
		mxArray* matFeatureMask = mxCreateNumericMatrix(toolbox->getModel()->getFeatureMask()->getHeight(), toolbox->getModel()->getFeatureMask()->getWidth(),mxINT32_CLASS,mxREAL);
		memcpy(mxGetPr(matFeatureMask),toolbox->getModel()->getFeatureMask()->get(),toolbox->getModel()->getFeatureMask()->getHeight()*toolbox->getModel()->getFeatureMask()->getWidth()*sizeof(int));
		mxSetFieldByNumber(plhs[0],0,fieldIndex,matFeatureMask);

//		fieldIndex = mxAddField(plhs[0],"statesPerLabel");
//		mxArray* matStatesPerLabel = mxCreateNumericMatrix(toolbox->getModel()->getStatesPerLabel().getHeight(), toolbox->getModel()->getStatesPerLabel().getWidth(),mxINT32_CLASS,mxREAL);
//		memcpy(mxGetPr(matStatesPerLabel),toolbox->getModel()->getStatesPerLabel().get(),toolbox->getModel()->getStatesPerLabel().getHeight()*toolbox->getModel()->getStatesPerLabel().getWidth()*sizeof(int));
//		mxSetFieldByNumber(plhs[0],0,fieldIndex,matStatesPerLabel);

//		fieldIndex = mxAddField(plhs[0],"labelPerState");
//		mxArray* matLabelPerState = mxCreateNumericMatrix(toolbox->getModel()->getLabelPerState().getHeight(), toolbox->getModel()->getLabelPerState().getWidth(),mxINT32_CLASS,mxREAL);
//		memcpy(mxGetPr(matLabelPerState),toolbox->getModel()->getLabelPerState().get(),toolbox->getModel()->getLabelPerState().getHeight()*toolbox->getModel()->getLabelPerState().getWidth()*sizeof(int));
//		mxSetFieldByNumber(plhs[0],0,fieldIndex,matLabelPerState);
	}
	if(nlhs > 1)
	{
		int nbFeatureType = (int)toolbox->getFeatureGenerator()->getListFeatureTypes().size();
		plhs[1] = mxCreateStructMatrix(1,nbFeatureType,0,0);
		list<FeatureType*>::iterator itFeatureType = toolbox->getFeatureGenerator()->getListFeatureTypes().begin();
		mxAddField(plhs[1],"name");
		mxAddField(plhs[1],"id");
		mxAddField(plhs[1],"offset");
		mxAddField(plhs[1],"nbFeatures");
		mxAddField(plhs[1],"offsetPerLabel");
		mxAddField(plhs[1],"nbFeaturePerLabel");
		for(int i = 0; i < nbFeatureType; i++)
		{
			mxSetFieldByNumber(plhs[1],i,0,mxCreateString((*itFeatureType)->getFeatureTypeName()));
			mxSetFieldByNumber(plhs[1],i,1,mxCreateDoubleScalar((*itFeatureType)->getFeatureTypeId()));
			mxSetFieldByNumber(plhs[1],i,2,mxCreateDoubleScalar((*itFeatureType)->getIdOffset()));
			mxSetFieldByNumber(plhs[1],i,3,mxCreateDoubleScalar((*itFeatureType)->getNumberOfFeatures()));
			mxArray* matOffsetPerLabel = mxCreateNumericMatrix((*itFeatureType)->getOffsetPerLabel().getHeight(), (*itFeatureType)->getOffsetPerLabel().getWidth(),mxINT32_CLASS,mxREAL);
			memcpy(mxGetPr(matOffsetPerLabel),(*itFeatureType)->getOffsetPerLabel().get(),(*itFeatureType)->getOffsetPerLabel().getHeight()*(*itFeatureType)->getOffsetPerLabel().getWidth()*sizeof(int));
			mxSetFieldByNumber(plhs[1],i,4,matOffsetPerLabel);
			mxArray* matFeaturesPerLabel = mxCreateNumericMatrix((*itFeatureType)->getNbFeaturePerLabel().getHeight(), (*itFeatureType)->getNbFeaturePerLabel().getWidth(),mxINT32_CLASS,mxREAL);
			memcpy(mxGetPr(matFeaturesPerLabel),(*itFeatureType)->getNbFeaturePerLabel().get(),(*itFeatureType)->getNbFeaturePerLabel().getHeight()*(*itFeatureType)->getNbFeaturePerLabel().getWidth()*sizeof(int));
			mxSetFieldByNumber(plhs[1],i,5,matFeaturesPerLabel);
			itFeatureType++;
		}
	}

}



static void unloadData(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	/* This function free the memory occupied by the data if any
	*/
	if(data == NULL)
		return;

	delete data;
	data = NULL;
}

static void clearToolbox(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	/*
	This function free the memory occupied by the toolbox (if any)
	*/
	if(toolbox == NULL)
		return;

	delete toolbox;
	toolbox = NULL;
}

static void make_func_lookup()
{
	// This function generate the look up map for the function in the mexfile
	mxAssert(func_lookup==0, "Function lookup already assigned");
	func_lookup = new DISPATCHTAB();
#define ADD(x) (*func_lookup)[#x] = x
	ADD(createToolbox);
	ADD(loadData);
	ADD(setData);
	ADD(train);
	ADD(test);
	ADD(get);
	ADD(set);
	ADD(getAllFeatures);
	ADD(getResults);
	ADD(loadModel);
	ADD(saveModel);
	ADD(setModel);
	ADD(getModel);
	ADD(unloadData);
	ADD(clearToolbox);
#undef ADD
}

void mexClean( void )
// This function is called by matlab (registered via mexAtExit). It release the used memory.
{
	if(data)
	{
		delete data;
		data = 0;
	}
	if(toolbox)
	{
		delete toolbox;
		toolbox = 0;
	}
}



void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
	if(nrhs<1)
		mexErrMsgTxt("hCRF: must specify operation name.\n");
	if(func_lookup==0)
	{
		// This is the first cal. We can decide if we go in object or
		// standard mode 
		if (mxIsUint64(prhs[0])){
			// We cannot clear the mex file because the object contain pointer
			// to memory allocated by the c++ code. 
			mexLock();
			objectMode = true;
		}
		make_func_lookup();
		mexAtExit(&mexClean);
	}
	if (objectMode == true)
	{
		if (!mxIsUint64(prhs[0]))
			mexErrMsgTxt("hCRF: object mode but no pointer are present at the start");
		unsigned long long * data_ptr;
		data_ptr = (unsigned long long *)mxGetData(prhs[0]);
		toolbox = (Toolbox *) data_ptr[0];
		data = (DataSet *) data_ptr[1];
		// We remove the argument from the argument list
		prhs++;
		nrhs--;
	}
	// we proceed on normaly

	// Get the name of the operation in a char array
	char str[100];
	int r = mxGetString(prhs[0], str, sizeof(str));
	if(r!=0) 
		mexErrMsgTxt("hCRF: first argument must be a string.\n");
// A map is used to find the relevant function to call
	DISPATCHTAB::iterator it = func_lookup->find(str);
	if(it == func_lookup->end())
	{
		char error[255];
		sprintf(error, "hCRF: unknown operation: %s", str);
		mexErrMsgTxt(error);
	}
	if (objectMode)
	{
		if (nlhs < 1)
			mexErrMsgTxt("hCRF: in object mode, at least one return argument is needed");
		(it->second)(nlhs-1, plhs+1, nrhs-1,prhs+1);
		// We return toolbox and data if they have changed
		const mwSize dims [] = {2,1};
		plhs[0] = mxCreateNumericArray(2, dims, mxUINT64_CLASS, mxREAL);
		unsigned long long * data_ptr;
		data_ptr = (unsigned long long *)mxGetData(plhs[0]);
		data_ptr[0] = (unsigned long long)toolbox;
		data_ptr[1] = (unsigned long long)data;
	}
	else
		(it->second)(nlhs, plhs, nrhs-1,prhs+1);

}


