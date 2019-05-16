#include "toolbox.h"
#include "optimizer.h"

ToolboxGHCRF::ToolboxGHCRF():ToolboxHCRF()
{
}

ToolboxGHCRF::ToolboxGHCRF(int nbHiddenStates, int opt, int windowSize):ToolboxHCRF()
{
	init(nbHiddenStates,opt, windowSize);
	setWeightInitType(INIT_GAUSSIAN);
}

ToolboxGHCRF::~ToolboxGHCRF()
{

}

// We dont use WindowSize
void ToolboxGHCRF::init(int nbHiddenStates, int opt, int )
{
	pModel = new Model();
	pModel->setNumberOfStates(nbHiddenStates);
	numberOfHiddenStates = nbHiddenStates; //{+KGB}

	pFeatureGenerator = new FeatureGenerator;
	pFeatureGenerator->addFeature(new EdgeFeatures());
	pFeatureGenerator->addFeature(new LabelEdgeFeatures());

	pFeatureGenerator->addFeature(new RawFeatures());
	pFeatureGenerator->addFeature(new FeaturesOne());
	pFeatureGenerator->addFeature(new RawFeaturesSquare());

	pInferenceEngine = new InferenceEngineFB(); //{=KGB}
	pGradient = new GradientHCRF (pInferenceEngine, pFeatureGenerator);
	pEvaluator = new EvaluatorHCRF (pInferenceEngine, pFeatureGenerator);

	if( opt == OPTIMIZER_BFGS)
		pOptimizer = new OptimizerUncOptim();
	else if(opt == OPTIMIZER_CG)
		pOptimizer = new OptimizerCG();
	#ifdef USEOWL
		else if(opt == OPTIMIZER_OWLQN)
			pOptimizer = new OptimizerOWL();
	#endif
	#ifdef USELBFGS
		else if(opt == OPTIMIZER_LBFGS)
			pOptimizer = new OptimizerLBFGS();
	#endif
	#ifndef _PUBLIC
	else if(opt== OPTIMIZER_ASA)
		pOptimizer  = new OptimizerASA();
	#endif

}

