#include "MyToolbox.h"

MyToolbox::MyToolbox():ToolboxCRF()
{
}

MyToolbox::MyToolbox(int opt, int windowSize):ToolboxCRF()
{
	init(opt);
}

MyToolbox::~MyToolbox()
{

}

void MyToolbox::init(int opt)
{
	pModel = new Model();

	pFeatureGenerator = new FeatureGenerator;
	//Replace the following line by our new class: MyFeature
//	pFeatureGenerator->addFeature(new WindowRawFeatures(windowSize));
	pFeatureGenerator->addFeature(new MyFeatures());
	pFeatureGenerator->addFeature(new EdgeFeatures());

	pInferenceEngine = new InferenceEngineBP();
	pGradient = new GradientCRF (pInferenceEngine, pFeatureGenerator);
	pEvaluator = new EvaluatorCRF (pInferenceEngine, pFeatureGenerator);

	if( opt == OPTIMIZER_BFGS)
		pOptimizer = new OptimizerUncOptim();
#ifndef _PUBLIC
	else if( opt == OPTIMIZER_ASA)
		pOptimizer = new OptimizerASA();
#ifdef USEOWL
	else if(opt == OPTIMIZER_OWLQN)
	   pOptimizer = new OptimizerOWL();
#else
	else if(opt == OPTIMIZER_OWLQN)
	   raise InvalidOptimizer("Not support for OWLQN compiled in the library");
#endif
#endif

#ifdef USELBFGS
	else if(opt == OPTIMIZER_LBFGS)
		pOptimizer = new OptimizerLBFGS();
#else
	else if(opt == OPTIMIZER_LBFGS)
	   raise InvalidOptimizer("Not support for LBFGS compiled in the library");
#endif
	else
		pOptimizer = new OptimizerCG();
}
