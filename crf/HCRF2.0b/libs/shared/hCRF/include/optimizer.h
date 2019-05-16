//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Optimizer
// Component
//
//	January 30, 2006

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

//hCRF Library includes
#include "dataset.h"
#include "model.h"
#include "gradient.h"
#include "evaluator.h"
#include "matrix.h"

//#include "asa.h"
//#include "asa_usr_asa.h"
//#include "asa_usr.h"
// We derive from OWLQN, so we have to include the header
#ifdef USEOWL
	#include "OWLQN.h"
#endif
//#ifdef USELBFGS
//	#include "lbfgs.h"
//#endif

class Optimizer {
public:
   Optimizer();
   virtual ~Optimizer();
//	char* getName()=0;
   virtual void optimize(Model* m, DataSet* X,
                         Evaluator* eval, Gradient* grad);
   virtual void optimize(Model* m, DataSet* X,
                         Evaluator* eval, GradientPerceptron* grad);
   virtual void setMaxNumIterations(int maxiter);
   virtual int getMaxNumIterations();
   virtual int getLastNbIterations();
   virtual double getLastFunctionError();
   virtual double getLastNormGradient();

protected:
   int maxit;
   void setConvergenceTolerance(double tolerance);
   int lastNbIterations;
   double lastFunctionError;
   double lastNormGradient;
};

class OptimizerCG: public Optimizer
{
public:
   OptimizerCG();
   ~OptimizerCG();
   virtual void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);

protected:
   static double callbackComputeError(double* weights);
   static void callbackComputeGradient(double* gradient, double* weights);
};



enum typeOptimizer
{
   optimBFGS = 0,
   optimDFP,
   optimFR,
   optimFRwithReset,
   optimPR,
   optimPRwithReset,
   optimPerceptronInitZero
};

class UnconstrainedOptimizer;


class OptimizerUncOptim: public Optimizer
{
public:
   ~OptimizerUncOptim();
   OptimizerUncOptim(typeOptimizer defaultOptimizer = optimBFGS);
   OptimizerUncOptim(const OptimizerUncOptim&);
   OptimizerUncOptim& operator=(const OptimizerUncOptim&){
        throw std::logic_error("Optimizer should not be copied");
    }
   void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);

private:
   UnconstrainedOptimizer* internalOptimizer;
   typeOptimizer optimizer;

};

struct USER_DEFINES;
typedef long int ALLOC_INT;

class OptimizerASA: public Optimizer
{
public:
    OptimizerASA();
    ~OptimizerASA();
    OptimizerASA(const OptimizerASA&);
    OptimizerASA& operator=(const OptimizerASA&){
        throw std::logic_error("Optimizer should not be copied");
    }
    virtual void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);
	
protected:
    // Protected, no need to have the definition here (we want to avoid including asas)
    static double callbackComputeError(double* weights,
                                       double *parameter_lower_bound,
                                       double *parameter_upper_bound,
                                       double *cost_tangents,
                                       double *cost_curvature,
                                       ALLOC_INT * parameter_dimension,
                                       int *parameter_int_real,
                                       int *cost_flag, int *exit_code,
                                       USER_DEFINES * USER_OPTIONS);
    static void callbackComputeGradient(double* gradient, double* weights);


private:
    double *parameter_lower_bound,  *parameter_upper_bound,  *cost_parameters;
    double *cost_tangents,  *cost_curvature;
    double cost_value;
    int *exit_code;
    USER_DEFINES *USER_OPTIONS;
    long int *rand_seed;
    int *parameter_int_real;
    long int *parameter_dimension;
    long int n_param;
    int initialize_parameters_value;
    int *cost_flag;
    static double (*rand_func_ptr)(long int *);
    static double randflt (long int * rand_seed);
    static double resettable_randflt (long int * rand_seed, int reset);
    static double myrand (long int * rand_seed);
    static int initialize_parameters(double *cost_parameters,
                                     double* parameter_lower_bound,
                                     double *parameter_upper_bound,
                                     long int *parameter_dimension,
                                     int *parameter_int_real);
    static Model* currentModel;
    static DataSet* currentDataset;
    static Evaluator* currentEvaluator;

};

#ifdef USEOWL
class OptimizerOWL: public Optimizer,DifferentiableFunction
{
public:
   ~OptimizerOWL();
   OptimizerOWL();
   OptimizerOWL(const OptimizerOWL&);
   OptimizerOWL& operator=(const OptimizerOWL&){
        throw std::logic_error("Optimizer should not be copied");
   };
   void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);

protected:
   double Eval(const DblVec& input, DblVec& gradient);
private:
   Model* currentModel;
   DataSet* currentDataset;
   Evaluator* currentEvaluator;
   Gradient* currentGradient;
//   typeOptimizer optimizer;
   dVector vecGradient;
   OWLQN opt;
   DifferentiableFunction *obj;
};
#endif

#ifdef USELBFGS
typedef double lbfgsfloatval_t;
class OptimizerLBFGS: public Optimizer
{
  public:
    ~OptimizerLBFGS();
    OptimizerLBFGS();
    OptimizerLBFGS(const OptimizerLBFGS&);
    OptimizerLBFGS& operator=(const OptimizerLBFGS&){
        throw std::logic_error("Optimizer should not be copied");
    }
    void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);
    
   
  protected:
    static lbfgsfloatval_t _evaluate( void *instance, const lbfgsfloatval_t *x,
                                      lbfgsfloatval_t *g, const int n,
                                      const lbfgsfloatval_t)
    {
        return reinterpret_cast<OptimizerLBFGS*>(instance)->Eval(x, g, n);
    }

    static int _progress( void *instance, const lbfgsfloatval_t *x,
                          const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                          const lbfgsfloatval_t xnorm,
                          const lbfgsfloatval_t gnorm,
                          const lbfgsfloatval_t step, int n, int k, int ls )
    {
        return reinterpret_cast<OptimizerLBFGS*>(instance)->progress(x, g, fx,
                                                                     xnorm,
                                                                     gnorm,
                                                                     step, n
                                                                     , k, ls);
    }
    
    double Eval(const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n);
    int progress( const lbfgsfloatval_t *x, const lbfgsfloatval_t *g,
                  const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,
                  const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
                  int n, int k, int ls);

  private:
    Model* currentModel;
    DataSet* currentDataset;
    Evaluator* currentEvaluator;
    Gradient* currentGradient;
    typeOptimizer optimizer;
    dVector vecGradient;
};
#endif

class OptimizerPerceptron: public Optimizer
{
public:
   OptimizerPerceptron(typeOptimizer defaultOptimizer = optimPerceptronInitZero);
   ~OptimizerPerceptron();
   virtual void optimize(Model* m, DataSet* X,Evaluator* eval, GradientPerceptron* grad);
private:
	Model* currentModel;
	DataSet* currentDataset;
	Evaluator* currentEvaluator;
	GradientPerceptron* currentGradient;  
    typeOptimizer optimizer;
	dVector vecGradient; // Is it useful?
};

#endif

