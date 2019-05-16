/* The interface description for wraping the hCRF library
   Hugues Salamin. 2009/09/25
*/
%module PyHCRF
%{
#define SWIG_FILE_WITH_INIT
#include "hCRF.h"
%}
%include "typemaps.i"
%include "std_vector.i"
%include "numpy.i"
%include "exception.i"

%exception{
   try{
      $action
   }
   catch(BadFileName &e) {
      SWIG_exception(SWIG_IOError, e.what());
   }
   catch(HcrfNotImplemented &e) {
      SWIG_exception(SWIG_UnknownError, e.what());
   }
   catch(HcrfBadModel &e) {
      SWIG_exception(SWIG_ValueError, e.what());
   }
   catch(InvalidOptimizer &e) {
      SWIG_exception(SWIG_ValueError, e.what());
   }
 }

%init %{
   import_array();
   %}

%template(DataSequenceVector) std::vector<DataSequence *>;


%include "toolbox.h"
%include "dataset.h"
%include "model.h"

#ifndef $self
#define $self self
#endif

%extend DataSet{
    PyObject* getLabels(){
        /* This function return the labels of a dataSet, to be used for computing
           performance measure
        */
         npy_intp* dims = new npy_intp[2];
         PyObject* label = PyList_New($self->size());
         int seqIndex = 0;
         for(std::vector<DataSequence*>::iterator itSeq =$self->begin(); itSeq != $self->end(); itSeq++)
         {
             if ((*itSeq)->getEstimatedStateLabels() == NULL){
                 PyErr_SetString(PyExc_RuntimeError, "No label  where estimated");
                 return NULL;
             }
             dims[0] = (*itSeq)->getStateLabels()->getWidth();
             int* data = (*itSeq)->getStateLabels()->get();
             PyObject * temp = PyArray_New(&PyArray_Type, 1, dims, NPY_INT,0 ,0 ,0 ,0 ,0);
             memcpy(PyArray_BYTES(temp), data,  dims[0] * sizeof(int));
             PyList_SetItem(label, seqIndex, temp);
             seqIndex++;
         };
         return label;
    }
 }

%extend Toolbox{
   PyObject* getResults(DataSet& X){
       /*
         This function return a tuple of two element.  The first
         element is a list of numpy array containing the probablity
         computed by the model. The second element is a list of label
         associated with each time for every element in the data
         sequence The toolbox must be trained
       */
       npy_intp* dims = new npy_intp[2];
       PyObject* label = PyList_New(X.size());
       int seqIndex = 0;
       for(std::vector<DataSequence*>::iterator itSeq = X.begin(); itSeq != X.end(); itSeq++)
       {
           if ((*itSeq)->getEstimatedStateLabels() == NULL){
               PyErr_SetString(PyExc_RuntimeError, "No label  where estimated");
               return NULL;
           }
           dims[0] = (*itSeq)->getEstimatedStateLabels()->getHeight();
           int* data = (*itSeq)->getEstimatedStateLabels()->get();
           PyObject * temp = PyArray_New(&PyArray_Type, 1, dims, NPY_INT,0 ,0 ,0 ,0 ,0);
           memcpy(PyArray_BYTES(temp), data,  dims[0] * sizeof(int));
           PyList_SetItem(label, seqIndex, temp);
           seqIndex++;
       };
       PyObject* proba =  PyList_New(X.size());
       seqIndex = 0;
       for(std::vector<DataSequence*>::iterator itSeq = X.begin(); itSeq != X.end(); itSeq++)
       {
           if ((*itSeq)->getEstimatedProbabilitiesPerStates() == NULL){
               PyErr_SetString(PyExc_RuntimeError, "No probability where estimated");
               return NULL;
           }
           dims[1] = (*itSeq)->getEstimatedProbabilitiesPerStates()->getHeight();
           dims[0] = (*itSeq)->getEstimatedProbabilitiesPerStates()->getWidth();
           double* data = (*itSeq)->getEstimatedProbabilitiesPerStates()->get();
           PyObject * temp = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,0 ,0 ,0 ,0 ,0);
           memcpy(PyArray_BYTES(temp), data,  dims[0] * dims[1] * sizeof(double));
           PyList_SetItem(proba, seqIndex, temp);
           seqIndex++;
       };
       delete[] dims;
       PyObject* ans = PyTuple_New(2);
       PyTuple_SetItem(ans, 0, label);
       PyTuple_SetItem(ans, 1, proba);
       return ans;
   }
 }

%extend ToolboxCRF{
   %feature("docstring") getModel "This function need to know the dataset on which the model was trained.
This function return two numpy arrays.
The first array is the weights between label and features. Arr1[label_num][features_num] where all index are zero based.
The second array contains the weight between the labels. Arr2[current_label][next_label]"
   PyObject* getModel(DataSet& X){
	  PyObject* ans = PyTuple_New(2);
	  double* data = $self->getModel()->getWeights(-1)->get();
	  npy_intp* dims = new npy_intp[2];
	  dims[0] = $self->getModel()->getNumberOfStates();
	  dims[1] = X.getNumberofRawFeatures();
	  PyObject * temp = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,0 ,0 ,0 ,0 ,0);
	  memcpy(PyArray_BYTES(temp), data,  dims[0] * dims[1] * sizeof(double));
	  data += dims[0]*dims[1];
	  PyTuple_SetItem(ans, 0, temp);
	  dims[0] = $self->getModel()->getNumberOfStates();
	  dims[1] = dims[0];
	  temp = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,0 ,0 ,0 ,0 ,0);
	  memcpy(PyArray_BYTES(temp), data,  dims[0] * dims[1] * sizeof(double));
	  PyTuple_SetItem(ans, 1, temp);
	  return ans;
   }
 }

