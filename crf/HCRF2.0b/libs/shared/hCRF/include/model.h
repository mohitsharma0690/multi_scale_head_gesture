//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Model Component
//
//	January 19, 2006

#ifndef MODEL_H
#define MODEL_H

#define MAX_NUMBER_OF_LEVELS 10

//Standard Template Library includes
#include <vector>
#include <iostream>
#include "hcrfExcep.h"

//hCRF library includes
#include "dataset.h"

class FeatureGenerator;

enum eFeatureTypes
{
   allTypes = 0,
   edgeFeaturesOnly,
   nodeFeaturesOnly
};


// Type of graph topology
enum eGraphTypes {
   CHAIN,
   DANGLING_CHAIN, // chain of hidden states with attached labels
   ADJMAT_PREDEFINED
};

enum{
   ALLSTATES,
   STATES_BASED_ON_LABELS,
   STATEMAT_PREDEFINED,
   STATEMAT_PROBABILISTIC
};

//-------------------------------------------------------------
// Model Class
//

class Model {
public:
   Model(int numberOfStates = 0, int numberOfSeqLabels = 0,
         int numberOfStateLabels = 0);
   ~Model();
   void setAdjacencyMatType(eGraphTypes atype, ...);
   eGraphTypes getAdjacencyMatType();
   int setStateMatType(int stype, ...);
   int getStateMatType();
// adjacency and state matrix sizes are max-sizes, based on the
// longest sequences seen thus far; use sequences length instead for
// width and height of these matrices
   void getAdjacencyMatrix(uMatrix&, DataSequence* seq);
   iMatrix * getStateMatrix(DataSequence* seq);
   iVector * getStateMatrix(DataSequence* seq, int nodeIndex);

   void setWeights(const dVector& weights);
   dVector * getWeights(int seqLabel = -1);
   void refreshWeights();

   int getNumberOfStates() const;
   void setNumberOfStates(int numberOfStates);

   int getNumberOfStateLabels() const;
   void setNumberOfStateLabels(int numberOfStateLabels);

   int getNumberOfSequenceLabels() const;
   void setNumberOfSequenceLabels(int numberOfSequenceLabels);
   
   int getNumberOfRawFeaturesPerFrame();
   void setNumberOfRawFeaturesPerFrame(int numberOfRawFeaturesPerFrame);	

   void setRegL1Sigma(double sigma, eFeatureTypes typeFeature = allTypes);
   void setRegL2Sigma(double sigma, eFeatureTypes typeFeature = allTypes);
   double getRegL1Sigma();
   double getRegL2Sigma();
   eFeatureTypes getRegL1FeatureTypes();
   eFeatureTypes getRegL2FeatureTypes();
   void setFeatureMask(iMatrix &ftrMask);
   iMatrix* getFeatureMask();
   int getNumberOfFeaturesPerLabel();

   iMatrix& getStatesPerLabel();
   iVector& getLabelPerState();
   int getDebugLevel();
   void setDebugLevel(int newDebugLevel);

   void load(const char* pFilename);
   void save(const char* pFilename) const;

   int read(std::istream* stream);
   int write(std::ostream* stream) const;

   uMatrix* getInternalAdjencyMatrix();
   iMatrix *getInternalStateMatrix();

private:
   int numberOfSequenceLabels;
   int numberOfStates;
   int numberOfStateLabels;
   int numberOfFeaturesPerLabel;   
   int numberOfRawFeaturesPerFrame;

   dVector weights;
   std::vector<dVector> weights_y;

   double regL1Sigma;
   double regL2Sigma;
   eFeatureTypes regL1FeatureType;
   eFeatureTypes regL2FeatureType;

   int debugLevel;

   eGraphTypes adjMatType;
   int stateMatType;
   uMatrix adjMat;
   iMatrix stateMat, featureMask;
   iMatrix statesPerLabel;
   iVector stateVec, labelPerState;


   int loadAdjacencyMatrix(const char *pFilename);
   int loadStateMatrix(const char *pFilename);

   void makeChain(uMatrix& m, int n);
   void predefAdjMat(uMatrix& m, int n);
   iMatrix * makeFullStateMat(int n);
   iMatrix * makeLabelsBasedStateMat(DataSequence* seq);
   iMatrix * predefStateMat(int n);
   void updateStatesPerLabel();
};

// stream io routines
std::istream& operator >>(std::istream& in, Model& m);
std::ostream& operator <<(std::ostream& out, const Model& m);

#endif
