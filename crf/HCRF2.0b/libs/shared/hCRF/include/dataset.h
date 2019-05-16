//-------------------------------------------------------------
// Hidden Conditional Random Field Library - DataSet Component
//
//	January 19, 2006

#ifndef DATASET_H
#define DATASET_H

//Standard Template Library includes
#include <vector>
#include <list>
#include <fstream>

//library includes
#include "matrix.h"
#include "hcrfExcep.h"

// #define HIDDEN -10000;

//-------------------------------------------------------------
// DataSequence Class
//



class DataSequence {
public:
    DataSequence();
    DataSequence(const DataSequence& other);
    DataSequence& operator=(const DataSequence&);
    virtual ~DataSequence();
    DataSequence(dMatrix* precomputedFeatures, iVector* stateLabels, int sequenceLabel);

    int load(std::istream* isData, std::istream* isLabels,
             std::istream* isAdjMat, std::istream* isStatesPerNodes, 
			 std::istream* isDataSparse);
    void equal(const DataSequence & seq);

//	void set(const std::vector<DataSample>& x);
//	void set(const DataSample& value, unsigned int nodeIndex);

//	DataSample get(int nodeIndex);

    int	length() const;

    void setStateLabels(iVector *v);
    iVector* getStateLabels() const;
    int getStateLabels(int nodeIndex) const;

    void setAdjacencyMatrix(uMatrix *m);
    void getAdjacencyMatrix(uMatrix& m) const;

    void setPrecomputedFeatures(dMatrix *m);
    dMatrix* getPrecomputedFeatures() const;
	
	void setPrecomputedFeaturesSparse(dMatrixSparse *m);
	dMatrixSparse* getPrecomputedFeaturesSparse() const;

    void setStatesPerNode(iMatrix* spn);
    iMatrix* getStatesPerNode() const;

    void setSequenceLabel(int seqLabel);
    int getSequenceLabel() const;

    void setEstimatedStateLabels(iVector *v);
    iVector* getEstimatedStateLabels() const;

    void setEstimatedSequenceLabel(int seqLabel);
    int getEstimatedSequenceLabel() const;

    void setEstimatedProbabilitiesPerStates(dMatrix *m);
    dMatrix* getEstimatedProbabilitiesPerStates() const;

    void  setWeightSequence(double w);
    double getWeightSequence() const;

  protected:
    void init();
    int sequenceLabel;
    double	 weightSequence;
    iVector* stateLabels;
    iMatrix* statesPerNode;
    uMatrix* adjMat;
    dMatrix* precompFeatures;
    dMatrixSparse* precompFeaturesSparse;
    int		 estimatedSequenceLabel;
    iVector* estimatedStateLabels;
    dMatrix* estimatedProbabilitiesPerStates;
};


class DataSequenceRealtime: public DataSequence
{
public:
	DataSequenceRealtime();
	DataSequenceRealtime(int windowSize, int width, int height, int numberOfLabels);
	virtual ~DataSequenceRealtime();
	int init(int windowSize, int width, int height, int numberOfLabels);
	void push_back(const dVector* const featureVector);
	dVector* getAlpha();
	void initializeAlpha(int height);
	int getPosition();
	int getWindowSize();
	bool isReady();
	

private:
	int pos; // position to insert the new frame
	dVector* alpha; // scanned until pos-1
	bool ready;
	int width;
	int height;
	int windowSize;
};





//-------------------------------------------------------------
// DataSet Class
//

class DataSet
{
  public:
    DataSet();
    ~DataSet();
    DataSet(const char *fileData, const char *fileStateLabels = NULL,
            const char *fileSeqLabels = NULL, const char * fileAdjMat = NULL,
            const char * fileStatesPerNodes = NULL, const char * fileDataSparse = NULL);

    int load(const char *fileData, const char *fileStateLabels = NULL,
             const char *fileSeqLabels = NULL, const char * fileAdjMat = NULL,
             const char * fileStatesPerNodes = NULL, const char * fileDataSparse = NULL);
    void clearSequence();
	
    int searchNumberOfStates();
    int searchNumberOfSequenceLabels();
    int getNumberofRawFeatures();
    
	void insert(std::vector<DataSequence*>::iterator iter, DataSequence* d){
		container.insert(iter, d);
	}
	
	DataSequence* at (size_t i) const{
       return container.at(i);
    }
    size_t size() const{
       return container.size();
    }
    typedef std::vector<DataSequence*>::iterator iterator;
    iterator begin(){
        return container.begin();
    }
    iterator end(){
        return container.end();
    }
    typedef std::vector<DataSequence*>::const_iterator const_iterator;
    const_iterator begin() const{
        return container.begin();
    }
    const_iterator end() const{
        return container.end();
    }
  private:
    std::vector<DataSequence*> container;
};

std::ostream& operator <<(std::ostream& out, const DataSequence& seq);
std::ostream& operator <<(std::ostream& out, const DataSet& data);

#endif
