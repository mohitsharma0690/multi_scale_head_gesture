//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Feature Generator
// Component
//
//	January 30, 2006

#ifndef FEATUREGENERATOR_H
#define FEATUREGENERATOR_H

//include Standard Template Library
#include<list>

#include "dataset.h"
#include "model.h"

enum {
	RAW_FEATURE_ID,	
	EDGE_FEATURE_ID,
	WINDOW_RAW_FEATURE_ID,
	WINDOW_RAW_FEATURE_ID2,
	WINDOW_RAW_FEATURE_REALTIME_ID,
	LABEL_EDGE_FEATURE_ID,
	SQUARE_RAW_FEATURE_ID,
	ONE_FEATURE_ID,
	LATENT_LABEL_FEATURE_ID,
	EDGE_OBSERVATION_FEATURE_ID
};

#define LAST_FEATURE_ID EDGE_OBSERVATION_FEATURE_ID



class feature {
public:
	int id,globalId;
	double value;
	int nodeState, prevNodeState, sequenceLabel;
	int nodeIndex, prevNodeIndex;
};

class featureVector
{
public:
	featureVector();
	featureVector(const featureVector& source);
	featureVector& operator= (const featureVector& source);
	~featureVector();

	feature* addElement();
	feature* getPtr();

	int size();
	void resize(int newSize);
	void clear();

private:
	int realSize;
	int capacity;
	feature* pFeatures;
};

class FeatureType
{
public:
	FeatureType();
	virtual ~FeatureType();

	virtual void init(const DataSet&, const Model&);
	virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, 
					int nodeIndex, int prevNodeIndex, int seqLabel = -1) = 0;
	virtual void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures) = 0;
	virtual void computeFeatureMask(iMatrix& matFeatureMask, const Model& m);
	virtual void read(std::istream& is);
	virtual void write(std::ostream& os);

	virtual char* getFeatureTypeName();
	virtual int getFeatureTypeId();

	virtual bool isEdgeFeatureType();
	int getNumberOfFeatures(int seqLabel = -1);
	void setIdOffset(int offset, int seqLabel = -1);
	void setNumberOfFeatures(int nFeatures);

	iVector& getOffsetPerLabel();
	iVector& getNbFeaturePerLabel();

	void setOffsetPerLabel(const iVector& newOffsetPerLabel);
	void setNbFeaturePerLabel(const iVector& newNbFeaturePerLabel);

	inline int getIdOffset(int seqLabel = -1)
	{
		if (seqLabel == -1 || idOffsetPerLabel.getLength() == 0)
			return idOffset;
		else
			return idOffsetPerLabel[seqLabel];
	}

protected:
	int idOffset;
	int nbFeatures;
	iVector idOffsetPerLabel;
	iVector nbFeaturesPerLabel;

	std::string strFeatureTypeName;
	int featureTypeId;
};

class FeatureGenerator {
public:
	FeatureGenerator();
	~FeatureGenerator();

	void addFeature(FeatureType* featureGen);
	void initFeatures(const DataSet& dataset, Model& m);
	void clearFeatureList();

#if defined(_VEC_FEATURES) || defined(_OPENMP)
	void getFeatures(featureVector& vecFeatures, DataSequence* X, Model* m, 
                   int nodeIndex, int prevNodeIndex, int seqLabel = -1);
#else
	featureVector* getFeatures(DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel = -1);
#endif
	int  getNumberOfFeatures(eFeatureTypes typeFeature = allTypes, int seqLabel = -1);
	double evaluateLabels(DataSequence* X, Model* m, int seqLabel = -1);

	featureVector* getAllFeatures(Model* m, int nbRawFeatures);

	void load(char* pFilename);
	void save(char* pFilename);

	std::list<FeatureType*>& getListFeatureTypes();
	void setMaxNumberThreads(int maxThreads);
	

private:
	std::list<FeatureType*> listFeatureTypes;
	featureVector vecFeatures;

	featureVector *vecFeaturesMP;
	int nbThreadsMP;

};

std::ostream& operator <<(std::ostream& out, const feature& f);
std::ostream& operator <<(std::ostream& out, featureVector& v);
#endif
