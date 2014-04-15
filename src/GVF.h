///////////////////////////////////////////////////////////////////////
//
//  GVF class
//
//  The library (GVF.cpp, GVF.h) has been created in 2010-2011 at Ircam Centre Pompidou by
//  - Baptiste Caramiaux
//  previously with Ircam Centre Pompidou and University Paris VI, since 2012 with Goldsmiths College, University of London
//  - Nicola Montecchio
//  previously with University of Padova, since 2012 with The Echo Nest
//
//  The library is maintained by Baptiste Caramiaux at Goldsmiths College, University of London
//
//  Copyright (C) 2013 Baptiste Caramiaux, Nicola Montecchio - STMS lab Ircam-CRNS-UPMC, University of Padova
//
//  The library is under the GNU Lesser General Public License (LGPL v3)
//
//
///////////////////////////////////////////////////////////////////////


#ifndef _H_GVF
#define _H_GVF

#include "GVFMatrix.h"

#include <map>
#include <tr1/random>
#include <iostream>

#define BOOSTLIB 0
#define OPTIMISD 0
#define VDSPOPTM 0
#define GESTLEARNT 8

//#define USE_MAXMSP

#if BOOSTLIB
#include <boost/random.hpp>
#endif

#ifndef USE_MAXMSP
static void post(const char * format, ...){
    va_list args;
    printf(format, args);
}
#endif

using namespace std;

// Recognizes gesture and tracks the variations. A set of gesture templates
// must be recorded. Then, at each new observation, the algorithm estimates
// which gesture is performed and adapts a set fo features (gesture variations)
// that are used as invariants for the recognition and gives continuous output
// parameters (e.g. for interaction)
//
// typical use:
//   GVF *myGVF = new GVF(...)
//   myGVF->addTemplate();
//   myGVF->fillTemplate();
//   myGVF->addTemplate();
//   myGVF->fillTemplate();
//   ...
//   myGVF->infer();
//   myGVF->getEstimatedStatus();

class GVF{
	
public:
	
    enum GVFState{
        STATE_CLEAR = 0,
        STATE_LEARNING,
        STATE_FOLLOWING
    };
    
    typedef struct{
        int inputDimensions;
        int numberParticles;
        float tolerance;
        int resamplingThreshold;
        float distribution;
        bool translate;
        bool allowSegmentation;
    } GVFParameters;
    
    typedef struct{
        float phaseVariance;
        float speedVariance;
        float scaleVariance;
        float rotationVariance;
    } GVFVarianceCoefficents;
    
    typedef struct{
        float phaseInitialSpreading;
        float speedInitialSpreading;
        vector<float> scaleInitialSpreading;
        vector<float> rotationInitialSpreading;
    } GVFInitialSpreadingParameters;
    
    
	// constructor of the gvf instance
	GVF(); // use defualt parameters
	GVF(GVFParameters parameters, GVFVarianceCoefficents coefficents);
    
	// destructor
    ~GVF();
	
    void setup();
    void setup(GVFParameters parameters, GVFVarianceCoefficents coefficents);

    // kind of learning (now only to set up dimensions)
    void learn();
    
	// add template to the vocabulary
	void addTemplate();
	void addTemplate(vector<float> & data);
    
	// fill template given by id with data vector
	void fillTemplate(int id, vector<float> & data);
    
    // clear template given by id
    void clearTemplate(int id);
    
	// clear the templates
	void clear();
    
	// spread particles
	void spreadParticles();         // use default parameter values
	//void spreadParticles(vector<float> & means, vector<float> & ranges);
    void spreadParticles(GVFInitialSpreadingParameters _spreadingParameters);

    	// spread particles
    void restart();
    
	// inference
    void particleFilter(vector<float> & obs);
	
    // resample particles according to the proba distrib given by the weights
    void resampleAccordingToWeights();
	
    // makes the inference by calling particleFilteringOptim
	void infer(vector<float> & data);   // rename to update?
	void updateEstimatedStatus();       // should be private?
    
    //////////////////////////
    // Gestures & Templates //
	//////////////////////////
    
    // GESTURE PROBABILITIES + POSITIONS
    
    int getMostProbableGestureIndex();
    vector<float> getMostProbableGestureStatus();
    float getMostProbableProbability(); // this is horrible - maybe probability should be at index 0 OR we should use a struct rather than a vector??
    vector< vector<float> > getEstimatedStatus();
    
    vector<float> getGestureProbabilities();
    vector< vector<float> > getParticlesPositions();
    
    // TEMPLATES
    
    int getNumberOfTemplates();
    vector< vector<float> >& getTemplateByIndex(int index);
    int getLengthOfTemplateByIndex(int index);
    

    
    ///////////////////////
    // Getters & Setters //
	///////////////////////
    
    // STATES
    
    GVFState getState();
    string getStateAsString();
    void setState(GVFState _state);
    
    // PARAMETERS
    
    void setParameters(GVFParameters parameters);
    GVFParameters getParameters();
    
    void setNumberOfParticles(int numberOfParticles);
    int getNumberOfParticles();
    
	void setResamplingThreshold(int resamplingThreshold);
    int getResamplingThreshold();
    
    void setTolerance(float tolerance);
    float getTolerance();
    
    void setDistribution(float distribution);
    float getDistribution();
    
    // VARIANCE COEFFICENTS
    
    void setVarianceCoefficents(GVFVarianceCoefficents coefficients);
    GVFVarianceCoefficents getVarianceCoefficents();
    
    void setPhaseVariance(float phaseVariance);
    float getPhaseVariance();
    
    void setSpeedVariance(float speedVariance);
    float getSpeedVariance();
    
    void setScaleVariance(float scaleVariance);
    float getScaleVariance();
    
    void setRotationVariance(float rotationVariance);
    float getRotationVariance();
    
    
    // SPREADING PARAMETERS
    
    void setSpreadingParameters(GVFInitialSpreadingParameters spreadingParameters);
    GVFInitialSpreadingParameters getSpreadingParameters();
    
    void setPhaseInitialSpreading(float phaseInitialSpreading);
    float getPhaseInitialSpreading();
    
    void setSpeedInitialSpreading(float speedInitialSpreading);
    float getSpeedInitialSpreading();
    
    void setScaleInitialSpreading(float speedInitialSpreading);
    vector<float> getScaleInitialSpreading();
    
    void setRotationInitialSpreading(float rotationInitialSpreading);
    vector<float> getRotationInitialSpreading();
    
    
    
    
    // Absolute weights and Offsets (for segmentation)
    
    vector<float> getAbsoluteProbabilities();
    vector<float>* getOffsets();
    
    // MATHS
    
    float   getObservationStandardDeviation();
    
    vector< vector<float> > getX();
	vector<int>    getG();
	vector<float>  getW();
    
    // UTILITIES
    
	void saveTemplates(string filename);
    void loadTemplates(string filename);
	
private:
    
    // private variables
    
    GVFParameters parameters;
    GVFVarianceCoefficents coefficents;
    GVFInitialSpreadingParameters spreadingParameters;
    
	float   nu;                 // degree of freedom for the t-distribution; if 0, use a gaussian
	float   sp, sv, sr, ss;     // sigma values (actually, their square root)
	int     resamplingThreshold;// resampling threshol
    int     ns;
	int     pdim;               // number of state dimension
	int     numTemplates;       // number of learned gestures (starts at 0)
    int     inputDim;           // Dimension of the input data
    
    int mostProbableIndex;                      // cached most probable index
    vector<float> mostProbableStatus;           // cached most probable status [phase, speed, scale[, rotation], probability]
    vector< vector<float> > S;                  // cached estimated status for all templates
	vector< vector<float> > X;                  // each row is a particle
	vector<int>             g;                  // gesture index for each particle [g is ns x 1]
	vector<float>           w;                  // weight of each particle [w is ns x 1]
    vector< vector<float> > offS;               // translation offset
	vector<float>           featVariances;      // vector of variances
	vector<float>           means;              // vector of means for particles initial spreading
	vector<float>           ranges;             // vector of ranges around the means for particles initial spreading
    
    // gesture 'history'
	map<int, vector< vector<float> > > R_single;   // gesture references (1 example)
    map<int, vector<float> > R_initial;  // gesture initial data
    vector<float> O_initial;  // observed initial data
    vector<vector<float> > O_initial_particle;  // observed initial data for each particle
    vector<int>    gestureLengths;             // length of each reference gesture
    vector<vector<float> >  observationBuffer;
    
    //in order to output particles
    vector< vector<float> > particlesPositions;
    
    GVFState state;                          // store current state of the gesture follower
    
    vector< vector<float> > EmptyTemplate;      // dummy empty template for passing as ref
	
    // random number generator
#if BOOSTLIB
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<float> > *rndnorm(rng, normdist);
	boost::mt19937 rng;
	boost::normal_distribution<float> normdist;
#else
    tr1::mt19937 rng;
    tr1::normal_distribution<float> *normdist;
    tr1::uniform_real<float> *unifdist;
	tr1::variate_generator<tr1::mt19937, tr1::normal_distribution<float> > *rndnorm;//(rng, *normdist);
#endif
    
	// Segmentation variables
    bool allowSegmentation;
	vector<float> abs_weights;
	double probThresh;
    double probThreshMin;
    int currentGest;
    bool compa;
    float old_max;
    vector<float> meansCopy;
    vector<float> rangesCopy;
    vector<float> origin;
    vector<float> *offset;
    bool new_gest;
    
    // private functions
    void initweights();                         // initialize weights
    
    // functions
    int frameCounter;
    vector<float> computeTemplateInstantaneousSpeed(int instantIndex, int smoothingWindowSize);
    float computeObservationInstantaneousSpeed();
    
};

#endif