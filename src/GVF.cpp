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

#include "GVF.h"
#include "maxcpp6.h"
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tr1/memory>
#include <unistd.h>

using namespace std;


vector<vector<float> > return_RotationMatrix_3d(float phi, float theta, float psi);


////////////////////////////////////////////////////////////////
//
// CONSTRUCTOR, DESTRUCTOR & SETUP
//
////////////////////////////////////////////////////////////////

// Constructor of the class GVF
// This creates an object that is able to learn gesture template,
// to recognize in realtime the live gesture and to estimate the
// live gesture variations according to the template (e.g. scale, speed, ...)
//
// typical use
//   GVF *myGVF;
//   myGVF = new GVF(NS, Sigs, Icov, ResThresh, Nu)
//
// ns is the number of particles
// Sigs is the variance for each varying feature that has to be estimated (speed, scale, twisting angle)
//
// Note about the current implementation: it involves geometric features: phase, speed, scale, angle of rotation
//    that are meant to be used for 2-dimensional input shapes. For general N-dimensional input, the class will
//    only consider phase, speed, scaling
//GVF::GVF(int ns, VectorXf sigs, float icov, int resThresh, float nu)

//--------------------------------------------------------------
GVF::GVF(){
    // nothing?
}

//--------------------------------------------------------------
GVF::GVF(GVFParameters _parameters, GVFVarianceCoefficents _coefficents){
    setup(parameters, _coefficents);
}

//--------------------------------------------------------------
void GVF::setup(){
    
    // use defualt parameters
    
    GVFParameters defaultParameters;
    
    defaultParameters.inputDimensions = 2;
    defaultParameters.numberParticles = 2000;
    defaultParameters.tolerance = 1.7f;
    defaultParameters.resamplingThreshold = 500;
    defaultParameters.distribution = 0.1f;
    defaultParameters.translate = true;
    defaultParameters.allowSegmentation = true;
    
    GVFVarianceCoefficents defaultCoefficients;
    
    defaultCoefficients.phaseVariance = 0.0005;
    defaultCoefficients.speedVariance = 0.001;
    defaultCoefficients.scaleVariance = 0.0001;
    defaultCoefficients.rotationVariance = 0.0;
    
    setup(defaultParameters, defaultCoefficients);
}

//--------------------------------------------------------------
void GVF::setup(GVFParameters _parameters, GVFVarianceCoefficents _coefficents){
    
    
    clear(); // just in case
    
    
    // Set parameters:
    //    input dimensions
    //    num of particles
    //    tolerance
    //    resampling threshold
    //    distribution (nu value of the Student's T distribution [default=0])
    parameters  = _parameters;
    
    
    // Set variances for variation tracking:
    //    phase
    //    speed
    //    scale
    //    rotation
    coefficents = _coefficents;
    
    parameters.inputDimensions = -1; // input dimensions not defined yet
    ns = parameters.numberParticles;
    
    /*
     if(inputDim > 2 && coefficents.rotationVariance != 0.0){
     cout << "Warning rotation variance will not be considered for more than 2 input dimensions!" << endl;
     coefficents.rotationVariance = 0.0f;
     }
     */
    
    numTemplates=-1;                        // Set num. of learned gesture to -1
    gestureLengths = vector<int>();         // Vector of gesture lengths
    
    
#if !BOOSTLIB
    normdist = new std::tr1::normal_distribution<float>();
    unifdist = new std::tr1::uniform_real<float>();
    rndnorm  = new std::tr1::variate_generator<std::tr1::mt19937, std::tr1::normal_distribution<float> >(rng, *normdist);
#endif
    
    // Variables used for segmentation -- experimental (research in progress)
    // TODO(baptiste)
    abs_weights = vector<float>();      // absolute weights used for segmentation
    O_initial_particle = vector<vector< float> > (parameters.numberParticles);
    
    
}

//----------------------------------------------
// kind of learning here
//
void GVF::learn(){
    
    if (numTemplates >= 0){
        
        post("dim %i ", R_single[0][0].size());

        featVariances.clear();
        
        parameters.inputDimensions = R_single[0][0].size();
        
        int scaleDim;
        int rotationDim;
        
        if (parameters.inputDimensions == 2){
            
            //post(" dim 2 -> pdim 4");
            
            // state space dimension = 4
            // phase, speed, scale, rotation
            
            scaleDim = 1;
            rotationDim = 1;
            
            pdim = 2 + scaleDim + rotationDim;
            
            featVariances = vector<float> (pdim);
            
            featVariances[0]=sqrt(coefficents.phaseVariance);
            featVariances[1]=sqrt(coefficents.speedVariance);
//            featVariances[2]=sqrt(coefficents.scaleVariance);
//            featVariances[3]=sqrt(coefficents.rotationVariance);
            for (int k=0;k<scaleDim;k++) featVariances[2+k]=sqrt(coefficents.scaleVariance);
            for (int k=0;k<rotationDim;k++) featVariances[2+scaleDim+k]=sqrt(coefficents.rotationVariance);

            
            
            // Spreading parameters: initial value for each variation (e.g. speed start at 1.0 [i.e. original speed])
            spreadingParameters.phaseInitialSpreading = 0.15;
            spreadingParameters.speedInitialSpreading = 1.0;
//            spreadingParameters.scaleInitialSpreading.push_back(1.0f);
//            spreadingParameters.rotationInitialSpreading.push_back(0.0f);
            spreadingParameters.scaleInitialSpreading = vector<float> (scaleDim);
            spreadingParameters.rotationInitialSpreading = vector<float> (rotationDim);
            for (int k=0;k<scaleDim;k++) spreadingParameters.scaleInitialSpreading[k]=1.0f;
            for (int k=0;k<rotationDim;k++) spreadingParameters.rotationInitialSpreading[k]=0.0f;
        
            
        }
        else if (parameters.inputDimensions == 3){
            
            // state space dimension = 8
            // phase, speed, scale (1d), rotation (3d)
            
            scaleDim = 3;
            rotationDim = parameters.inputDimensions;
            
            pdim = 2 + scaleDim + rotationDim;
            
            featVariances = vector<float> (pdim);
            
            featVariances[0]=sqrt(coefficents.phaseVariance);
            featVariances[1]=sqrt(coefficents.speedVariance);
            for (int k=0;k<scaleDim;k++) featVariances[2+k]=sqrt(coefficents.scaleVariance);
            for (int k=0;k<rotationDim;k++) featVariances[2+scaleDim+k]=sqrt(coefficents.rotationVariance);
            
            
            spreadingParameters.phaseInitialSpreading = 0.15;
            spreadingParameters.speedInitialSpreading = 1.0;
            spreadingParameters.scaleInitialSpreading = vector<float> (scaleDim);
            spreadingParameters.rotationInitialSpreading = vector<float> (rotationDim);
            for (int k=0;k<scaleDim;k++) spreadingParameters.scaleInitialSpreading[k]=1.0f;
            for (int k=0;k<rotationDim;k++) spreadingParameters.rotationInitialSpreading[k]=0.0f;
            
        }
        else {
            
            scaleDim = 1;
            rotationDim = 0;
            
            pdim = 2 + scaleDim + rotationDim;
            
            featVariances = vector<float> (pdim);
            
            featVariances[0]=sqrt(coefficents.phaseVariance);
            featVariances[1]=sqrt(coefficents.speedVariance);
            featVariances[2]=sqrt(coefficents.scaleVariance);

            // Spreading parameters: initial value for each variation (e.g. speed start at 1.0 [i.e. original speed])
            spreadingParameters.phaseInitialSpreading = 0.15;
            spreadingParameters.speedInitialSpreading = 1.0;
            spreadingParameters.scaleInitialSpreading = vector<float> (scaleDim);
//            spreadingParameters.rotationInitialSpreading = vector<float> (rotationDim);
            for (int k=0;k<scaleDim;k++) spreadingParameters.scaleInitialSpreading[k]=1.0f;
//            for (int k=0;k<rotationDim;k++) spreadingParameters.rotationInitialSpreading[k]=0.0f;
        }
        
        
        
        
        initMat(X, parameters.numberParticles, pdim);           // Matrix of NS particles
        initVec(g, parameters.numberParticles);                 // Vector of gesture class
        initVec(w, parameters.numberParticles);                 // Weights
    }
}




// Destructor of the class
GVF::~GVF(){
#if !BOOSTLIB
    if(normdist != NULL)
        delete (normdist);
    if(unifdist != NULL)
        delete (unifdist);
#endif
    
    // should we free here other variables such X, ...??
    //TODO(baptiste)
    
    clear(); // not really necessary but it's polite ;)
    
}

////////////////////////////////////////////////////////////////
//
// ADD & FILL TEMPLATES FOR GESTURES
//
////////////////////////////////////////////////////////////////

//--------------------------------------------------------------
// Add a template into the vocabulary. This method does not add the data but allocate
// the memory and increases the number of learned gesture
void GVF::addTemplate(){
	numTemplates++;                                         // increment the num. of learned gesture
	R_single[numTemplates]  = vector< vector<float> >();    // allocate the memory for the gesture's data
    R_initial[numTemplates] = vector<float> ();             // allocate memory for initial points (offsets)
    gestureLengths.push_back(0);                            // add an element (0) in the gesture lengths table
    abs_weights.resize(numTemplates+1);
}

void GVF::addTemplate(vector<float> & data){
	addTemplate();
    fillTemplate(getNumberOfTemplates(), data);
}

//--------------------------------------------------------------
// Fill the template given by the integer 'id' by appending the current data vector 'data'
// This example fills the template 1 with the live gesture data (stored in liveGesture)
// for (int k=0; k<SizeLiveGesture; k++)
//    myGVF->fillTemplate(1, liveGesture[k]);
void GVF::fillTemplate(int id, vector<float> & data){
	
    //post("id = %i || numTemplates = %i", id, numTemplates);
    
    if (id <= numTemplates){
        
        if( parameters.translate ){
            
            // store initial point
            if(R_single[id].size() == 0) R_initial[id] = data;
            
            // 'center' data
            for(int i = 0; i < data.size(); i++){
                data[i] -= R_initial[id][i];
            }
        }
        
		R_single[id].push_back(data);
		gestureLengths[id] = gestureLengths[id]+1;
	}
}

//--------------------------------------------------------------
// clear template given by id
void GVF::clearTemplate(int id){
    if (id <= numTemplates){
        R_single[id] = vector< vector<float> >();      // allocate the memory for the gesture's data
        gestureLengths[id] = 0;                // add an element (0) in the gesture lengths table
    }
}

//--------------------------------------------------------------
// Clear the internal data (templates)
void GVF::clear(){

    state = STATE_CLEAR;
	R_single.clear();
    R_initial.clear();
    O_initial.clear();
    O_initial_particle.clear();
	gestureLengths.clear();
    mostProbableIndex = -1;
    mostProbableStatus.clear();
	numTemplates=-1;
}

////////////////////////////////////////////////////////////////
//
// STATE SET & GET - eg., clear, learn, follow
//
////////////////////////////////////////////////////////////////

//--------------------------------------------------------------
void GVF::setState(GVFState _state){
    switch (_state) {
        case STATE_CLEAR:
            clear();
            break;
        case STATE_LEARNING:
            state = _state;
            break;
        case STATE_FOLLOWING:
            spreadParticles(); // TODO provide setter for mean and range on init
            state = _state;
            break;
    }
}

//--------------------------------------------------------------
GVF::GVFState GVF::getState(){
    return state;
}

//--------------------------------------------------------------
string GVF::getStateAsString(){
    switch (state) {
        case STATE_CLEAR:
            return "STATE_CLEAR";
            break;
        case STATE_LEARNING:
            return "STATE_LEARNING";
            break;
        case STATE_FOLLOWING:
            return "STATE_FOLLOWING";
            break;
    }
}

////////////////////////////////////////////////////////////////
//
// CORE FUNCTIONS & MATH
//
////////////////////////////////////////////////////////////////

//--------------------------------------------------------------
void GVF::spreadParticles(){
    
    // use default means and ranges - taken from gvfhandler
    // BAPTISTE: what are these magic numbers ? ;)
    
    spreadParticles(spreadingParameters);
    
}


//--------------------------------------------------------------
// Spread the particles by sampling values from intervals given my their means and ranges.
// Note that the current implemented distribution for sampling the particles is the uniform distribution
//void GVF::spreadParticles(vector<float> & means, vector<float> & ranges){
void GVF::spreadParticles(GVFInitialSpreadingParameters _spreadingParameters){
    
    
    // USE BOOST FOR UNIFORM DISTRIBUTION!!!!
    // deprecated class should use uniform_real_distribution
#if BOOSTLIB
	boost::uniform_real<float> ur(0,1);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > rnduni(rng, ur);
#else
    std::tr1::uniform_real<float> ur(0,1);
    std::tr1::variate_generator<std::tr1::mt19937, std::tr1::uniform_real<float> > rnduni(rng, ur);
#endif
	
	unsigned int ngestures = numTemplates+1;
    
    float spreadRangePhase = 0.3;
    float spreadRange = 0.1;
    int scalingCoefficients  = _spreadingParameters.scaleInitialSpreading.size();
    int numberRotationAngles = _spreadingParameters.rotationInitialSpreading.size();
    
    post("scale %i rotation %i", scalingCoefficients, numberRotationAngles);
    
    
    // Spread particles using a uniform distribution
	//for(int i = 0; i < pdim; i++)
    for(int n = 0; n < ns; n++){
        X[n][0] = (rnduni() - 0.5) * spreadRangePhase + _spreadingParameters.phaseInitialSpreading;
        X[n][1] = (rnduni() - 0.5) * spreadRange + _spreadingParameters.speedInitialSpreading;
        for (int nn=0; nn<scalingCoefficients; nn++)
            X[n][2+nn] = (rnduni() - 0.5) * spreadRange
            + _spreadingParameters.scaleInitialSpreading[nn];
        for (int nn=0; nn<numberRotationAngles; nn++)
            X[n][2+scalingCoefficients+nn] = (rnduni() - 0.5) * 0.0 //spreadRange/2
            + _spreadingParameters.rotationInitialSpreading[nn];
    }
    
    
    // Weights are also uniformly spread
    initweights();
    //    logW.setConstant(0.0);
	
    // Spread uniformly the gesture class among the particles
	for(int n = 0; n < ns; n++){
		g[n] = n % ngestures;
        //  post("%i, %i", ngestures, n % ngestures);
        
        // offsets are set to 0
        //for (int k=0; k<inputDim; k++)
        //    offS[n][k]=0.0;
    }
    
}

//--------------------------------------------------------------
// Restart functions spread the particles according to
// means and ranges
void GVF::restart(){
    
    //spreadParticles(means, ranges);
    spreadParticles(spreadingParameters);
    O_initial.clear();
    O_initial_particle.clear();
    O_initial_particle = vector<vector< float> > (parameters.numberParticles);
    
}

//--------------------------------------------------------------
// Initialialize the weights of the particles. The initial values of the weights is a
// unifrom weight over the particles
void GVF::initweights(){
    for (int k=0; k<ns; k++)
        w[k]=1.0/ns;
}

//--------------------------------------------------------------
float distance_weightedEuclidean(vector<float> x, vector<float> y, vector<float> w){
    int count = x.size();
    // the size must be > 0
    if (count<=0)
        return 0;
    
    float dist=0;
    for(int k=0;k<count;k++){
        dist+=w[k]*pow((x[k]-y[k]),2);
        //post(" k=%i | %f %f %f",k,x[k],y[k],w[k]);
    }
    return dist;
}

//--------------------------------------------------------------
// Performs the inference based on a given new observation. This is the core algorithm: does
// one step of inference using particle filtering. It is the optimized version of the
// function ParticleFilter(). Note that the inference is possible only if some templates
// have been learned beforehand
//
// The inferring values are the weights of each particle that represents a possible gesture,
// plus a possible configuration of the features (value of speec, scale,...)
void GVF::particleFilter(vector<float> & obs){
    
    
    float tolerance = parameters.tolerance;
    int inputDim = parameters.inputDimensions;
    
    if( parameters.translate && !parameters.allowSegmentation ){
        
        // then it's a new gesture observation - cleared in spreadParticles -> store initial obs data
        if(O_initial.size() == 0) O_initial = obs;
        
        // 'center' data
        for(int i = 0; i < obs.size(); i++)
            obs[i] -= O_initial[i];
        
    }
    
    
    /*
     // fill buffer of observations (to compute derivatives)
     for (int k=0; k<observationBuffer.size()-1; k++)
     observationBuffer[k]=observationBuffer[k+1];
     observationBuffer[observationBuffer.size()-1] = obs;
     
     //post("framecounter=%i",frameCounter);
     if (frameCounter<4) frameCounter++;
     
     //for (int k=0; k<observationBuffer.size(); k++)
     //    post("obs: %f %f %f", observationBuffer[k][0], observationBuffer[k][1], observationBuffer[k][2]);
     */
    
    
#if BOOSTLIB
	boost::uniform_real<float> ur(0,1);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > rnduni(rng, ur);
#else
    std::tr1::uniform_real<float> ur(0,1);
    std::tr1::variate_generator<std::tr1::mt19937, std::tr1::uniform_real<float> > rnduni(rng, ur);
#endif
    
    
    // zero abs weights
    for(int i = 0 ; i < getNumberOfTemplates(); i++){
        abs_weights[i] = 0.0;
    }
    
    // clear any previous information about the particles' positions
    // (this is used for possible visualization but not in the inference)
    particlesPositions.clear();
    
    float sumw=0.0;
    
    
    // Parameters for re-spreading particles in case they fall above phase=1 or below phase=0
    //   some magic numbers...
    float spreadRangePhase = 0.3;
    float spreadRangeSpeed = 0.1;
    float spreadRangeScale = 0.1;
    float spreadRangeRotation = 0.0;
    //   some sizes
    int scalingCoefficients  = spreadingParameters.scaleInitialSpreading.size();
    int numberRotationAngles = spreadingParameters.rotationInitialSpreading.size();
    
    
    // MAIN LOOP: same process for EACH particle (row n in X)
    for(int n = ns-1; n >= 0; --n)
    {
        
        // Move the particle
        // Position respects a first order dynamic: p = p + v/L
		//X(n,0) = X(n,0) + (*rndnorm)() * featVariances(0) + X(n,1)/gestureLengths[g(n)];
        X[n][0] += (*rndnorm)() * featVariances[0] + X[n][1]/gestureLengths[g[n]];
        
        
		// Move the other state elements according a gaussian noise
        // featVariances vector of variances
        for(int l = pdim-1; l>=1 ; --l){
			X[n][l] += (*rndnorm)() * featVariances[l];
        }
        
        
        
		if(X[n][0] < 0.0 || X[n][0] > 1.0)  // if the particle falls below phase = 0 (before the gesture's starting point)
        {                                   // or above phase = 1 (after the gesture's ending point)
            // we re-initialize its values and spread it at the beginning of a random gesture
            
            
            //if (abs(X[n][1])<0.5) {     // criterion should be on the decrease of the speed and not the speed
            //    w[n] = 0.0f; // heuristic: if static don't re-spread the particles
            //}
            //else{
            
            if ( parameters.allowSegmentation ) {
                
                // re-initialize particle's values according to initial spreading parameters
                X[n][0] = (rnduni()-0.5) * spreadRangePhase + spreadingParameters.phaseInitialSpreading;
                X[n][1] = (rnduni()-0.5) * spreadRangeSpeed + spreadingParameters.speedInitialSpreading;
                for (int nn=0; nn<scalingCoefficients; nn++)
                    X[n][2+nn] = (rnduni()-0.5) * spreadRangeScale + spreadingParameters.scaleInitialSpreading[nn];
                for (int nn=0; nn<numberRotationAngles; nn++)
                    X[n][2+scalingCoefficients+nn] = (rnduni()-0.5) * spreadRangeRotation //spreadRange/2
                    + spreadingParameters.rotationInitialSpreading[nn];
                
                // assign particle to a random gesture
                g[n] = n % gestureLengths.size();
                
                
                // assign a new weight to that particle (heuristics!)
                w[n] = 1.0/(ns*ns);
                // w[n]=0;
                
                //            post("respreading particle %i onto gesture %i w/ %.4f -- %.4f, %.4f, %.4f, %.4f ...",
                //                 n, g[n], w[n], X[n][0], X[n][1], X[n][2], X[n][3]);
                
                // TODO-update offset??
                O_initial_particle[n].clear();
                //}
            } else {
                w[n] = 0.0;
            }
            
        }
        
        else{       // ...otherwise we propagate the particle's values and update its weight
            
            vector<float> x_n = X[n];
            vector<float> obs_tmp = obs;
            
            if (allowSegmentation){
                
                if (parameters.translate) {
                    if (O_initial_particle[n].size()==0) {
                        O_initial_particle[n] = vector<float> (obs.size());
                        setVec(O_initial_particle[n], obs);
                    }
                }
                else {
                    if (O_initial_particle[n].size()==0) {
                        O_initial_particle[n] = vector<float> (obs.size());
                        setVec(O_initial_particle[n], (float)0.0);
                    }
                }
                
                for(int i = 0; i < obs.size(); i++)
                    obs_tmp[i] -= O_initial_particle[n][i];
            }
            else
                obs_tmp = obs;
            
            
            
            
            // gesture index for the particle
            int pgi = g[n];
            
            
            // given the phase between 0 and 1 (first value of the particle x),
            // return the index of the corresponding gesture, given by g(n)
            int frameindex = min((int)(gestureLengths[pgi]-1),(int)(floor(x_n[0] * gestureLengths[pgi])));
            
            
            // given the index, return the gesture template value at this index
            vector<float> vref(inputDim);
            setVec(vref,R_single[pgi][frameindex]);
            
            vector<float> vobs(inputDim);
            setVec(vobs,obs);
            
            
            
            
            
            // If incoming data is 2-dimensional: we estimate phase, speed, scale, angle
            if (inputDim == 2){
                
                // offset!
                //for (int k=0;k<inputDim;k++)
                //    vobs[k] -= offS[n][k];
                
                // sca1ing
                for (int k=0;k<inputDim;k++)
                    vref[k] *= x_n[2];
                
                // rotation
                float alpha = x_n[3];
                float tmp0=vref[0]; float tmp1=vref[1];
                vref[0] = cos(alpha)*tmp0 - sin(alpha)*tmp1;
                vref[1] = sin(alpha)*tmp0 + cos(alpha)*tmp1;
                
                // put the positions into vector
                // [used for visualization]
                std::vector<float> temp;
                temp.push_back(vref[0]);
                temp.push_back(vref[1]);
                particlesPositions.push_back(temp);
                
            }
            // If incoming data is 3-dimensional
            else if (inputDim == 3){
                
                // Scale template sample according to the estimated scaling coefficients
                int numberScaleCoefficients = spreadingParameters.scaleInitialSpreading.size();
                for (int k=0;k<numberScaleCoefficients;k++)
                    vref[k] *= x_n[2+k];
                
                // Rotate template sample according to the estimated angles of rotations (3d)
                vector<vector< float> > RotMatrix = return_RotationMatrix_3d(x_n[2+numberScaleCoefficients],
                                                                             x_n[2+numberScaleCoefficients+1],
                                                                             x_n[2+numberScaleCoefficients+2]);
                vref = multiplyMat(RotMatrix, vref);
                
                // put the positions into vector
                // [used for visualization]
                std::vector<float> temp;
                for (int ndi=0; ndi<inputDim; ndi++)
                    temp.push_back(vref[ndi]);
                particlesPositions.push_back(temp);
                
            }
            else {
                
                // sca1ing
                for (int k=0;k<inputDim;k++)
                    vref[k] *= x_n[2];
                
                // put the positions into vector
                // [used for visualization]
                std::vector<float> temp;
                for (int ndi=0; ndi<inputDim; ndi++)
                    temp.push_back(vref[ndi]);
                particlesPositions.push_back(temp);
                
            }
            
            // compute distance between estimation given the current particle values
            // and the incoming observation
            
            // define weights here on the dimension if needed
            vector<float> dimWeights(inputDim);
            for(int k=0;k<inputDim;k++) dimWeights[k]=1.0/inputDim;
            
            
            // observation likelihood and update weights
            //            float dist = sqrt( distance_weightedEuclidean(vref,obs,dimWeights) * 1/(tolerance*tolerance) );
            
            float dist = sqrt( distance_weightedEuclidean(vref,obs_tmp,dimWeights) * 1/(tolerance*tolerance) );
            
            
            
            if(parameters.distribution == 0.0f)      // Gaussian distribution
            {
                w[n]   *= exp(-dist);
                abs_weights[g[n]] += exp(-dist);
                
            }
            else                // Student's distribution
            {
                w[n]   *= pow(dist/nu + 1,-nu/2-1);
                abs_weights[g[n]] += exp(-dist);
            }
        }
        sumw+=w[n];
    }
    
    // normalize weights and compute criterion for degeneracy
    float dotProdw=0.0;
    for (int k=0;k<ns;k++){
        w[k]/=sumw; dotProdw+=w[k]*w[k];
    }
    float neff = 1./dotProdw;
    
    
    //    std::string directory = "/Users/caramiaux/Research/PatchsMax/GVFs/Leap/particles.txt";
    //    std::ofstream file_write(directory.c_str());
    //    for(int j=0; j<w.size(); j++)
    //    {
    //        file_write << j << " " << g[j] << " " << X[j][0] << " " << w[j] << endl;
    //    }
    //    file_write.close();
    
    
    
    // Try segmentation from here...
    
    //     post("");
    //     if(maxSoFar > probThreshMin && !compa)
    //     {
    //     old_max = maxSoFar;
    //     compa = true;
    //     }
    //
    //     if(maxSoFar < probThresh && compa)
    //     {
    //     spreadParticles();
    //     new_gest = true;
    //     //     (*offset)[0] = obs[0];
    //     //     (*offset)[1] = obs[1];
    //     O_initial.clear();
    //     compa = false;
    //     }
    
    // ... to here.
    
    
    // avoid degeneracy (no particles active, i.e. weights = 0) by resampling
    // around the active particles
	if(neff<parameters.resamplingThreshold)
    {
        //cout << "Resampling" << endl;
        resampleAccordingToWeights();
        initweights();
    }
    
    
}

//--------------------------------------------------------------
// Resampling function. The function resamples the particles based on the weights.
// Particles with negligeable weights will be respread near the particles with non-
// neglieable weigths (which means the most likely estimation).
// This steps is important to avoid degeneracy problem
void GVF::resampleAccordingToWeights()
{
#if BOOSTLIB
    boost::uniform_real<float> ur(0,1);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > rnduni(rng, ur);
#else
    std::tr1::uniform_real<float> ur(0,1);
    std::tr1::variate_generator<std::tr1::mt19937, std::tr1::uniform_real<float> > rnduni(rng, ur);
#endif
    
    
    //    post("Resampling!");
    
    
    vector< vector<float> > oldX;
    setMat(oldX,X);
    vector<int> oldG;
    setVec(oldG, g);
    vector<float> c(ns);
    
    c[0] = 0;
    for(int i = 1; i < ns; i++)
        c[i] = c[i-1] + w[i];
    int i = 0;
    float u0 = rnduni()/ns;
    int free_pool = 0;
    for (int j = 0; j < ns; j++)
    {
        float uj = u0 + (j + 0.) / ns;
        
        while (uj > c[i] && i < ns - 1){
            i++;
        }
        
        if(j < ns - free_pool){
            for (int kk=0;kk<X[0].size();kk++)
                X[j][kk] = oldX[i][kk];
            g[j] = oldG[i];
            //            logW(j) = oldLogW(i);
        }
    }
    
}

//--------------------------------------------------------------
// Step function is the function called outside for inference. It
// has been originally created to be able to infer on a new observation or
// a set of observation.
void GVF::infer(vector<float> & vect){
    particleFilter(vect);
    updateEstimatedStatus();
}

void GVF::updateEstimatedStatus(){
    // get the number of gestures in the vocabulary
	unsigned int ngestures = numTemplates+1;
	//cout << "getEstimatedStatus():: ngestures= "<< numTemplates+1<< endl;
    
    //    vector< vector<float> > es;
    setMat(S, 0.0f, ngestures, pdim+1);   // rows are gestures, cols are features + probabilities
	//printMatf(es);
    
	// compute the estimated features by computing the expected values
    // sum ( feature values * weights)
	for(int n = 0; n < ns; n++){
        int gi = g[n];
        for(int m=0; m<pdim; m++){
            S[gi][m] += X[n][m] * w[n];
        }
		S[gi][pdim] += w[n];
    }
	
    // calculate most probable index during scaling...
    float maxProbability = 0.0f;
    mostProbableIndex = -1;
    
	for(int gi = 0; gi < ngestures; gi++){
        for(int m=0; m<pdim; m++){
            S[gi][m] /= S[gi][pdim];
        }
        if(S[gi][pdim] > maxProbability){
            maxProbability = S[gi][pdim];
            mostProbableIndex = gi;
        }
		//es.block(gi,0,1,pdim) /= es(gi,pdim);
	}
    
    if(mostProbableIndex > -1) mostProbableStatus = S[mostProbableIndex];
    
}

////////////////////////////////////////////////////////////////
//
// PROBABILITY AND TEMPLATE ACCESS
//
////////////////////////////////////////////////////////////////

// GESTURE PROBABILITIES + POSITIONS

//--------------------------------------------------------------
// Returns the index of the currently recognized gesture
// NOW CACHED DURING 'infer' see updateEstimatedStatus()
int GVF::getMostProbableGestureIndex(){
    //    vector< vector< float> > M = getEstimatedStatus();
    //    float maxProbability = 0.0f;
    //    int indexMostProb = -1; // IMPORTANT: users need to check for negative index!!!
    //    for (int k=0; k<M.size(); k++){
    //        cout << M[k][M[0].size() - 1] << " > " << maxProbability << endl;
    //        if (M[k][M[0].size() - 1] > maxProbability){
    //            maxProbability = M[k][M[0].size() - 1];
    //            indexMostProb = k;
    //        }
    //    }
    //    return indexMostProb;
    return mostProbableIndex;
}

//--------------------------------------------------------------
// Returns the index of the currently recognized gesture
vector<float> GVF::getMostProbableGestureStatus(){
    return mostProbableStatus;
}

//--------------------------------------------------------------
// Returns the probability of the currently recognized gesture
float GVF::getMostProbableProbability(){
    return mostProbableStatus[mostProbableStatus.size() - 1];
}

//--------------------------------------------------------------
// Returns the estimates features. It calls status to refer to the status of the state
// space which comprises the features to be adapted. If features are phase, speed, scale and angle,
// the function will return these estimateed features for each gesture, plus their probabilities.
// The returned matrix is nxm
//   rows correspond to the gestures in the vocabulary
//   cols correspond to the features (the last column is the [conditionnal] probability of each gesture)
// The output matrix is an Eigen matrix
// NOW CACHED DURING 'infer' see updateEstimatedStatus()
vector< vector<float> > GVF::getEstimatedStatus(){
    return S;
}

//--------------------------------------------------------------
// Returns the probabilities of each gesture. This probability is conditionnal
// because it depends on the other gestures in the vocabulary:
// probability to be in gesture A knowing that we have gesture A, B, C, ... in the vocabulary
vector<float> GVF::getGestureProbabilities()
{
	unsigned int ngestures = numTemplates+1;
    
	vector<float> gp(ngestures);
    setVec(gp, 0.0f);
	for(int n = 0; n < ns; n++)
		gp[g[n]] += w[n];
    
	return gp;
}

//--------------------------------------------------------------
vector< vector<float> > GVF::getParticlesPositions(){
    return particlesPositions;
}

// TEMPLATES

//--------------------------------------------------------------
// Return the number of templates in the vocabulary
int GVF::getNumberOfTemplates(){
    return gestureLengths.size();
}

//--------------------------------------------------------------
// Return the template given by its index in the vocabulary
vector< vector<float> >& GVF::getTemplateByIndex(int index){
	if (index < gestureLengths.size())
		return R_single[index];
	else
		return EmptyTemplate;
}

//--------------------------------------------------------------
// Return the length of a specific template given by its index
// in the vocabulary
int GVF::getLengthOfTemplateByIndex(int index){
	if (index < gestureLengths.size())
		return gestureLengths[index];
	else
		return -1;
}

////////////////////////////////////////////////////////////////
//
// GET & SET FUNCTIONS FOR ALL INTERNAL VALUES
//
////////////////////////////////////////////////////////////////

// PARAMETERS

//--------------------------------------------------------------
void GVF::setParameters(GVFParameters _parameters){
    parameters = _parameters;
}

GVF::GVFParameters GVF::getParameters(){
    return parameters;
}

//--------------------------------------------------------------
// Update the number of particles
void GVF::setNumberOfParticles(int numberOfParticles){
    particlesPositions.clear();
    initMat(X, numberOfParticles, pdim);          // Matrix of NS particles
    initVec(g, numberOfParticles);               // Vector of gesture class
    initVec(w, numberOfParticles);               // Weights
    //    logW = VectorXf(newNs);
    spreadParticles();
}

//--------------------------------------------------------------
int GVF::getNumberOfParticles(){
    return ns; // Return the number of particles
}

//--------------------------------------------------------------
// Update the resampling threshold used to avoid degeneracy problem
void GVF::setResamplingThreshold(int _resamplingThreshold){
    if (_resamplingThreshold >= ns) _resamplingThreshold = floor(ns/2.0f); // TODO: we should provide feedback to the GUI!!! maybe a get max resampleThresh func??
    parameters.resamplingThreshold = _resamplingThreshold;
}

//--------------------------------------------------------------
// Return the resampling threshold used to avoid degeneracy problem
int GVF::getResamplingThreshold(){
    return parameters.resamplingThreshold;
}

//--------------------------------------------------------------
// Update the standard deviation of the observation distribution
// this value acts as a tolerance for the algorithm
// low value: less tolerant so more precise but can diverge
// high value: more tolerant so less precise but converge more easily
void GVF::setTolerance(float _tolerance){
    if (_tolerance == 0.0) parameters.tolerance = 0.1; // TODO: we should provide feedback to the GUI!!!
    //_tolerance = 1.0f / (_tolerance * _tolerance);
	//tolerance = _tolerance > 0.0f ? _tolerance : tolerance;
    else parameters.tolerance = _tolerance;
}

//--------------------------------------------------------------
float GVF::getTolerance(){
    return parameters.tolerance;
}

//--------------------------------------------------------------
void GVF::setDistribution(float _distribution){
    nu = _distribution;
}

//--------------------------------------------------------------
float GVF::getDistribution(){
    return nu;
}

// COEFFICIENTS

//--------------------------------------------------------------
void GVF::setVarianceCoefficents(GVFVarianceCoefficents _coefficients){
    
    int inputDim = parameters.inputDimensions;
    coefficents = _coefficients;
    
    // change the variance coefficients only if some gestures are learned
    if (numTemplates>=0) {
        
        post("redifining variance coefficients");
        
        featVariances.clear();
        
        if (parameters.inputDimensions == 2){
            int scaleDim = spreadingParameters.scaleInitialSpreading.size();
            int rotationDim = spreadingParameters.rotationInitialSpreading.size();
            
            featVariances = vector<float> (pdim);
            
            featVariances[0]=sqrt(coefficents.phaseVariance);
            featVariances[1]=sqrt(coefficents.speedVariance);
            for (int k=0;k<scaleDim;k++) featVariances[2+k]=sqrt(coefficents.scaleVariance);
            for (int k=0;k<rotationDim;k++) featVariances[2+scaleDim+k]=sqrt(coefficents.rotationVariance);
            
            // Spreading parameters: initial value for each variation (e.g. speed start at 1.0 [i.e. original speed])
            spreadingParameters.phaseInitialSpreading = 0.15;
            spreadingParameters.speedInitialSpreading = 1.0;
            spreadingParameters.scaleInitialSpreading = vector<float> (scaleDim);
            spreadingParameters.rotationInitialSpreading = vector<float> (rotationDim);
            for (int k=0;k<scaleDim;k++) spreadingParameters.scaleInitialSpreading[k]=1.0f;
            for (int k=0;k<rotationDim;k++) spreadingParameters.rotationInitialSpreading[k]=0.0f;

        }
        else if (parameters.inputDimensions == 3){
            int scaleDim = spreadingParameters.scaleInitialSpreading.size();
            int rotationDim = spreadingParameters.rotationInitialSpreading.size();
            
            featVariances = vector<float> (pdim);
            
            featVariances[0]=sqrt(coefficents.phaseVariance);
            featVariances[1]=sqrt(coefficents.speedVariance);
            for (int k=0;k<scaleDim;k++) featVariances[2+k]=sqrt(coefficents.scaleVariance);
            for (int k=0;k<rotationDim;k++) featVariances[2+scaleDim+k]=sqrt(coefficents.rotationVariance);
            
            
            spreadingParameters.phaseInitialSpreading = 0.15;
            spreadingParameters.speedInitialSpreading = 1.0;
            spreadingParameters.scaleInitialSpreading = vector<float> (scaleDim);
            spreadingParameters.rotationInitialSpreading = vector<float> (rotationDim);
            for (int k=0;k<scaleDim;k++) spreadingParameters.scaleInitialSpreading[k]=1.0f;
            for (int k=0;k<rotationDim;k++) spreadingParameters.rotationInitialSpreading[k]=0.0f;
            
        }
        else {
            int scaleDim = spreadingParameters.scaleInitialSpreading.size();
            int rotationDim = spreadingParameters.rotationInitialSpreading.size();
            
            featVariances = vector<float> (pdim);
            
            featVariances[0]=sqrt(coefficents.phaseVariance);
            featVariances[1]=sqrt(coefficents.speedVariance);
            featVariances[2]=sqrt(coefficents.scaleVariance);
            
            // Spreading parameters: initial value for each variation (e.g. speed start at 1.0 [i.e. original speed])
            spreadingParameters.phaseInitialSpreading = 0.15;
            spreadingParameters.speedInitialSpreading = 1.0;
            spreadingParameters.scaleInitialSpreading = vector<float> (scaleDim);
            for (int k=0;k<scaleDim;k++) spreadingParameters.scaleInitialSpreading[k]=1.0f;
        }
        
    }
}

//--------------------------------------------------------------
GVF::GVFVarianceCoefficents GVF::getVarianceCoefficents(){
    return coefficents;
}

//--------------------------------------------------------------
void GVF::setPhaseVariance(float phaseVariance){
    coefficents.phaseVariance = phaseVariance;
    featVariances[0] = phaseVariance;
}

//--------------------------------------------------------------
float GVF::getPhaseVariance(){
    return coefficents.phaseVariance;
}

//--------------------------------------------------------------
void GVF::setSpeedVariance(float speedVariance){
    coefficents.speedVariance = speedVariance;
    featVariances[1] = speedVariance;
}

//--------------------------------------------------------------
float GVF::getSpeedVariance(){
    return coefficents.speedVariance;
}

//--------------------------------------------------------------
void GVF::setScaleVariance(float scaleVariance){
    coefficents.scaleVariance = scaleVariance;
    featVariances[2] = scaleVariance;
}

//--------------------------------------------------------------
float GVF::getScaleVariance(){
    return coefficents.scaleVariance;
}

//--------------------------------------------------------------
void GVF::setRotationVariance(float rotationVariance){
    if(parameters.inputDimensions > 2 && rotationVariance != 0.0){
        cout << "Warning rotation variance will not be considered for more than 2 input dimensions!" << endl;
        rotationVariance = 0.0f;
    }
    coefficents.rotationVariance = rotationVariance;
    featVariances[3] = rotationVariance;
}

//--------------------------------------------------------------
float GVF::getRotationVariance(){
    return coefficents.rotationVariance;
}

//--------------------------------------------------------------
void GVF::setSpreadingParameters(GVFInitialSpreadingParameters spreadingParameters){
    // TODO
}

//--------------------------------------------------------------
GVF::GVFInitialSpreadingParameters GVF::getSpreadingParameters(){
    return spreadingParameters;
}

//--------------------------------------------------------------
void GVF::setPhaseInitialSpreading(float phaseInitialSpreading){
    // TODO
}

//--------------------------------------------------------------
float GVF::getPhaseInitialSpreading(){
    return spreadingParameters.phaseInitialSpreading;
}

//--------------------------------------------------------------
void GVF::setSpeedInitialSpreading(float speedInitialSpreading){
    // TODO
}

//--------------------------------------------------------------
float GVF::getSpeedInitialSpreading(){
    return spreadingParameters.speedInitialSpreading;
}

//--------------------------------------------------------------
void GVF::setScaleInitialSpreading(float speedInitialSpreading){
    
}

//--------------------------------------------------------------
vector<float> GVF::getScaleInitialSpreading(){
    return spreadingParameters.scaleInitialSpreading;
}

//--------------------------------------------------------------
void GVF::setRotationInitialSpreading(float rotationInitialSpreading){
    // TODO
}

//--------------------------------------------------------------
vector<float> GVF::getRotationInitialSpreading(){
    return spreadingParameters.rotationInitialSpreading;
}



// MATHS

//--------------------------------------------------------------
// Return the standard deviation of the observation likelihood
float GVF::getObservationStandardDeviation(){
    return parameters.tolerance;
}

//--------------------------------------------------------------
// Return the particle data (each row is a particle)
vector< vector<float> > GVF::getX(){
    return X;
}

//--------------------------------------------------------------
// Return the gesture index for each particle
vector<int> GVF::getG(){
    return g;
}

//--------------------------------------------------------------
// Return particles' weights
vector<float> GVF::getW(){
    return w;
}

//--------------------------------------------------------------
// Return gestures' absolute probabilities
vector<float> GVF::getAbsoluteProbabilities(){
    return abs_weights;
}


//--------------------------------------------------------------
// Return gesture offset
vector<float>* GVF::getOffsets(){
    //return offset;
}


// UTILITIES

//--------------------------------------------------------------
// compute speed at an instant given by the index. A smoothing parameter
// allows to compute the averaged speed on a window of size smoothingWindowSize
vector<float> GVF::computeTemplateInstantaneousSpeed(int instantIndex, int smoothingWindowSize) {
    
    int numberOfTemplates = getNumberOfTemplates();
    vector<float> derivatives(numberOfTemplates);
    setVec(derivatives,(float)0.0);
    
    for (int tmplt=0; tmplt<numberOfTemplates; tmplt++){
        for (int k=instantIndex; k>instantIndex-smoothingWindowSize+1; k--) {
            
            vector<float> v1(parameters.inputDimensions);
            
            if (k>0) setVec(v1,R_single[tmplt][k]);
            else for (int l=0; l<parameters.inputDimensions; l++) v1[l]=0.0;
            
            vector<float> v2(parameters.inputDimensions);
            
            if (k-1>0) setVec(v2,R_single[tmplt][k-1]);
            else for (int l=0; l<parameters.inputDimensions; l++) v2[l]=0.0;
            
            vector<float> v3(parameters.inputDimensions);
            for (int l=0; l<parameters.inputDimensions; l++) v3[l]=v1[l]-v2[l];
            
            derivatives[tmplt]+=getNorm2(v3)/smoothingWindowSize;
        }
    }
    
    return derivatives;
    
}

//--------------------------------------------------------------
// compute speed of observations.
float GVF::computeObservationInstantaneousSpeed() {
    
    float derivative = 0.0;
    
    for (int k=observationBuffer.size()-1; k>0; k--){
        
        vector<float> v1(parameters.inputDimensions);
        setVec(v1,observationBuffer[k]);
        
        vector<float> v2(parameters.inputDimensions);
        setVec(v2,observationBuffer[k-1]);
        
        vector<float> v3(parameters.inputDimensions);
        for (int l=0; l<parameters.inputDimensions; l++) v3[l]=v1[l]-v2[l];
        
        derivative+=getNorm2(v3)/observationBuffer.size();
    }
    
    return derivative;
    
}


//--------------------------------------------------------------
// Save function. This function is used by applications to save the
// vocabulary in a text file given by filename (filename is also the complete path + filename)
void GVF::saveTemplates(string filename){
    std::string directory = filename;
    
    std::ofstream file_write(directory.c_str());
    for(int i=0; i<R_single.size(); i++){
        file_write << "template " << i << " " << inputDim << endl;
        for(int j=0; j<R_single[i].size(); j++)
        {
            for(int k=0; k<inputDim; k++)
                file_write << R_single[i][j][k] << " ";
            file_write << endl;
        }
    }
    file_write.close();
}

//--------------------------------------------------------------
// Load function. This function is used by applications to load a vocabulary
// given by filename (filename is also the complete path + filename)
void GVF::loadTemplates(string filename){
    clear();
    
    ifstream infile;
    stringstream doung;
    
    infile.open (filename.c_str(), ifstream::in);
    
    string line;
    vector<string> list;
    int cl=-1;
    while(!infile.eof())
    {
        cl++;
        infile >> line;
        //post("%i %s",cl,line.c_str());
        list.push_back(line);
    }
    
    int k=0;
    int template_starting_point = 1;
    int template_id=-1;
    int template_dim = 0;
    float* vect_0_l;
    //post("list size %i",list.size());
    
    while (k < (list.size()-1) ){ // TODO to be changed if dim>2
        if (!strcmp(list[k].c_str(),"template"))
        {
            template_id = atoi(list[k+1].c_str());
            template_dim = atoi(list[k+2].c_str());
            k=k+3;
            //post("add template %i with size %i (k=%i)", template_id, template_dim,k);
            addTemplate();
            template_starting_point = 1;
        }
        
        if (template_dim<=0){
            //post("bug dim = -1");
        }
        else{
            
            vector<float> vect(template_dim);
            if (template_starting_point==1)
            {
                // keep track of the first point
                for (int kk=0; kk<template_dim; kk++)
                {
                    vect[kk] = (float)atof(list[k+kk].c_str());
                    vect_0_l[kk] = vect[kk];
                }
                template_starting_point=0;
            }
            // store the incoming list as a vector of float
            for (int kk=0; kk<template_dim; kk++)
            {
                vect[kk] = (float)atof(list[k+kk].c_str());
                vect[kk] = vect[kk]-vect_0_l[kk];
            }
            //post("fill %i with %f %f",numTemplates,vect[0],vect[1]);
            fillTemplate(numTemplates,vect);
        }
        
        k+=template_dim;
        
    }
    
    infile.close();
}



//-----------------------------------------------------------------------------
// OTHERS FUNCTIONS

vector<vector<float> > return_RotationMatrix_3d(float phi, float theta, float psi)
{
    vector< vector<float> > M;
    initMat(M,3,3);
    
    M[0][0] = cos(theta)*cos(psi);
    M[0][1] = -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi);
    M[0][2] = sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi);
    
    M[1][0] = cos(theta)*sin(psi);
    M[1][1] = cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi);
    M[1][2] = -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi);
    
    M[2][0] = -sin(theta);
    M[2][1] = sin(phi)*cos(theta);
    M[2][2] = cos(phi)*cos(theta);
    
    return M;
    
}



