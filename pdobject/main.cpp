#include <boost/random.hpp>
#include <Eigen/Core>
#include <vector>

#include "m_pd.h"
#include "gfpf.h"
#include "globalutilities.h"

#include <iostream>
#include <fstream>
#include <string>

static t_class *gfpf_class;


typedef struct _gfpf {
    t_object  x_obj;
    gfpf *bubi;
	t_int state;
	t_int lastreferencelearned;
    std::map<int,std::vector<std::pair<float,float> > > *refmap;
	t_int Nspg, Rtpg;
	t_float sp, sv, sr, ss, so; // pos,vel,rot,scal,observation
	t_int pdim;
	Eigen::VectorXf mpvrs;
	Eigen::VectorXf rpvrs;
    
    // outlets
    t_outlet *Position,*Vitesse,*Rotation,*Scaling,*Recognition,*Likelihoods,* ActiveGesture,* TotalActiveGesture;
} t_gfpf;

static void gfpf_learn      (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_follow     (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_clear      (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_data       (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_printme    (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_restart    (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_std        (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_rt         (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_means      (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_ranges     (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gfpf_adaptspeed (t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv);

static void gfpf_auto(t_gfpf *x);

t_int restarted_l;
t_int restarted_d;
std::pair<t_float,t_float> xy0_l;
std::pair<t_float,t_float> xy0_d;
std::vector<t_float> vect_0_l;
std::vector<t_float> vect_0_d;
enum {STATE_CLEAR, STATE_LEARNING, STATE_FOLLOWING};

static void gfpf_auto(t_gfpf *x)
{

    int num_templates = 3;
    gfpf_clear(x, 0, 0, 0);
    std::string dir = "/Users/thomasrushmore/EAVI/PD/gfpflibrary/test gestures/tem";
    std::string dirg = "/Users/thomasrushmore/EAVI/PD/gfpflibrary/test gestures/g";
    
    for(int i = 0 ; i < num_templates; i++)
    {
        t_atom ab;
        ab.a_type = A_FLOAT;
        ab.a_w.w_float = i;
        gfpf_learn(x, 0, 1, &ab);
        std::string state_file(dir);
        char buf[10];
        sprintf(buf, "%d", i+1);
        state_file.append(buf);
        state_file.append(".txt");
        post(state_file.c_str());
        std::ifstream state_summary;
        state_summary.open(state_file.c_str());
        float a;
        float b;
        t_atom *ar = new t_atom[2];
                
        ar[0].a_type = A_FLOAT;
        ar[1].a_type = A_FLOAT;
        
        while(true){
            state_summary >> a;
            state_summary >> b;
            ar[0].a_w.w_float = a;
            ar[1].a_w.w_float = b;
            gfpf_data(x, 0, 2, ar);
            if(state_summary.eof()) break;
        }
        gfpf_restart(x, 0, 0, 0);
        state_summary.close();
        delete ar;
    }
    
    // gfpf follow
    gfpf_follow(x,0, 0, 0);
    // gfpf data
    std::string state_file(dirg);
    char buf[10];
    int gesture = 4;
    
    sprintf(buf, "%d",gesture);
    state_file.append(buf);
    state_file.append(".txt");
    post(state_file.c_str());
    std::ifstream state_summary;
    state_summary.open(state_file.c_str());
    t_atom *ar = new t_atom[2];
    
    ar[0].a_type = A_FLOAT;
    ar[1].a_type = A_FLOAT;
    float a,b;
    while(true){
        state_summary >> a;
        state_summary >> b;
        ar[0].a_w.w_float = a;
        ar[1].a_w.w_float = b;
        gfpf_data(x, 0, 2, ar);
        if(state_summary.eof()) break;
    }
    gfpf_restart(x,0, 0, 0);
}

static void *gfpf_new(t_symbol *s, int argc, t_atom *argv)
{
    post("\ngfpfpd - realtime adaptive gesture recognition (version: 13-09-2013)");
    post("(c) Goldsmiths, University of London and Ircam - Centre Pompidou");
    post("    contact: Baptiste Caramiaux b.caramiaux@gold.ac.uk");
    post("pd object port - v 1.1 Tom Rushmore, Goldsmiths");
    
    t_gfpf *x = (t_gfpf *)pd_new(gfpf_class);
    
    x->Nspg = 2000;
    t_int ns = x->Nspg; //!!
    x->Rtpg = 500;
    t_int rt = x->Rtpg; //!!
    
    x->sp = 0.0001;
    x->sv = 0.01;
    x->ss = 0.0001;
    x->sr = 0.000001;
    x->so = 0.2;
    x->pdim = 4;
    
    
    Eigen::VectorXf sigs(x->pdim);
    sigs << x->sp, x->sv, x->ss, x->sr;
    
    x->refmap = new std::map<int,std::vector<std::pair<float,float> > >;
    //x->refmap.clear();

    int num_particles = 2000;

    x->bubi = new gfpf(num_particles, sigs, 1./(x->so * x->so), rt, 0.);
    x->mpvrs = Eigen::VectorXf(x->pdim);
    x->rpvrs = Eigen::VectorXf(x->pdim);
    x->mpvrs << 0.05, 1.0, 1.0, 0.0;
    x->rpvrs << 0.1,  0.4, 0.3, 0.0;
    
    restarted_l=1;
    restarted_d=1;
    
    x->state = STATE_CLEAR;
    x->lastreferencelearned = -1;
    
   
    if (argc > 0)
        x->Nspg = getint(argv);
    if (argc > 1)
        x->Rtpg = getint(argv + 1);
    
    x->Position     = outlet_new(&x->x_obj, &s_list);
    x->Vitesse      = outlet_new(&x->x_obj, &s_list);
    x->Rotation     = outlet_new(&x->x_obj, &s_list);
    x->Scaling      = outlet_new(&x->x_obj, &s_list);
    x->Recognition  = outlet_new(&x->x_obj, &s_list);
    x->Likelihoods  = outlet_new(&x->x_obj, &s_list);
    x->ActiveGesture= outlet_new(&x->x_obj, &s_list);
    x->TotalActiveGesture = outlet_new(&x->x_obj, &s_list);
    
    return (void *)x;
}

static void gfpf_destructor(t_gfpf *x)
{
    post("gfpf destructor...");
    if(x->bubi != NULL)
        delete x->bubi;
    if(x->refmap != NULL)
        delete x->refmap;
    post("destructor complete");
}



// Methods order:
// - learn
// - follow
// - data
// - save_vocabulary
// - load_vocabulary
// - clear
// - printme
// - restart
// - tolerance
// - resampling_threshold
// - spreading_means
// - spreading_ranges
// - adaptation_speed


///////////////////////////////////////////////////////////
//====================== LEARN
///////////////////////////////////////////////////////////
static void gfpf_learn(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    if(argc != 1)
    {
        post("need another argument");
        return;
    }
    int refI = getint(argv);
    if(refI != x->lastreferencelearned+1)
    {
        post("you need to learn reference %d first",x->lastreferencelearned+1);
        return;
    }
    x->lastreferencelearned++;
    //x->refmap[refI];// = std::vector<std::pair<float, float> >();
    (*x->refmap)[refI] = std::vector<std::pair<float, float> >();
    post("learning reference %d", refI);
    x->bubi->addTemplate();
    x->state = STATE_LEARNING;
    restarted_l=1;
}


///////////////////////////////////////////////////////////
//====================== FOLLOW
///////////////////////////////////////////////////////////
static void gfpf_follow(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    if(x->lastreferencelearned >= 0)
    {
        x->bubi->spreadParticles(x->mpvrs,x->rpvrs);
        x->state = STATE_FOLLOWING;
    }
    else
    {
        post("no reference has been learned");
        return;
    }
}


///////////////////////////////////////////////////////////
//====================== DATA
///////////////////////////////////////////////////////////
static void gfpf_data(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    if(x->state == STATE_CLEAR)
    {
        post("what am I supposed to do ... I'm in standby!");
        return;
    }
    if(argc == 0)
    {
        post("invalid format, points have at least 1 coordinate");
        return;
    }
    if(x->state == STATE_LEARNING)
    {
        std::vector<float> vect(argc);
        if (argc ==2){
            if (restarted_l==1)
            {
                // store the incoming list as a vector of float
                for (int k=0; k<argc; k++){
                    //vect[k] = GetFloat(argv[k]);
                    //may not be correct..
                    vect[k] = getfloat(argv + k);
                }
                // keep track of the first point
                vect_0_l = vect;
                restarted_l=0;
            }
            for (int k=0; k<argc; k++){
                // repeat of above, could remove
                vect[k] = getfloat(argv + k);
                vect[k] = vect[k]-vect_0_l[k];
            }
        }
        else {
            for (int k=0; k<argc; k++)
                vect[k] = getfloat(argv + k);
        }
        
        //				pair<float,float> xy;
        //				xy.first  = x -xy0_l.first;
        //				xy.second = y -xy0_l.second;
        
        // Fill template
        x->bubi->fillTemplate(x->lastreferencelearned,vect);
        
    }
    else if(x->state == STATE_FOLLOWING)
    {
        std::vector<float> vect(argc);
        if (argc==2){
            if (restarted_d==1)
            {
                // store the incoming list as a vector of float
                for (int k=0; k<argc; k++){
                    vect[k] = getfloat(argv + k);
                }
                // keep track of the first point
                vect_0_d = vect;
                restarted_d=0;
            }
            for (int k=0; k<argc; k++){
                //?
                vect[k] = vect[k] = getfloat(argv + k);
                vect[k] = vect[k] - vect_0_d[k];
            }
        }
        else{
            post("%i",argc);
            for (int k=0; k<argc; k++)
                vect[k] = vect[k] = getfloat(argv + k);
        }
        //post("%f %f",xy(0,0),xy(0,1));
        // ------- Fill template
        x->bubi->infer(vect);
        // output recognition
        Eigen::MatrixXf statu = x->bubi->getEstimatedStatus();
        //getGestureProbabilities();
        
        t_atom *outAtoms = new t_atom[statu.rows()];
        // de-refed. may be wrong.
        for(int j = 0; j < statu.rows(); j++)
            SETFLOAT(&outAtoms[j],statu(j,0));
        outlet_list(x->Position, &s_list, statu.rows(), outAtoms);
        delete[] outAtoms;
        
        outAtoms = new t_atom[statu.rows()];
        for(int j = 0; j < statu.rows(); j++)
            SETFLOAT(&outAtoms[j],statu(j,1));
        outlet_list(x->Vitesse, &s_list, statu.rows(), outAtoms);
        delete[] outAtoms;
        
        outAtoms = new t_atom[statu.rows()];
        for(int j = 0; j < statu.rows(); j++)
            SETFLOAT(&outAtoms[j],statu(j,2));
        outlet_list(x->Rotation, &s_list, statu.rows(), outAtoms);
        delete[] outAtoms;
        
        outAtoms = new t_atom[statu.rows()];
        for(int j = 0; j < statu.rows(); j++)
            SETFLOAT(&outAtoms[j],statu(j,3));
        outlet_list(x->Scaling, &s_list, statu.rows(), outAtoms);
        delete[] outAtoms;
        
        Eigen::VectorXf gprob = x->bubi->getGestureConditionnalProbabilities();
        outAtoms = new t_atom[gprob.size()];
        for(int j = 0; j < gprob.size(); j++)
            SETFLOAT(&outAtoms[j],gprob(j,0));
        outlet_list(x->Recognition, &s_list, gprob.size(), outAtoms);
        delete[] outAtoms;
        
        gprob = x->bubi->getGestureLikelihoods();
        outAtoms = new t_atom[gprob.size()];
        for(int j = 0; j < gprob.size(); j++)
            SETFLOAT(&outAtoms[j],gprob(j,0));
        outlet_list(x->Likelihoods, &s_list, gprob.size(), outAtoms);
        delete[] outAtoms;
        
        
        
    }
}


///////////////////////////////////////////////////////////
//====================== SAVE_VOCABULARY
///////////////////////////////////////////////////////////
static void gfpf_save_vocabulary(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    unsigned int bufsize = 200;
    char * mpath;
    mpath = (char*)malloc(bufsize*sizeof(bufsize));
    atom_string(argv, mpath, bufsize);
    int i=0;
    while ( *(mpath+i)!='/' )
        i++;
    mpath = mpath+i;
    std::string filename(mpath);
    x->bubi->saveTemplates(filename);
}


///////////////////////////////////////////////////////////
//====================== LOAD_VOCABULARY
///////////////////////////////////////////////////////////
void load_vocabulary(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    unsigned int bufsize = 200;
    char * mpath;
    mpath = (char*)malloc(bufsize*sizeof(bufsize));
    atom_string(argv, mpath, bufsize);
    int i=0;
    while ( *(mpath+i)!='/' )
        i++;
    mpath = mpath+i;
    std::string filename(mpath);
    //std::string filename = "/Users/caramiaux/gotest.txt";
    x->bubi->loadTemplates(filename);
    x->lastreferencelearned=x->bubi->getNbOfTemplates()-1;
/*
    t_atom* outAtoms = new t_atom[1];
    SETFLOAT(&outAtoms[0], x->bubi->getNbOfTemplates());
    outlet_list(x->Likelihoods, &s_list, 1, outAtoms);
    delete[] outAtoms;
 */
    
}


///////////////////////////////////////////////////////////
//====================== CLEAR
///////////////////////////////////////////////////////////
static void gfpf_clear(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    x->lastreferencelearned = -1;
    x->bubi->clear();
    restarted_l=1;
    restarted_d=1;
    x->state = STATE_CLEAR;
}


///////////////////////////////////////////////////////////
//====================== PRINTME
///////////////////////////////////////////////////////////
static void gfpf_printme(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    post("\nN. particles %d: ", x->bubi->getNbOfParticles());
    post("Resampling Th. %d: ", x->bubi->getResamplingThreshold());
    post("Means %.3f %.3f %.3f %.3f: ", x->mpvrs[0], x->mpvrs[1], x->mpvrs[2], x->mpvrs[3]);
    post("Ranges %.3f %.3f %.3f %.3f: ", x->rpvrs[0], x->rpvrs[1], x->rpvrs[2], x->rpvrs[3]);
    for(int i = 0; i < x->bubi->getNbOfTemplates(); i++)
    {
        post("reference %d: ", i);
        std::vector<std::vector<float> > tplt = x->bubi->getTemplateByInd(i);
        for(int j = 0; j < tplt.size(); j++)
        {
            post("%02.4f  %02.4f", tplt[j][0], tplt[j][1]);
        }
    }
}


///////////////////////////////////////////////////////////
//====================== RESTART
///////////////////////////////////////////////////////////
static void gfpf_restart(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    restarted_l=1;
    if(x->state == STATE_FOLLOWING)
    {
        x->bubi->spreadParticles(x->mpvrs,x->rpvrs);
        restarted_l=1;
        restarted_d=1;
    }

}


///////////////////////////////////////////////////////////
//====================== tolerance
///////////////////////////////////////////////////////////
static void gfpf_tolerance(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    float stdnew = getfloat(argv);
    if (stdnew == 0.0)
        stdnew = 0.1;
    x->bubi->setIcovSingleValue(1/(stdnew*stdnew));
}


///////////////////////////////////////////////////////////
//====================== resampling_threshold
///////////////////////////////////////////////////////////
static void gfpf_resampling_threshold(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    int rtnew = getint(argv);
    int cNS = x->bubi->getNbOfParticles();
    if (rtnew >= cNS)
        rtnew = floor(cNS/2);
    x->bubi->setResamplingThreshold(rtnew);
}


///////////////////////////////////////////////////////////
//====================== spreading_means
///////////////////////////////////////////////////////////
static void gfpf_spreading_means(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    x->mpvrs = Eigen::VectorXf(x->pdim);
    x->mpvrs << getfloat(argv), getfloat(argv + 1), getfloat(argv + 2), getfloat(argv + 3);
}


///////////////////////////////////////////////////////////
//====================== spreading_ranges
///////////////////////////////////////////////////////////
static void gfpf_spreading_ranges(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    x->rpvrs = Eigen::VectorXf(x->pdim);
    x->rpvrs << getfloat(argv), getfloat(argv + 1), getfloat(argv + 2), getfloat(argv + 3);
}


///////////////////////////////////////////////////////////
//====================== adaptation_speed
///////////////////////////////////////////////////////////
static void gfpf_adaptation_speed(t_gfpf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    std::vector<float> as;
    as.push_back(getfloat(argv));
    as.push_back(getfloat(argv + 1));
    as.push_back(getfloat(argv + 2));
    as.push_back(getfloat(argv + 3));
    x->bubi->setAdaptSpeed(as);
}


extern "C"
{
    void gfpf_setup(void) {
        gfpf_class = class_new( gensym("gfpf"),(t_newmethod)gfpf_new,(t_method)gfpf_destructor,
                                sizeof(t_gfpf),CLASS_DEFAULT,A_GIMME,0);
        
        class_addmethod(gfpf_class,(t_method)gfpf_learn,gensym("learn"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_follow,gensym("follow"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_clear,gensym("clear"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_data,gensym("data"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_printme,gensym("printme"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_restart,gensym("restart"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_tolerance,gensym("std"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_resampling_threshold,gensym("rt"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_spreading_means,gensym("means"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_spreading_ranges,gensym("ranges"),A_GIMME,0);
        class_addmethod(gfpf_class,(t_method)gfpf_adaptation_speed,gensym("adaptspeed"),A_GIMME,0);        
    }
}