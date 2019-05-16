// =============================================================================
// UNConstrained OPTIMizatoin library.
// programmer: A. Katanforoush,
// last update: 7/2/2003,
// related web page: http://math.ipm.ac.ir/scc/PointsOnSpheres,
// keywords: unconstrained optimization, secant methods, DFP method, BFGS method,
//           conjugate gradient methods, FR method, PR method, line search algorithms.
// =============================================================================

#ifndef __UNCOPTIM_H
#define __UNCOPTIM_H

#include <math.h>

#define GOLDENSEARCH	0
#define BSSEARCH		1
#define SSEARCH			2
#define RSSEARCH		3

// ========= Preset Parameter for Line Search Algorithms ================

#define DEFAULT_MAXIT 200
#define DEFAULT_ZERO  1E-8
#define DEFAULT_GOLDEN1  0.61803398875
#define DEFAULT_GOLDEN2  0.38196601125
#define DEFAULT_RHO  0.01
//  0 < RHO < 0.5
#define DEFAULT_SIGMA  0.2
// RHO < SIGMA < 1
#define DEFAULT_TAW1  9
//    TAW1 > 1
#define DEFAULT_TAW2  0.1
// TAW2 < SIGMA , Recommeded
#define DEFAULT_TAW3  0.5
// 0 < TAW2 <  TAW3 <  0.5
#define DEFAULT_T1  0.05
// ~0.1
#define DEFAULT_T2  0.05
// ~0.5
#define DEFAULT_T3  1.05
// ~1+T1
#define DEFAULT_T4  10
// 4-10
#define DEFAULT_T5  0.0025
// ~T1^2/(1-T1)
#define DEFAULT_T6  0.5
// ~0.5


inline double min(double a, double b) {
    return a<b ? a:b;
}

inline double max(double a, double b) {
    return a>b ? a:b;
}

inline double arg(double x, double y) {
    double z;
    if(fabs(x)>fabs(y)) {
        z=atan(y/x);
        if(x<0)
            z+=3.1415926536;
    }
    else {
        z=atan(x/y);
        if(y>0)
            z=1.5707963268-z;
        else if(x>0)
            z=-z-1.5707963268;
        else
            z=4.7123889804-z;
    }
    return z;
}


class UnconstrainedOptim {
public:
    int n;
    double *x;
    double f;
    double *g;
    int cnls,cnf,cng;
    int defaultLineSearchMethod;
	int maxit;
private:
    double flb, fopt;
    double *xk, *sk, *gk;
    int gknown;

    const double ZERO;
    const double GOLDEN1;
    const double GOLDEN2;
    const double RHO;
    const double SIGMA;
    const double TAW1;
    const double TAW2;
    const double TAW3;
    const double T1;
    const double T2;
    const double T3;
    const double T4;
    const double T5;
    const double T6;

public:
    UnconstrainedOptim(double FLB = 0.0, double FOPT = 0.0, int lineSearchMethod=BSSEARCH);
    UnconstrainedOptim(int N, double FLB = 0.0, double FOPT = 0.0, int lineSearchMethod=BSSEARCH);
    UnconstrainedOptim(int N, double FLB, double FOPT, int lineSearchMethod, double _ZERO,
                       double _GOLDEN1, double _GOLDEN2,	double _RHO, double _SIGMA,
                       double _TAW1, double _TAW2, double _TAW3,
                       double _T1, double _T2, double _T3, double _T4, double _T5, double _T6);

    virtual ~UnconstrainedOptim();

	void setDimension(int N);
	void setMaxIterations( int MaxIt);
	int getMaxIterations();

    void resetCounter() {
        cnls=cnf=cng=0;
    }

    virtual void F() = 0;	// f=F(x)
    virtual void G() = 0;	// g=Grad F(x)

private:
    double fx() {
        F();
        ++cnf;
        return f;
    }
    void gx() {
        G();
        ++cng;
        gknown=1;
    }
    void xh(double h) {
        int i;
        i=n;
        while(--i>=0)
            x[i]=xk[i]+h*sk[i];
        gknown=0;
    }
    double gs(void) {
        double w=0;
        int i=n;
        while(--i>=0)
            w+=g[i]*sk[i];
        return w;
    }

    // ==========================================================================
    // ==========================================================================

    // Q(x)=const+ px+ qx^2 ; interpolation 3 point
    static void makeQ3p(double a, double fa, double b, double fb,
                        double c, double fc, double &p, double &q) {
        double d1,d2,d3;
        double z1,z2,z3;
        d1=c-b; d2=c-a; d3=b-a;
        z1=fa/d2/d3; z2=fb/d1/d3; z3=fc/d1/d2;

        p=-((a+b)*z3-(a+c)*z2+(b+c)*z1);
        q=z1-z2+z3;
    }

    // Q(x)=const+ px+ qx^2 ; interpolation point, slope, point
    static void makeQpsp(double a, double fa, double ga, double b, double fb,
                         double &p, double &q) {
        double d,z;
        d=b-a;
        z=(fb-fa)/d;

        p=((a+b)*ga-2*a*z)/d;
        q=(z-ga)/d;
    }

    // C(x)=const+ px+ qx^2+ rx^3 ; interpolation 2 point, 2 slope
    static void makeC2ps(double a, double fa, double ga,
                         double b, double fb, double gb,
                         double &p, double &q, double &r) {
        double d, z0, z1, z2, z3;
        d=b-a; z0=a+b; z1=a+z0; z2=b+z0;
        z3=(fb-fa)/d;
        d=d*d;

        p=(a*z2*gb+b*z1*ga-6*a*b*z3)/d;
        q=-(z1*gb+z2*ga-3*z0*z3)/d;
        r=(gb+ga-2*z3)/d;
    }

    // C(x)=const+ px+ qx^2+ rx^3 ; interpolation point, slope and 2 point
    static void makeCps2p(double a, double fa, double ga,
                          double b, double fb, double c, double fc,
                          double &p, double &q, double &r) {
        double d1, d2, d3;
        double z1, z2, z3;

        d1=c-b; d2=c-a; d3=b-a;
        z1=fa/d2/d3; z2=fb/d1/d3; z3=fc/d1/d2;

        r=(ga+(d2+d3)*z1-d2*z2+d3*z3)/d2/d3;
        p=(a*b+b*c+c*a)*r-((a+b)*z3-(a+c)*z2+(b+c)*z1);
        q=z1-z2+z3-(a+b+c)*r;
    }

    // Q(returned value)=Min Q(x)=const+ px+ qx^2  on [a,b]
    static double cminQuad(double p, double q, double a, double b) {
        double w;
        if(fabs(q)<DEFAULT_ZERO) {
            if(p<0)
                return b;
            else
                return a;
        }
        else {
            w=-p/2/q;
            if(q>0) {
                if(w<a)
                    return a;
                else if(w>b)
                    return b;
                else
                    return w;
            }
            else {
                if(2*w<a+b)
                    return b;
                else
                    return a;
            }
        }
    }

    // C(returned value)=Min C(x)=const+ px+ qx^2+ rx^3  on [a,b]
    static double cminCubic(double p, double q, double r, double a, double b) {
        double delta,w,z,cw,cz;
        if(fabs(r)<DEFAULT_ZERO)
            z=cminQuad(p,q,a,b);
        else {
            delta=q*q-3*p*r;
            if( delta>0 ) {
                w=(sqrt(delta)-q)/3/r;
                z=b;
                cz=(p+(q+r*z)*z)*z;
                if( a<w && w<b ) {
                    cw=(p+(q+r*w)*w)*w;
                    if( cw<cz ) {
                        z=w;
                        cz=cw;
                    }
                }
                if((p+(q+r*a)*a)*a<cz)
                    z=a;
            }
            else { // delta<=0
                if( r>0 )
                    z=a;
                else
                    z=b;
            }
        }
        return z;
    }

    static double chooseQ(double a, double b, double h1,double fh1, double gh1, double h2, double fh2) {
        double p,q;
        makeQpsp(h1,fh1,gh1,h2,fh2,p,q);
        if( a<b )
            return cminQuad(p,q,a,b);
        else
            return cminQuad(p,q,b,a);
    }

    static double chooseC(double a, double b, double h1, double fh1, double gh1, double h2, double fh2, double gh2) {
        double p,q,r;
        makeC2ps(h1,fh1,gh1,h2,fh2,gh2,p,q,r);
        if( a<b )
            return cminCubic(p,q,r,a,b);
        else
            return cminCubic(p,q,r,b,a);
    }

    static double chooseCC(double a, double b, double h1, double fh1, double gh1, double h2, double fh2, double h3, double fh3) {
        double p,q,r;
        if( fabs(h1-h2)<DEFAULT_ZERO  || fabs(h2-h3)<DEFAULT_ZERO ) {
            makeQpsp(h1,fh1,gh1,h3,fh3,p,q);
            if( a<b )
                return cminQuad(p,q,a,b);
            else
                return cminQuad(p,q,b,a);
        }
        else {
            makeCps2p(h1,fh1,gh1,h2,fh2,h3,fh3,p,q,r);
            if( a<b )
                return cminCubic(p,q,r,a,b);
            else
                return cminCubic(p,q,r,b,a);
        }
    }


public:
    void update() {
        int i=n;
        while(--i>=0) {
            xk[i]=x[i];
            gk[i]=g[i];
        }
    }

    void goldenSearch(double &hk);
    void BSsearch(double &h, double &g0, double &f_1);
    void Ssearch(double &h, double &g0, double &f_1);
    void RSsearch(double &h, double &g0, double &f_1);

	//Davidon-Fletcher-Powell
    int DFPoptimize() {
        return DFPoptimize(defaultLineSearchMethod);
    }
	//Broyden-Fletcher-Goldfarb-Shanno [22]
    int BFGSoptimize() {
        return BFGSoptimize(defaultLineSearchMethod);
    }
	//Fletcher–Reeves
    int FRoptimize(int reset=0) {
        return FRoptimize(defaultLineSearchMethod, reset);
    }
	//Polak–Ribiere
    int PRoptimize(int reset=0) {
        return PRoptimize(defaultLineSearchMethod, reset);
    }

    int DFPoptimize(int lineSearchMethod);
    int BFGSoptimize(int lineSearchMethod);
    int FRoptimize(int lineSearchMethod, int reset);
    int PRoptimize(int lineSearchMethod, int reset);

private:
    int FRoptimizeWithReset(int lineSearchMethod);
    int PRoptimizeWithReset(int lineSearchMethod);
};

#endif

