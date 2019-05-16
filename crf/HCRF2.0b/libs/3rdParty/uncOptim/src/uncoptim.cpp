// =============================================================================
// UNConstrained OPTIMizatoin library.
// programmer: A. Katanforoush,
// last update: 7/2/2003,
// related web page: http://math.ipm.ac.ir/scc/PointsOnSpheres,
// keywords: unconstrained optimization, secant methods, DFP method, BFGS method,
//           conjugate gradient methods, FR method, PR method, line search algorithms.
// =============================================================================

#include "uncoptim.h"
#include <math.h>

UnconstrainedOptim::UnconstrainedOptim(double FLB, double FOPT, int lineSearchMethod)
        :ZERO(DEFAULT_ZERO),GOLDEN1(DEFAULT_GOLDEN1),GOLDEN2(DEFAULT_GOLDEN2),
        RHO(DEFAULT_RHO),SIGMA(DEFAULT_SIGMA),TAW1(DEFAULT_TAW1),TAW2(DEFAULT_TAW2),TAW3(DEFAULT_TAW3),
T1(DEFAULT_T1),T2(DEFAULT_T2),T3(DEFAULT_T3),T4(DEFAULT_T4),T5(DEFAULT_T5),T6(DEFAULT_T6),maxit(DEFAULT_MAXIT) {
    n=0;
    x=0;
    g=0;
    xk=0;
    sk=0;
    gk=0;
    flb=FLB;
    fopt=FOPT;
    cnls=cnf=cng=0;
    defaultLineSearchMethod=lineSearchMethod;
}


UnconstrainedOptim::UnconstrainedOptim(int N, double FLB, double FOPT, int lineSearchMethod)
        :ZERO(DEFAULT_ZERO),GOLDEN1(DEFAULT_GOLDEN1),GOLDEN2(DEFAULT_GOLDEN2),
        RHO(DEFAULT_RHO),SIGMA(DEFAULT_SIGMA),TAW1(DEFAULT_TAW1),TAW2(DEFAULT_TAW2),TAW3(DEFAULT_TAW3),
T1(DEFAULT_T1),T2(DEFAULT_T2),T3(DEFAULT_T3),T4(DEFAULT_T4),T5(DEFAULT_T5),T6(DEFAULT_T6),maxit(DEFAULT_MAXIT) {
    n=N;
    x=new double[n];
    g=new double[n];
    xk=new double[n];
    sk=new double[n];
    gk=new double[n];
    flb=FLB;
    fopt=FOPT;
    cnls=cnf=cng=0;
    defaultLineSearchMethod=lineSearchMethod;
}

UnconstrainedOptim::UnconstrainedOptim(int N, double FLB, double FOPT, int lineSearchMethod, double _ZERO,
                                       double _GOLDEN1, double _GOLDEN2,	double _RHO, double _SIGMA,
                                       double _TAW1, double _TAW2, double _TAW3,
                                       double _T1, double _T2, double _T3, double _T4, double _T5, double _T6)
        :ZERO(_ZERO),GOLDEN1(_GOLDEN1),GOLDEN2(_GOLDEN2),
        RHO(_RHO),SIGMA(_SIGMA),TAW1(_TAW1),TAW2(_TAW2),TAW3(_TAW3),
T1(_T1),T2(_T2),T3(_T3),T4(_T4),T5(_T5),T6(_T6),maxit(DEFAULT_MAXIT) {
    n=N;
    x=new double[n];
    g=new double[n];
    flb=FLB;
    fopt=FOPT;
    xk=new double[n];
    sk=new double[n];
    gk=new double[n];
    cnls=cnf=cng=0;
    defaultLineSearchMethod=lineSearchMethod;
}

UnconstrainedOptim::~UnconstrainedOptim() {
    delete x;
    delete g;
    delete xk;
    delete sk;
    delete gk;
}


int UnconstrainedOptim::getMaxIterations()
{
	return maxit;
}

void UnconstrainedOptim::setMaxIterations( int MaxIt)
{
	maxit = MaxIt;
}

void UnconstrainedOptim::setDimension(int N)
{
    if(n!=N)
	{
		if(x) delete x;
		if(g) delete g;
		if(xk) delete xk;
		if(sk) delete sk;
		if(gk) delete gk;

		n = N;
		x=new double[n];
		g=new double[n];
		xk=new double[n];
		sk=new double[n];
		gk=new double[n];
	}
}


void UnconstrainedOptim::goldenSearch(double &hk) {
    double h1,h2,fh1,fh2,u,b;
    b=0; u=1.05;
    h1=b+GOLDEN2*(u-b);
    h2=b+GOLDEN1*(u-b);
    xh(h1);	fh1=fx();
    xh(h2);	fh2=fx();
    while( fabs(fh2-fh1)>ZERO ) {
        if( fh1>fh2 ) {
            b=h1;
            h1=h2; fh1=fh2;
            h2=b+GOLDEN1*(u-b);
            xh(h2); fh2=fx();
        }
        else {
            u=h2;
            h2=h1; fh2=fh1;
            h1=b+GOLDEN2*(u-b);
            xh(h1); fh1=fx();
        }
    }
    if( fh1>fh2 )
        hk=h2;
    else
        hk=h1;
}

void UnconstrainedOptim::BSsearch(double &h, double &g0, double &f_1) {
    // f0=f(xk), f_1=f(x(k-1))
    double f0=f;
    double m,gh;
    double h0,fh0,gh0;
    double a,b,fa,fb,ga,gb;
    int gbknown;
    int termin;

    if( fabs(g0)<1E-10 ) {
        f_1=f0;
        return;
    }

    m=(flb-f0)/(RHO*g0);

    h0=0; fh0=f0; gh0=g0;

    // f_1:=1E30; // Makes h become 1
    h=max(f_1-f0, 10*ZERO);
    h=min(-2*h/g0, 1);

    termin=0;

    // =====  Bracketing Phase  ============================
    while( termin==0 ) {
        xh(h); fx();
        if( f<=flb )
            termin=2;
        else if( f>f0+h*RHO*g0 || f>=fh0 ) {
            a=h0; fa=fh0; ga=gh0;
            b=h;  fb=f;  gbknown=0;
            termin=1;
        }
        else {
            gx();
            gh=gs();
            if( fabs(gh)<=-SIGMA*g0 )
                termin=2;
            else if( gh>0 ) {
                a=h;  fa=f;  ga=gh;
                b=h0; fb=fh0; gb=gh0;
                gbknown=1;
                termin=1;
            }
            else {
                a=2*h-h0;
                if( m<=a ) {
                    h0=h; fh0=f; gh0=gh;
                    h=m;
                }
                else {
                    b=min(h+TAW1*(h-h0), m);
                    a=UnconstrainedOptim::chooseC(a,b,h0,fh0,gh0,h,f,gh);
                    h0=h; fh0=f; gh0=gh;
                    h=a;
                }
            }
        }
    }

    if( termin==1 ) {
        termin=0;
        // =====  Sectioning Phase  ============================
        while( termin==0 ) {
            if( gbknown )
                h=UnconstrainedOptim::chooseC(a+TAW2*(b-a), b-TAW3*(b-a), a, fa, ga, b, fb, gb);
            else
                h=UnconstrainedOptim::chooseQ(a+TAW2*(b-a), b-TAW3*(b-a), a, fa, ga, b, fb);
            xh(h); fx();
            // **** Line Search End Condition by Fletcher ****
            if( (a-h)*ga<=ZERO )
                termin=1;
            else {
                if( f>f0+h*RHO*g0  || f>=fa ) {
                    b=h; fb=f; gbknown=0;
                }
                else {
                    gx();
                    gh=gs();
                    if( fabs(gh)<=-SIGMA*g0 )
                        termin=1;
                    else {
                        if( (b-a)*gh>=0 ) {
                            b=a; fb=fa; gb=ga; gbknown=1;
                        }
                        a=h; fa=f; ga=gh;
                    }
                }
            }
        }
    }

    f_1=f0; g0=gh;
}

void UnconstrainedOptim::Ssearch(double &h, double &g0, double &f_1) {
    // f0=f(xk), f_1=f(x(k-1))
    double f0=f;
    double m,a,b,a1;
    double gh,fa,ga;
    int cont;

    if( fabs(g0)<1E-10 ) {
        f_1=f0;
        return;
    }

    m=(flb-f0)/(RHO*g0);

    a=0; fa=f0; ga=g0;
    // f_1:=1E30; // Makes h become 1
    h=max(f_1-f0, 10*ZERO);
    h=min(-2*h/g0, 1);
    b=2*m+1;

    cont=1;

    while( cont ) {
        xh(h); fx();
        if( f<=flb )
            cont=0;
        else if( f>f0+h*RHO*g0 || f>=fa ) {
            if( fabs((h-a)*ga)<=ZERO )
                cont=0;
            else {
                b=h;
                h=UnconstrainedOptim::chooseQ(a+T1*(h-a),h-T2*(h-a),a,fa,ga,h,f);
            }
        }
        else {
            gx();
            gh=gs();
            if( fabs(gh)<=-SIGMA*g0 )
                cont=0;
            else {
                a1=h;
                if( (b-a)*gh<0 ) {
                    if( b<=m )
                        h=UnconstrainedOptim::chooseC(h+T5*(b-h),b-T6*(b-h), a,fa,ga, h,f,gh);
                    else
                        h=UnconstrainedOptim::chooseC(min(T3*h,m),min(T4*h,m),a,fa,ga,h,f,gh);
                }
                else {
                    h=UnconstrainedOptim::chooseC(a+T1*(h-a),h-T2*(h-a),a,fa,ga,h,f,gh);
                    b=a;
                }
                a=a1; fa=f; ga=gh;
            }
        }
    }

    f_1=f0; g0=gh;
}

void UnconstrainedOptim::RSsearch(double &h, double &g0, double &f_1) {
    // f0=f(xk), f_1=f(x(k-1))
    double f0=f;
    double m,a,b,a1,temp;
    double gh=g0,fa,ga,fa1,fb;
    int cont;

    if( fabs(g0)<1E-10 ) {
        f_1=f0;
        return;
    }

    m=(flb-f0)/(RHO*g0);

    a=0; fa=f0; ga=g0;
    // f_1:=1E30; // Makes h become 1
    h=max(f_1-f0, 10*ZERO);
    h=min(-2*h/g0, 1);
    b=2*m+1;

    cont=1;

    while( cont==1 ) {
        a1=a; fa1=fa;

        cont=2;
        xh(h); fx();
        if( f<=flb )
            cont=0;
        else if( f>f0+h*RHO*g0 || f>=fa ) {
            if( fabs((h-a)*ga)<=ZERO )
                cont=0;
            else {
                temp=h;
                if( b<=m )
                    h=UnconstrainedOptim::chooseCC(a+T1*(h-a),h-T2*(h-a),a,fa,ga,h,f,b,fb);
                else
                    h=UnconstrainedOptim::chooseQ(a+T1*(h-a),h-T2*(h-a),a,fa,ga,h,f);
                b=temp; fb=f;
                cont=1;
            }
        }

        while( cont==2 && fa1-f>max(10*ZERO,-SIGMA*g0*fabs(a1-h)) ) {
            temp=h;
            if( b<=m )
                h=UnconstrainedOptim::chooseCC(a+T1*(b-a),b-T2*(b-a),a,fa,ga,a1,fa1,h,f);
            else
                h=UnconstrainedOptim::chooseCC(a+T1*(h-a),h-T2*(h-a),a,fa,ga,a1,fa1,h,f);
            a1=temp; fa1=f;
            xh(h); fx();
            if( f<=flb )
                cont=0;
            else if( f>f0+h*RHO*g0 || f>=fa ) {
                if( fabs((h-a)*ga)<=ZERO )
                    cont=0;
                else {
                    temp=h;
                    h=UnconstrainedOptim::chooseCC(a+T1*(h-a),h-T2*(h-a),a,fa,ga,a1,fa1,h,f);
                    b=temp; fb=f;
                    cont=1;
                }
            }
        }

        if( cont==2 ) {
            cont=1;
            if( fa1<f ) {
                h=a1; f=fa1;
                xh(h);
            }

            gx();
            gh=gs();
            if( fabs(gh)<=-SIGMA*g0 )
                cont=0;
            else {
                temp=h;
                if((b-a)*gh<0) {
                    if( b<=m )
                        h=UnconstrainedOptim::chooseC(h+T5*(b-h),b-T6*(b-h), a,fa,ga,h,f,gh);
                    else
                        h=UnconstrainedOptim::chooseC(min(T3*h,m),min(T4*h,m),a,fa,ga,h,f,gh);
                }
                else {
                    h=UnconstrainedOptim::chooseC(a+T1*(h-a),h-T2*(h-a),a,fa,ga,h,f,gh);
                    b=a; fb=fa;
                }
                a=temp; fa=f; ga=gh;
            }
        }
    }

    f_1=f0; g0=gh;
}

int UnconstrainedOptim::DFPoptimize(int lineSearchMethod) {
    ///////  DFP  //////////////////////////////////////
    cnls=0; cnf=0; cng=0;

    double *gm=new double[n];
    double *hg=new double[n];

    int i,j;
    double **h;
    h=new double *[n];
    i=n;
    while(--i>=0) {
        h[i]=new double[n];
        j=n;
        while(--j>=0)
            h[i][j]=0;
        h[i][i]=1;
    }
    double fk_1=1E30;
    double g0;
    fx();
    gx();
    int cont=1;
    do {
        i=n;
        double w;
        while(--i>=0) {
            w=0;
            j=n;
            while(--j>=0)
                w-=h[i][j]*g[j];
            sk[i]=w;
        }
        g0=gs();
        update();

        switch(lineSearchMethod) {
        case GOLDENSEARCH:
            fk_1=f;
            goldenSearch(w);
            break;
        case BSSEARCH:
            BSsearch(w,g0,fk_1);
            break;
        case SSEARCH:
            Ssearch(w,g0,fk_1);
            break;
        case RSSEARCH:
            RSsearch(w,g0,fk_1);
				}
        ++cnls;

		if(cnls > maxit)
			cont = 0;

		if( fk_1-f<ZERO )
            cont=0;

        if(!gknown)
            gx();

        w=0;
        i=n;
        while(--i>=0) {
            gm[i]=g[i]-gk[i];
            w+=(x[i]-xk[i])*gm[i];
        }
        i=n;
        double v;
        while(--i>=0) {
            v=0;
            j=n;
            while(--j>=0)
                v+=h[i][j]*gm[j];
            hg[i]=v;
        }
        v=0;
        i=n;
        while(--i>=0)
            v+=gm[i]*hg[i];
        if( fabs(w)<ZERO || fabs(v)<ZERO )
            cont=0;
        else {
            i=n;
            while(--i>=0) {
                j=n;
                while(--j>=0)
                    h[i][j]+=(x[i]-xk[i])*(x[j]-xk[j])/w-hg[i]*hg[j]/v;
            }
        }
    }
    while(cont);


    delete hg;
    delete gm;
    i=n;
    while(--i>=0)
        delete h[i];
    delete h;


    if( fabs(f-fopt)<ZERO )
        return 0;
    else
        return (int)floor((1+(log(fabs(f-fopt))-log(ZERO))/log(10.0))/2.0);
}

int UnconstrainedOptim::BFGSoptimize(int lineSearchMethod) {
    ///////  BFGS  //////////////////////////////////////
    cnls=0; cnf=0; cng=0;

    double *gm=new double[n];
    double *hg=new double[n];

    int i,j;
    double **h;
    h=new double *[n];
    i=n;
    while(--i>=0) {
        h[i]=new double[n];
        j=n;
        while(--j>=0)
            h[i][j]=0;
        h[i][i]=1;
    }
    double fk_1=1E30;
    double g0;
    fx();
    gx();
    int cont=1;
    do {
        i=n;
        double w;
        while(--i>=0) {
            w=0;
            j=n;
            while(--j>=0)
                w-=h[i][j]*g[j];
            sk[i]=w;
        }
        g0=gs();
        update();

        switch(lineSearchMethod) {
        case GOLDENSEARCH:
            fk_1=f;
            goldenSearch(w);
						break;
        case BSSEARCH:
            BSsearch(w,g0,fk_1);
						break;
        case SSEARCH:
            Ssearch(w,g0,fk_1);
						break;
        case RSSEARCH:
            RSsearch(w,g0,fk_1);
				}
        ++cnls;

		if(cnls > maxit)
			cont = 0;

        if( fk_1-f<ZERO )
            cont=0;

        if(!gknown)
            gx();

        w=0;
        i=n;
        while(--i>=0) {
            gm[i]=g[i]-gk[i];
            w+=(x[i]-xk[i])*gm[i];
        }
        i=n;
        double v;
        while(--i>=0) {
            v=0;
            j=n;
            while(--j>=0)
                v+=h[i][j]*gm[j];
            hg[i]=v;
        }
        v=0;
        i=n;
        while(--i>=0)
            v+=gm[i]*hg[i];
        if( fabs(w)<ZERO )
            cont=0;
        else {
            v=1+v/w;
            i=n;
            while(--i>=0) {
                j=n;
                while(--j>=0) {
                    double di=x[i]-xk[i];
                    double dj=x[j]-xk[j];
                    h[i][j]+=v*di*dj/w-(di*hg[j]+dj*hg[i])/w;
                }
            }
        }
    }
    while(cont);

    delete hg;
    delete gm;
    i=n;
    while(--i>=0)
        delete h[i];
    delete h;

    if( fabs(f-fopt)<ZERO )
        return 0;
    else
        return (int)floor((1+(log(fabs(f-fopt))-log(ZERO))/log(10.0))/2.0);
}

int UnconstrainedOptim::FRoptimize(int lineSearchMethod, int reset) {
    ///////  Fletcher_Reeves  ///////////////////////////////
    if(reset)
        return FRoptimizeWithReset(lineSearchMethod);

    cnls=0; cnf=0; cng=0;

    double w,v,g0;
    int i;
    double fk_1=1E30;
    fx();
    gx();

    v=0;
    i=n;
    while(--i>=0) {
        sk[i]=0;
        v+=g[i]*g[i];
    }

    w=0;
    int cont=1;
    do {
        i=n;
        while(--i>=0)
            sk[i]=w*sk[i]-g[i];

        g0=gs();
        if(g0>0) {
            i=n;
            while(--i>=0)
                sk[i]=-g[i];
            g0=-v;
        }
        update();

        switch(lineSearchMethod) {
        case GOLDENSEARCH:
            fk_1=f;
            goldenSearch(w);
						break;
        case BSSEARCH:
            BSsearch(w,g0,fk_1);
						break;
        case SSEARCH:
            Ssearch(w,g0,fk_1);
						break;
        case RSSEARCH:
            RSsearch(w,g0,fk_1);
				}
        ++cnls;

		if(cnls > maxit)
			cont = 0;

		if( fk_1-f<ZERO )
            cont=0;

        if(!gknown)
            gx();

        w=0;
        i=n;
        while(--i>=0)
            w+=g[i]*g[i];
        if( fabs(v)<ZERO )
            cont=0;
        else {
            double z=w;
            w=w/v;
            v=z;
        }
    }
    while(cont);

    if( fabs(f-fopt)<ZERO )
        return 0;
    else
        return (int)floor((1+(log(fabs(f-fopt))-log(ZERO))/log(10.0))/2.0);
}

int UnconstrainedOptim::PRoptimize(int lineSearchMethod, int reset) {
    ///////  Polak_Ribiere  ///////////////////////////////
    if(reset)
        return PRoptimizeWithReset(lineSearchMethod);

    cnls=0; cnf=0; cng=0;

    double w,v,g0;
    int i;
    double fk_1=1E30;
    fx();
    gx();

    v=0;
    i=n;
    while(--i>=0) {
        sk[i]=0;
        v+=g[i]*g[i];
    }

    w=0;
    int cont=1;
    do {
        i=n;
        while(--i>=0)
            sk[i]=w*sk[i]-g[i];

        g0=gs();
        if(g0>0) {
            i=n;
            while(--i>=0)
                sk[i]=-g[i];
            g0=-v;
        }
        update();

        switch(lineSearchMethod) {
        case GOLDENSEARCH:
            fk_1=f;
            goldenSearch(w);
						break;
        case BSSEARCH:
            BSsearch(w,g0,fk_1);
						break;
        case SSEARCH:
            Ssearch(w,g0,fk_1);
						break;
        case RSSEARCH:
            RSsearch(w,g0,fk_1);
				}
        ++cnls;

		if(cnls > maxit)
			cont = 0;

		if( fk_1-f<ZERO )
            cont=0;

        if(!gknown)
            gx();

        w=0; g0=0;
        i=n;
        while(--i>=0) {
            w+=g[i]*g[i];
            g0+=g[i]*gk[i];
        }
        if( fabs(v)<ZERO )
            cont=0;
        else {
            double z=w;
            w=(w-g0)/v;
            v=z;
        }
    }
    while(cont);

    if( fabs(f-fopt)<ZERO )
        return 0;
    else
        return (int)floor((1+(log(fabs(f-fopt))-log(ZERO))/log(10.0))/2.0);
}

int UnconstrainedOptim::FRoptimizeWithReset(int lineSearchMethod) {
    ///////  Fletcher_Reeves  ///////////////////////////////
    cnls=0; cnf=0; cng=0;

    double w,v,g0;
    int i;
    double fk_1=1E30;
    fx();
    gx();

    v=0;
    i=n;
    while(--i>=0)
        v+=g[i]*g[i];

    w=0;
    int cont=1;
    int r=0;
    do {
        if(r) {
            i=n;
            while(--i>=0)
                sk[i]=w*sk[i]-g[i];
        }
        else {
            i=n;
            while(--i>=0)
                sk[i]=-g[i];
            g0=-v;
        }
        ++r;
        if(r==n)
            r=0;
        g0=gs();
        update();

        switch(lineSearchMethod) {
        case GOLDENSEARCH:
            fk_1=f;
            goldenSearch(w);
						break;
        case BSSEARCH:
            BSsearch(w,g0,fk_1);
						break;
        case SSEARCH:
            Ssearch(w,g0,fk_1);
						break;
        case RSSEARCH:
            RSsearch(w,g0,fk_1);
				}
        ++cnls;

		if(cnls > maxit)
			cont = 0;

        if( fk_1-f<ZERO )
            cont=0;

        if(!gknown)
            gx();

        w=0;
        i=n;
        while(--i>=0)
            w+=g[i]*g[i];
        if( fabs(v)<ZERO )
            cont=0;
        else {
            double z=w;
            w=w/v;
            v=z;
        }
    }
    while(cont);

    if( fabs(f-fopt)<ZERO )
        return 0;
    else
        return (int)floor((1+(log(fabs(f-fopt))-log(ZERO))/log(10.0))/2.0);
}

int UnconstrainedOptim::PRoptimizeWithReset(int lineSearchMethod) {
    ///////  Polak_Ribiere  ///////////////////////////////
    cnls=0; cnf=0; cng=0;

    double w,v,g0;
    int i;
    double fk_1=1E30;
    fx();
    gx();

    v=0;
    i=n;
    while(--i>=0)
        v+=g[i]*g[i];

    w=0;
    int cont=1;
    int r=0;
    do {
        if(r) {
            i=n;
            while(--i>=0)
                sk[i]=w*sk[i]-g[i];
        }
        else {
            i=n;
            while(--i>=0)
                sk[i]=-g[i];
            g0=-v;
        }
        ++r;
        if(r==n)
            r=0;
        g0=gs();
        update();

        switch(lineSearchMethod) {
        case GOLDENSEARCH:
            fk_1=f;
            goldenSearch(w);
						break;
        case BSSEARCH:
            BSsearch(w,g0,fk_1);
						break;
        case SSEARCH:
            Ssearch(w,g0,fk_1);
						break;
        case RSSEARCH:
            RSsearch(w,g0,fk_1);
				}
        ++cnls;

		if(cnls > maxit)
			cont = 0;

		if( fk_1-f<ZERO )
            cont=0;

        if(!gknown)
            gx();

        w=0; g0=0;
        i=n;
        while(--i>=0) {
            w+=g[i]*g[i];
            g0+=g[i]*gk[i];
        }
        if( fabs(v)<ZERO )
            cont=0;
        else {
            double z=w;
            w=(w-g0)/v;
            v=z;
        }
    }
    while(cont);

    if( fabs(f-fopt)<ZERO )
        return 0;
    else
        return (int)floor((1+(log(fabs(f-fopt))-log(ZERO))/log(10.0))/2.0);
}

