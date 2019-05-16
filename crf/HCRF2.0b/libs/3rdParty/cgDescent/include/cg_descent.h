#include <math.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef __CG_DESCENT_H
#define __CG_DESCENT_H

#ifndef MAX
	#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
	#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif
#define ABS(a)   (((a) > 0  ) ? (a):  (-(a)))

#define TRUE 1
#define FALSE 0
#define MAXLINE 200
#define EMPTY (-1)
#ifndef NULL
#define NULL 0
#endif

typedef struct cg_stats_struct
{
    double               f ; /*function value at solution */
    double           gnorm ; /* max abs component of gradient */
    int               iter ; /* number of iterations */
    int              nfunc ; /* number of function evaluations */
    int              ngrad ; /* number of gradient evaluations */
} cg_stats ;

typedef struct cg_parameter_struct
{
    double           delta ; /* Wolfe line search parameter */
    double           sigma ; /* Wolfe line search parameter */
    double           gamma ; /* decay factor for bracket interval width */
    double             rho ; /* growth factor when searching for initial
                                bracketing interval */
    double             tol ; /* computing tolerance */
    double             eta ; /* lower bound for the conjugate gradient update
                                parameter beta_k is eta*||d||_2 */
    double             eps ; /* perturbation parameter for computing fpert */
    double           fpert ; /* perturbation is eps*Ck if PertRule is T */
    double              f0 ; /* old function value */
    double              Ck ; /* average cost as given by the rule:
                                Qk = Qdecay*Qk + 1, Ck += (fabs (f) - Ck)/Qk */
    double          Qdecay ; /* factor used to compute average cost */
    double        wolfe_hi ; /* upper bound for slope in Wolfe test */
    double        wolfe_lo ; /* lower bound for slope in Wolfe test */
    double       awolfe_hi ; /* upper bound for slope, approximate Wolfe test */
    double      QuadCutOff ; /* QuadStep used when relative change in f >
                                QuadCutOff */
    double         StopFac ; /* factor multiplying starting |grad|_infty in
                                the stopping condition StopRule (see below) */
    double       AWolfeFac ; /* if AWolfe is F, then set AWolfe = T when
                                |f-f0| < Awolfe_fac*Ck) */
    double            feps ; /* stop when value change <= feps*|f| */
    double            psi0 ; /* factor used in starting guess for iteration 1 */
    double            psi1 ; /* in performing a QuadStep, we evaluate the
                                function at psi1*previous step */
    double            psi2 ; /* when starting a new cg iteration, our initial
                                guess for the line search stepsize is
                                psi2*previous step */
    int                  n ; /* problem dimension */
    int                 nf ; /* number of function evaluations */
    int                 ng ; /* number of gradient evaluations */
    int           nrestart ; /* the conjugate gradient algorithm is restarted
                                every nrestart iteration (= restart_fac*n) */
    int            nexpand ; /* abort the line search if the bracketing
                                interval is expanded nexpand times */
    int            nsecant ; /* abort the line search if the number of secant
                                iterations is nsecant */
    int              maxit ; /* abort when number of iterations reaches maxit
                                (= maxit_fac*n) */
    double           alpha ; /* stepsize along search direction */
    double               f ; /* function value for step alpha */
    double              df ; /* function derivative for step alpha */

    /*logical parameters, T means TRUE, F meand FALSE */
    int           PertRule ; /* error estimate for function value,
                                F => eps, T => eps*Ck */
    int             QuadOK ; /* T (quadratic step successful) */
    int           QuadStep ; /* T (use quadratic interpolation in line search)*/
    int         PrintLevel ; /* F (no print) T (intermediate results) */
    int         PrintFinal ; /* F (no printing) T (statistics, function value)*/
    int           StopRule ; /* T (|grad|_infty <=
                                   max(tol, initial |grad|_infty*StopFact)
                                F (|grad|_infty <= tol*(1+|f|)) */
    int             AWolfe ; /* F (use Wolfe line search)
                                T (use approximate Wolfe line search) */
    int               Step ; /* F (let code compute initial line search guess)
                                T (initial line search guess given in step
                                   arguement of cg_descent) */
    int              debug ; /* F (no debugging)
                                T (check for no increase in function value)*/
} cg_parameter ;

int cg_descent /*  return  0 (convergence tolerance satisfied)
                           1 (change in func <= feps*|f|)
                           2 (total iterations exceeded maxit)
                           3 (slope always negative in line search)
                           4 (number secant iterations exceed nsecant)
                           5 (search direction not a descent direction)
                           6 (line search fails in initial interval)
                           7 (line search fails during bisection)
                           8 (line search fails during interval update)
                           9 (debugger is on and the function value increases)*/

(
    double      grad_tol , /* StopRule = 1: |g|_infty <= max (grad_tol,
                                            StopFac*initial |g|_infty) [default]
                              StopRule = 0: |g|_infty <= grad_tol(1+|f|) */
    double            *x , /* input: starting guess, output: the solution */
    int              dim , /* problem dimension (also denoted n) */
    double    (*cg_value)  /* user provided routine to return the function */
               (double *), /* value at x */
    void       (*cg_grad)  /* user provided routine, returns in g the */
      (double *, double*), /* gradient at x*/
    double         *work , /* working array with at least 4n elements */
    double          step , /* initial step for line search
                              ignored unless Step != 0 in cg.parm */
    cg_stats      *Stats ,  /* structure with statistics(see cg_descent.h) */
	int			  maxit = -1  /* Maximum number of iterations */
) ;

int  cg_descent_init
(
    int               dim ,
    cg_parameter    *Parm
) ;

int cg_Wolfe
(
    double       alpha , /* stepsize */
    double           f , /* function value associated with stepsize alpha */
    double        dphi , /* derivative value associated with stepsize alpha */
    cg_parameter *Parm   /* cg parameters */
) ;

int cg_tol
(
    double           f , /* function value associated with stepsize */
    double       gnorm , /* gradient sup-norm */
    cg_parameter *Parm   /* cg parameters */
) ;

double cg_dot
(
    double *x , /* first vector */
    double *y , /* second vector */
    int     n  /* length of vectors */
) ;

void cg_step
(
    double *xtemp , /*output vector */
    double     *x , /* initial vector */
    double     *d , /* search direction */
    double  alpha , /* stepsize */
    int         n   /* length of the vectors */
) ;

int cg_line
(
    double       dphi0 , /* function derivative at starting point (alpha = 0) */
    double          *x , /* current iterate */
    double      *xtemp , /* x + alpha*d */
    double          *d , /* current search direction */
    double      *gtemp , /* gradient at x + alpha*d */
    double    (*cg_value)  /* user provided routine to return the function */
               (double *), /* value at x */
    void       (*cg_grad)  /* user provided routine, returns in g the */
      (double *, double*), /* gradient at x*/
    cg_parameter *Parm      /* cg parameters */
) ;

int cg_lineW
(
    double       dphi0 , /* function derivative at starting point (alpha = 0) */
    double          *x , /* current iterate */
    double      *xtemp , /* x + alpha*d */
    double          *d , /* current search direction */
    double      *gtemp , /* gradient at x + alpha*d */
    double    (*cg_value)  /* user provided routine to return the function */
               (double *), /* value at x */
    void       (*cg_grad)  /* user provided routine, returns in g the */
      (double *, double*), /* gradient at x*/
    cg_parameter *Parm      /* cg parameters */
) ;

int cg_update
(
    double          *a , /* left side of bracketing interval */
    double      *dphia , /* derivative at a */
    double          *b , /* right side of bracketing interval */
    double      *dphib , /* derivative at b */
    double      *alpha , /* trial step (between a and b) */
    double        *phi , /* function value at alpha (returned) */
    double       *dphi , /* function derivative at alpha (returned) */
    double          *x , /* current iterate */
    double      *xtemp , /* x + alpha*d */
    double          *d , /* current serach direction */
    double      *gtemp , /* gradient at x + alpha*d */
    double    (*cg_value)  /* user provided routine to return the function */
               (double *), /* value at x */
    void       (*cg_grad)  /* user provided routine, returns in g the */
      (double *, double*), /* gradient at x*/
    cg_parameter *Parm   /* cg parameters */
) ;

int cg_updateW
(
    double          *a , /* left side of bracketing interval */
    double      *dpsia , /* derivative at a */
    double          *b , /* right side of bracketing interval */
    double      *dpsib , /* derivative at b */
    double      *alpha , /* trial step (between a and b) */
    double        *phi , /* function value at alpha (returned) */
    double       *dphi , /* derivative of phi at alpha (returned) */
    double       *dpsi , /* derivative of psi at alpha (returned) */
    double          *x , /* current iterate */
    double      *xtemp , /* x + alpha*d */
    double          *d , /* current search direction */
    double      *gtemp , /* gradient at x + alpha*d */
    double    (*cg_value)  /* user provided routine to return the function */
               (double *), /* value at x */
    void       (*cg_grad)  /* user provided routine, returns in g the */
      (double *, double*), /* gradient at x*/
    cg_parameter *Parm   /* cg parameters */
) ;
#endif 
