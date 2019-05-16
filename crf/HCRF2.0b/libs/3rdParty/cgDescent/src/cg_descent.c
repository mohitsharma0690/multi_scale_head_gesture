#include "cg_descent.h"
/*     ________________________________________________________________
      |      A conjugate gradient method with guaranteed descent       |
      |             C-code Version 1.1  (October 6, 2005)              |
      |                    Version 1.2  (November 14, 2005)            |
      |           William W. Hager    and   Hongchao Zhang             |
      |          hager@math.ufl.edu       hzhang@math.ufl.edu          |
      |                   Department of Mathematics                    |
      |                     University of Florida                      |
      |                 Gainesville, Florida 32611 USA                 |
      |                      352-392-0281 x 244                        |
      |                                                                |
      |                 Copyright by William W. Hager                  |
      |                                                                |
      |          http://www.math.ufl.edu/~hager/papers/CG              |
      |________________________________________________________________|
       ________________________________________________________________
      |This program is free software; you can redistribute it and/or   |
      |modify it under the terms of the GNU General Public License as  |
      |published by the Free Software Foundation; either version 2 of  |
      |the License, or (at your option) any later version.             |
      |This program is distributed in the hope that it will be useful, |
      |but WITHOUT ANY WARRANTY; without even the implied warranty of  |
      |MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the   |
      |GNU General Public License for more details.                    |
      |                                                                |
      |You should have received a copy of the GNU General Public       |
      |License along with this program; if not, write to the Free      |
      |Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, |
      |MA  02110-1301  USA                                             |
      |________________________________________________________________|
       _________________________________________________________________
      |Note: The file cg_descent_c.parm must be placed in the directory|
      |      where the code is run                                     |
      |________________________________________________________________|
*/

int cg_descent /*  return  0 (convergence tolerance satisfied)
                           1 (change in func <= feps*|f|)
                           2 (total iterations exceeded maxit)
                           3 (slope always negative in line search)
                           4 (number secant iterations exceed nsecant)
                           5 (search direction not a descent direction)
                           6 (line search fails in initial interval)
                           7 (line search fails during bisection)
                           8 (line search fails during interval update)
                           9 (debugger is on and the function value increases)
                          -1 (parameter file not found)
                          -2 (missing parameter value in parameter file)
                          -3 (comment in parameter file too long) */
(
    double      grad_tol , /* StopRule = 1: |g|_infty <= max (grad_tol,
                                            StopFac*initial |g|_infty) [default]
                              StopRule = 0: |g|_infty <= grad_tol(1+|f|) */
    double            *x , /* input: starting guess, output: the solution */
    int              dim , /* problem dimension (also denoted n) */
    double    (*cg_value)  /* user provided routine to return the function */
               (double *), /* value cg_value(x) at x */
    void       (*cg_grad)  /* user provided routine cg_grad (g, x), g is*/
     (double *, double *), /* the gradient at x*/
    double         *work , /* working array with at least 4n elements */
    double          step , /* initial step for line search
                              ignored unless Step != 0 in cg_descent_c.parm */
    cg_stats      *Stats   /* structure with statistics (see cg_descent.h) */
)
{
    double  *d, *g, *xtemp, *gtemp, *d1, *d2, *d3, *d4,
            zero, delta2, eta_sq, Qk, Ck,
            f, ftemp, gnorm, xnorm, gnorm2, dnorm2, denom,
            t, t1, t2, t3, t4, t5, dphi, dphi0, alpha, talpha,
            yk, ykyk, ykgk, dkyk, yk1, yk2, yk3, yk4, yk5, beta ;

    int     n, n5, iter, maxit, status, i,
            cg_tol (double, double, cg_parameter *) ;

    cg_parameter Parm ;

/* initialize the parameters */

    zero = 0. ;
    gnorm = -1. ;
    iter = 0 ;
    f = zero ;
    n = dim ;
    n5 = n % 5 ;
    d = work ;
    d1 = d - 1 ;
    d2 = d - 2 ;
    d3 = d - 3 ;
    d4 = d - 4 ;
    g = d+dim ;
    xtemp = g+dim ;
    gtemp = xtemp+dim ;

    status = cg_descent_init (dim, &Parm) ;
    if ( status ) goto Exit ;
    maxit = Parm.maxit ;

    if ( Parm.Step ) alpha = step ;
    delta2 = 2*Parm.delta - 1. ;
    eta_sq = Parm.eta*Parm.eta ;
    Ck = zero ;
    Qk = zero ;

/* initial function and gradient evaluations, initial direction */

    f = cg_value (x) ;
    Parm.nf++ ;
    cg_grad (g, x) ;
    Parm.ng++ ;
    Parm.f0 = f + f ;
    gnorm = zero ;
    xnorm = zero ;
    gnorm2 = zero ;
    for (i = 0; i < n5; i++)
    {
        t = fabs (x [i]) ;
        xnorm = MAX (xnorm, t) ;
        t = g [i] ;
        d [i] = -t ;
        t = fabs (t) ;
        gnorm2 += t*t ;
        gnorm = MAX (gnorm, t) ;
    }
    for (; i < n;)
    {
        t1 = fabs (x [i]) ;
        xnorm = MAX (xnorm, t1) ;
        t1 = g [i] ;
        d [i] = -t1 ;
        t1 = fabs (t1) ;
        i++ ;

        t2 = fabs (x [i]) ;
        xnorm = MAX (xnorm, t2) ;
        t2 = g [i] ;
        d [i] = -t2 ;
        t2 = fabs (t2) ;
        i++ ;

        t3 = fabs (x [i]) ;
        xnorm = MAX (xnorm, t3) ;
        t3 = g [i] ;
        d [i] = -t3 ;
        t3 = fabs (t3) ;
        i++ ;

        t4 = fabs (x [i]) ;
        xnorm = MAX (xnorm, t4) ;
        t4 = g [i] ;
        d [i] = -t4 ;
        t4 = fabs (t4) ;
        i++ ;

        t5 = fabs (x [i]) ;
        xnorm = MAX (xnorm, t5) ;
        t5 = g [i] ;
        d [i] = -t5 ;
        t5 = fabs (t5) ;
        i++ ;

        gnorm2 += t1*t1 + t2*t2 + t3*t3 + t4*t4 + t5*t5 ;
        t2 = MAX (t2, t1) ;
        t3 = MAX (t3, t2) ;
        t4 = MAX (t4, t3) ;
        t5 = MAX (t5, t4) ;
        gnorm = MAX (gnorm, t5) ;
    }

    if ( Parm.StopRule ) Parm.tol = MAX (gnorm*Parm.StopFac, grad_tol) ;
    else                 Parm.tol = grad_tol ;

    if ( Parm.PrintLevel )
    {
        printf ("iter: %5i f = %14.6e gnorm = %14.6e AWolfe = %2i\n",
                iter, f, gnorm, Parm.AWolfe) ;
    }

    if ( cg_tol (f, gnorm, &Parm) )
    {
        status = 0 ;
        goto Exit ;
    }

    dphi0 = -gnorm2 ;
    if ( !Parm.Step )
    {
        alpha = Parm.psi0*xnorm/gnorm ;
        if ( xnorm == zero )
        {
            if ( f != zero ) alpha = Parm.psi0*fabs (f)/gnorm2 ;
            else             alpha = 1. ;
        }
    }
 
/*  start the conjugate gradient iteration
    alpha starts as old step, ends as initial step for next iteration
    f is function value for alpha = 0
    QuadOK = TRUE means that a quadratic step was taken */
 
    for (iter = 1; iter <= maxit; iter++)
    {
        Parm.QuadOK = FALSE ;
        alpha = Parm.psi2*alpha ;
        if ( Parm.QuadStep )
        {
            if ( f != zero ) t = fabs ((f-Parm.f0)/f) ;
            else             t = 1. ;
            if ( t > Parm.QuadCutOff )
            {
                talpha = Parm.psi1*alpha ;
                cg_step (xtemp, x, d, talpha, n) ;
                ftemp = cg_value (xtemp) ;
                Parm.nf++ ;
                if ( ftemp < f )
                {
                   denom = 2.*(((ftemp-f)/talpha)-dphi0) ;
                   if ( denom > zero )
                   {
                       Parm.QuadOK = TRUE ;
                       alpha = -dphi0*talpha/denom ;
                   }
                }
            }
        }
        Parm.f0 = f ;

        if ( Parm.PrintLevel )
        {
            printf ("QuadOK: %2i initial a: %14.6e f0: %14.6e dphi: %14.6e\n",
                    Parm.QuadOK, alpha, Parm.f0, dphi0) ;
        }

/* parameters in Wolfe and approximate Wolfe conditions, and in update */

        Qk = Parm.Qdecay*Qk + 1. ;
        Ck = Ck + (fabs (f) - Ck)/Qk ;

        if ( Parm.PertRule ) Parm.fpert = f + Parm.eps*Ck ;
        else                 Parm.fpert = f + Parm.eps ;

        Parm.wolfe_hi = Parm.delta*dphi0 ;
        Parm.wolfe_lo = Parm.sigma*dphi0 ;
        Parm.awolfe_hi = delta2*dphi0 ;
        Parm.alpha = alpha ;
        Parm.f = f ;
        Parm.df = dphi ;
        
        if ( Parm.AWolfe )
        {
            status =
                cg_line (dphi0, x, xtemp, d, gtemp, cg_value, cg_grad, &Parm) ;
        }
        else
        {
            status =
                cg_lineW (dphi0, x, xtemp, d, gtemp, cg_value, cg_grad, &Parm) ;
        }

        alpha = Parm.alpha ;
        f = Parm.f ;
        dphi = Parm.df ;

        if ( status ) goto Exit ;

/*Test for convergence to within machine epsilon
  [set feps to zero to remove this test] */
 
        if ( -alpha*dphi0 <= Parm.feps*fabs (f) )
        {
            status = 1 ;
            goto Exit ;
        }

/* compute beta, yk2, gnorm, gnorm2, dnorm2, update x and g */

        if ( iter % Parm.nrestart != 0 )
        {
            gnorm = zero ;
            dnorm2 = zero ;
            ykyk = zero ;
            ykgk = zero ;
            for (i = 0; i < n5; i++)
            {
                x [i] = xtemp [i] ;
                t = gtemp [i] ;
                yk = t - g [i] ;
                ykyk += yk*yk ;
                ykgk += yk*t ;
                g [i] = t ;
                t = fabs (t) ;
                gnorm = MAX (gnorm, t) ;
                dnorm2 = dnorm2 + d [i]*d [i] ;
            }
            for (; i < n; )
            {
                x [i] = xtemp [i] ;
                t1 = gtemp [i] ;
                yk1 = t1 - g [i] ;
                g [i] = t1 ;
                i++ ;

                x [i] = xtemp [i] ;
                t2 = gtemp [i] ;
                yk2 = t2 - g [i] ;
                g [i] = t2 ;
                i++ ;

                x [i] = xtemp [i] ;
                t3 = gtemp [i] ;
                yk3 = t3 - g [i] ;
                g [i] = t3 ;
                i++ ;

                x [i] = xtemp [i] ;
                t4 = gtemp [i] ;
                yk4 = t4 - g [i] ;
                g [i] = t4 ;
                i++ ;

                x [i] = xtemp [i] ;
                t5 = gtemp [i] ;
                yk5 = t5 - g [i] ;
                g [i] = t5 ;

                dnorm2 = dnorm2 + d [i]*d [i] + d1 [i]*d1 [i] + d2 [i]*d2 [i]
                                              + d3 [i]*d3 [i] + d4 [i]*d4 [i] ;
                i++ ;
                ykyk += yk1*yk1 + yk2*yk2 + yk3*yk3 + yk4*yk4 + yk5*yk5 ;
                ykgk += yk1*t1 + yk2*t2 + yk3*t3 + yk4*t4 + yk5*t5 ;
                t1 = fabs (t1) ;
                gnorm = MAX (gnorm, t1) ;
                t2 = fabs (t2) ;
                gnorm = MAX (gnorm, t2) ;
                t3 = fabs (t3) ;
                gnorm = MAX (gnorm, t3) ;
                t4 = fabs (t4) ;
                gnorm = MAX (gnorm, t4) ;
                t5 = fabs (t5) ;
                gnorm = MAX (gnorm, t5) ;
            }

            if ( cg_tol (f, gnorm, &Parm) )
            {
                status = 0 ;
                goto Exit ;
            }
            dkyk = dphi - dphi0 ;
            beta = (ykgk - 2.*dphi*ykyk/dkyk)/dkyk ;
/*
    faster: initialize dnorm2 = gnorm2 at start, then
            dnorm2 = gnorm2 + beta**2*dnorm2 - 2.*beta*dphi
            gnorm2 = ||g_{k+1}||^2
            dnorm2 = ||d_{k+1}||^2
            dpi = g_{k+1}' d_k */

            t = -1./sqrt (dnorm2*MIN (eta_sq, gnorm2)) ;
            beta = MAX (beta, t) ;

/*    update search direction d = -g + beta*dold */

            gnorm2 = zero ;
            for (i = 0; i < n5; i++)
            {
                t = g [i] ;
                d [i] = -t + beta*d [i] ;
                gnorm2 += t*t ;
            }
            for (; i < n; )
            {
                t1 = g [i] ;
                d [i] = -t1 + beta*d [i] ;
                i++ ;

                t2 = g [i] ;
                d [i] = -t2 + beta*d [i] ;
                i++ ;

                t3 = g [i] ;
                d [i] = -t3 + beta*d [i] ;
                i++ ;

                t4 = g [i] ;
                d [i] = -t4 + beta*d [i] ;
                i++ ;

                t5 = g [i] ;
                d [i] = -t5 + beta*d [i] ;
                i++ ;

                gnorm2 += t1*t1 + t2*t2 + t3*t3 + t4*t4 + t5*t5 ;
            }
            dphi0 = -gnorm2 + beta*dphi ;
        }
        else
        {

/*    search direction d = -g */

            if ( Parm.PrintLevel ) printf ("RESTART CG\n") ;
            gnorm = zero ;
            gnorm2 = zero ;
            for (i = 0; i < n5; i++)
            {
                x [i] = xtemp [i] ;
                t = gtemp [i] ;
                g [i] = t ;
                d [i] = -t ;
                t = fabs (t) ;
                gnorm = MAX (gnorm, t) ;
                gnorm2 += t*t ;
            }
            for (; i < n; )
            {
                x [i] = xtemp [i] ;
                t1 = gtemp [i] ;
                g [i] = t1 ;
                d [i] = -t1 ;
                t1 = fabs (t1) ;
                i++ ;

                x [i] = xtemp [i] ;
                t2 = gtemp [i] ;
                g [i] = t2 ;
                d [i] = -t2 ;
                t2 = fabs (t2) ;
                i++ ;

                x [i] = xtemp [i] ;
                t3 = gtemp [i] ;
                g [i] = t3 ;
                d [i] = -t3 ;
                t3 = fabs (t3) ;
                i++ ;

                x [i] = xtemp [i] ;
                t4 = gtemp [i] ;
                g [i] = t4 ;
                d [i] = -t4 ;
                t4 = fabs (t4) ;
                i++ ;

                x [i] = xtemp [i] ;
                t5 = gtemp [i] ;
                g [i] = t5 ;
                d [i] = -t5 ;
                t5 = fabs (t5) ;
                i++ ;

                t2 = MAX (t2, t1) ;
                t3 = MAX (t3, t2) ;
                t4 = MAX (t4, t3) ;
                t5 = MAX (t5, t4) ;
                gnorm = MAX (gnorm, t5) ;
                gnorm2 += t1*t1 + t2*t2 + t3*t3 + t4*t4 + t5*t5 ;
            }
            if ( cg_tol (f, gnorm, &Parm) )
            {
                status = 0 ;
                goto Exit ;
            }
            dphi0 = -gnorm2 ;
        }
        if ( !Parm.AWolfe )
        {
            if ( fabs (f-Parm.f0) < Parm.AWolfeFac*Ck ) Parm.AWolfe = TRUE ;
        }
    
        if ( Parm.PrintLevel )
        {
            printf ("\niter: %5i f = %14.6e gnorm = %14.6e AWolfe = %2i\n",
                     iter, f, gnorm, Parm.AWolfe) ;
        }

        if ( Parm.debug )
        {
            if ( f > Parm.f0 + 1.e-10*Ck )
            {
                status = 9 ;
                goto Exit ;
            }
        }
                
        if ( dphi0 > zero )
        {
           status = 5 ;
           goto Exit ;
        }
    }
    status = 2 ;

Exit:
    Stats->f = f ;
    Stats->gnorm = gnorm ;
    Stats->nfunc = Parm.nf ;
    Stats->ngrad = Parm.ng ;
    Stats->iter = iter ;
    if ( status > 2 )
    {
        gnorm = zero ;
        for (i = 0; i < n; i++)
        {
            x [i] = xtemp [i] ;
            g [i] = gtemp [i] ;
            t = fabs (g [i]) ;
            gnorm = MAX (gnorm, t) ;
        }
        Stats->gnorm = gnorm ;
    }
    if ( Parm.PrintFinal || Parm.PrintLevel )
    {
        const char mess1 [] = "Possible causes of this error message:" ;
        const char mess2 [] = "   - your tolerance may be too strict: "
                              "grad_tol = " ;
        const char mess3 [] = "Line search fails" ;
        const char mess4 [] = "   - your gradient routine has an error" ;
        const char mess5 [] = "   - the parameter epsilon in cg_descent_c.parm "
                              "is too small" ;
        printf ("\nTermination status: %i\n", status) ;
        if ( status == 0 )
        {
            printf ("Convergence tolerance for gradient satisfied\n") ;
        }
        else if ( status == 1 )
        {
            printf ("Terminating since change in function value "
                    "<= feps*|f|\n") ;
        }
        else if ( status == 2 )
        {
            printf ("Number of iterations reached max allowed: %10i\n", maxit) ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
        }
        else if ( status == 3 )
        {
            printf ("Slope always negative in line search\n") ;
            printf ("%s\n", mess1) ;
            printf ("   - your cost function has an error\n") ;
            printf ("%s\n", mess4) ;
        }
        else if ( status == 4 )
        {
            printf ("Line search fails, too many secant steps\n") ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
        }
        else if ( status == 5 )
        {
            printf ("Search direction not a descent direction\n") ;
        }
        else if ( status == 6 )
        {
            printf ("%s\n", mess3) ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
            printf ("%s\n", mess4) ;
            printf ("%s\n", mess5) ;
        }
        else if ( status == 7 )
        {
            printf ("%s\n", mess3) ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
        }
        else if ( status == 8 )
        {
            printf ("%s\n", mess3) ;
            printf ("%s\n", mess1) ;
            printf ("%s %e\n", mess2, grad_tol) ;
            printf ("%s\n", mess4) ;
            printf ("%s\n", mess5) ;
        }
        else if ( status == 9 )
        {
            printf ("Debugger is on, function value does not improve\n") ;
            printf ("new value: %25.16e old value: %25.16e\n", f, Parm.f0) ;
        }
        else if ( status < 0 )
        {
            if ( status == -1 )
            {
                printf ("cg_descent parameter file (cg_descent_c.parm) "
                        "not found\n") ;
            }
            else if ( status == -2 )
            {
                printf ("cg_descent_c.parm file is incomplete\n") ;
            }
            else if ( status == -3 )
            {
                printf ("the length of a comment statement in the "
                         "file cg_descent_c.parm exceeds the maximum allowed "
                         "length %i\n", MAXLINE) ;
            }
        }

        printf ("absolute largest component of gradient: %e\n", gnorm) ;
        printf ("function value: %e\n", f) ;
        printf ("cg iterations: %i\n", iter) ;
        printf ("function evaluations: %i\n", Parm.nf) ;
        printf ("gradient evaluations: %i\n", Parm.ng) ;
    }
    return (status) ;
}

/*
    PARAMETERS:
 
    delta - range [0, .5], used in the Wolfe conditions
    sigma - range [delta, 1], used in the Wolfe conditions
    eps - range [0, infty], used to compute line search perturbation
    gamma - range [0,1], determines when to perform bisection step
    rho   - range [1, infty], growth factor when finding initial interval
    eta   - range [0, infty], used in lower bound for beta
    psi0  - range [0, 1], factor used in very initial starting guess
    psi1  - range [0, 1], factor previous step multiplied by in QuadStep
    psi2  - range [1, infty], factor previous step is multipled by for startup
    QuadCutOff - perform QuadStep if relative change in f > QuadCutOff
    StopFac - used in StopRule
    AWolfeFac - used to decide when to switch from Wolfe to AWolfe
    restart_fac - range [0, infty] restart cg when iter = n*restart 
    maxit_fac - range [0, infty] terminate in maxit = maxit_fac*n iterations
    feps - stop when -alpha*dphi0 [est. change in value] <= feps*|f|
           [feps = 0 removes this test, example: feps = eps*1.e-5
            where eps is machine epsilon]
    tol   - range [0, infty], convergence tolerance
    nexpand - range [0, infty], number of grow/shrink allowed in bracket
    nsecant - range [0, infty], maximum number of secant steps
    PertRule - gives the rule used for the perturbation in f
                F => fpert = eps
                T => fpert = eps*Ck, Ck is an average of prior |f|
                            Ck is an average of prior |f|
    QuadStep- TRUE [use quadratic step] FALSE [no quadratic step]
    PrintLevel- FALSE [no printout] TRUE [print intermediate results]
    PrintFinal- FALSE [no printout] TRUE [print messages, final error]
    StopRule - TRUE [max abs grad <= max [tol, StopFac*initial abs grad]]
               FALSE [... <= tol*[1+|f|]]
    AWolfe - FALSE [use standard Wolfe initially]
           - TRUE [use approximate + standard Wolfe]
    Step - FALSE [program computing starting step at iteration 0]
         - TRUE [user provides starting step in gnorm argument of cg_descent
    debug - FALSE [no debugging]
          - TRUE [check that function values do not increase]
    info  - same as status
 
    DEFAULT PARAMETER VALUES:
 
        delta : 0.1
        sigma : 0.9
        eps : 1.e-6
        gamma : 0.66
        rho   : 5.0
        restart: 1.0
        eta   : 0.01
        psi0  : 0.01
        psi1  : 0.1 
        psi2  : 2.0 
        QuadCutOff: 1.d-12
        StopFac: 0.
        AWolfeFac: 1.d-3
        tol   : grad_tol
        nrestart: n [restart_fac = 1]
        maxit : 500*n [maxit_fac = 500]
        feps : 0.0
        Qdecay : 0.7
        nexpand: 50
        nsecant: 50
        PertRule: TRUE
        QuadStep: TRUE
        PrintLevel: FALSE
        PrintFinal: TRUE
        StopRule: TRUE
        AWolfe: TRUE
        Step: FALSE
        debug: FALSE
        info  : 0
        feps  : 0.0
 
     (double) grad_tol-- used in stopping rule
     (int)    dim     --problem dimension [also denoted n]
*/

int  cg_descent_init
(
    int               dim ,
    cg_parameter    *Parm
)
{
    char junk [MAXLINE+1] ;
    int info ;
    double restart_fac, maxit_fac ;
    FILE *ParmFile ;

    Parm->n = dim ;
    Parm->nf = 0 ;
    Parm->ng = 0 ;

    ParmFile = fopen ("cg_descent_c.parm", "r") ;
    if ( ParmFile == NULL ) return (-1) ;

    info = fscanf (ParmFile, "%lg", &(Parm->delta)) ;
    if ( info != 1 )
    {
        printf ("delta parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->sigma)) ;
    if ( info != 1 )
    {
        printf ("sigma parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->eps)) ;
    if ( info != 1 )
    {
        printf ("eps parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-2) ;

    info = fscanf (ParmFile, "%lg", &(Parm->gamma)) ;
    if ( info != 1 )
    {
        printf ("gamma parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->rho)) ;
    if ( info != 1 )
    {
        printf ("rho parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->eta)) ;
    if ( info != 1 )
    {
        printf ("eta parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->psi0)) ;
    if ( info != 1 )
    {
        printf ("psi0 parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->psi1)) ;
    if ( info != 1 )
    {
        printf ("psi1 parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->psi2)) ;
    if ( info != 1 )
    {
        printf ("psi2 parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->QuadCutOff)) ;
    if ( info != 1 )
    {
        printf ("QuadCutOff parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->StopFac)) ;
    if ( info != 1 )
    {
        printf ("StopFact parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->AWolfeFac)) ;
    if ( info != 1 )
    {
        printf ("AWolfeFac parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &restart_fac) ;
    if ( info != 1 )
    {
        printf ("restart_fac parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    Parm->nrestart = dim*restart_fac ;
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &maxit_fac) ;
    if ( info != 1 )
    {
        printf ("maxit_fac parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    Parm->maxit = dim*maxit_fac ;
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->feps)) ;
    if ( info != 1 )
    {
        printf ("feps parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%lg", &(Parm->Qdecay)) ;
    if ( info != 1 )
    {
        printf ("Qdecay parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->nexpand)) ;
    if ( info != 1 )
    {
        printf ("nexpand parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->nsecant)) ;
    if ( info != 1 )
    {
        printf ("nsecant parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->PertRule)) ;
    if ( info != 1 )
    {
        printf ("PertRule parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->QuadStep)) ;
    if ( info != 1 )
    {
        printf ("QuadStep parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->PrintLevel)) ;
    if ( info != 1 )
    {
        printf ("PrintLevel parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->PrintFinal)) ;
    if ( info != 1 )
    {
        printf ("PrintFinal parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->StopRule)) ;
    if ( info != 1 )
    {
        printf ("StopRule parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->AWolfe)) ;
    if ( info != 1 )
    {
        printf ("AWolfe parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->Step)) ;
    if ( info != 1 )
    {
        printf ("Step parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;

    info = fscanf (ParmFile, "%i", &(Parm->debug)) ;
    if ( info != 1 )
    {
        printf ("debug parameter missing in cg_descent_c.parm\n") ;
        return (-2) ;
    }
    fgets (junk, MAXLINE, ParmFile) ;
    if (strlen (junk) >= MAXLINE-1) return (-3) ;
    fclose (ParmFile) ;
    return (0) ;
}

/*check whether the Wolfe or the approximate Wolfe conditions are satisfied */

int cg_Wolfe
(
    double       alpha , /* stepsize */
    double           f , /* function value associated with stepsize alpha */
    double        dphi , /* derivative value associated with stepsize alpha */
    cg_parameter *Parm   /* cg parameters */
)
{
    if ( dphi >= Parm->wolfe_lo )
    {

/* test original Wolfe conditions */

        if ( f - Parm->f0 <= alpha*Parm->wolfe_hi )
        {
            if ( Parm->PrintLevel )
            {
                printf ("wolfe f: %14.6e f0: %14.6e dphi: %14.6e\n",
                         f, Parm->f0, dphi) ;
            }
            return (1) ;
        }
/* test approximate Wolfe conditions */
        else if ( Parm->AWolfe )
        {
            if ( (f <= Parm->fpert) && (dphi <= Parm->awolfe_hi) )
            {
                if ( Parm->PrintLevel )
                {
                    printf ("f: %14.6e fpert: %14.6e dphi: %14.6e awolf_hi: "
                            "%14.6e\n", f, Parm->fpert, dphi, Parm->awolfe_hi) ;
                }
                return (1) ;
            }
        }
    }
    return (0) ;
}

/* check for convergence of the cg iterations */

int cg_tol
(
    double           f , /* function value associated with stepsize */
    double       gnorm , /* gradient sup-norm */
    cg_parameter *Parm   /* cg parameters */
)
{
    if ( Parm->StopRule )
    {
        if ( gnorm <= Parm->tol ) return (1) ;
    }
    else if ( gnorm <= Parm->tol*(1.0 + fabs (f)) ) return (1) ;
    return (0) ;
}

/* compute dot product of x and y, vectors of length n */

double cg_dot
(
    double *x , /* first vector */
    double *y , /* second vector */
    int     n  /* length of vectors */
)
{
    int i, n5 ;
    double t ;
    t = 0. ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) t += x [i]*y [i] ;
    for (; i < n; i += 5)
    {
        t += x [i]*y[i] + x [i+1]*y [i+1] + x [i+2]*y [i+2]
           + x [i+3]*y [i+3] + x [i+4]*y [i+4] ;
    }
    return (t) ;
}

/* compute xtemp = x + alpha d */

void cg_step
(
    double *xtemp , /*output vector */
    double     *x , /* initial vector */
    double     *d , /* search direction */
    double  alpha , /* stepsize */
    int         n   /* length of the vectors */
)
{
    int n5, i ;
    n5 = n % 5 ;
    for (i = 0; i < n5; i++) xtemp [i] = x[i] + alpha*d[i] ;
    for (; i < n;)
    { 
        xtemp [i] = x [i] + alpha*d [i] ;
        i++ ;
        xtemp [i] = x [i] + alpha*d [i] ;
        i++ ;
        xtemp [i] = x [i] + alpha*d [i] ;
        i++ ;
        xtemp [i] = x [i] + alpha*d [i] ;
        i++ ;
        xtemp [i] = x [i] + alpha*d [i] ;
        i++ ;
    }
}

/* approximate Wolfe line search routine */

int cg_line
(
    double       dphi0 , /* function derivative at starting point (alpha = 0) */
    double          *x , /* current iterate */
    double      *xtemp , /* x + alpha*d */
    double          *d , /* current search direction */
    double      *gtemp , /* gradient at x + alpha*d */
    double  (*cg_value)  /* user provided routine to return the function */
             (double *), /* value at x */
    void     (*cg_grad)  /* user provided routine, returns in g the */
   (double *, double *), /* gradient at x*/
    cg_parameter *Parm      /* cg parameters */
)
{
    double a, dphia, b, dphib, c, alpha, phi, dphi,
           a0, da0, b0, db0, width, fquad, zero,
           cg_dot (double *, double *, int) ;

    int n, nsecant, nshrink, ngrow, status, iter,
        cg_Wolfe (double, double, double, cg_parameter *),
        cg_update (double *, double *, double *, double *, double *, double *,
                   double *, double *, double *, double *, double *,
                   double (double *), void (double *, double *),
                   cg_parameter *) ;

    alpha = Parm->alpha ;
    phi = Parm->f ;
    dphi = Parm->df ;
    n = Parm->n ;
    zero = 0. ;
    cg_step (xtemp, x, d, alpha, n) ;
    cg_grad (gtemp, xtemp) ;
    Parm->ng++ ;
    dphi = cg_dot (gtemp, d, n) ;
 
/*Find initial interval [a,b] such that dphia < 0, dphib >= 0,
         and phia <= phi0 + feps*fabs (phi0) */
 
    a = zero ;
    dphia = dphi0  ;
    ngrow = 0 ;
    nshrink = 0 ;
    while ( dphi < zero )
    {
        phi = cg_value (xtemp) ;
        Parm->nf++ ;

/* if quadstep in effect and quadratic conditions hold, check wolfe condition*/

        if ( Parm->QuadOK )
        {
            if ( ngrow == 0 ) fquad = MIN (phi, Parm->f0) ;
            if ( phi <= fquad )
            {
                if ( Parm->PrintLevel )
                {
                    printf ("alpha: %14.6e phi: %14.6e fquad: %14.6e\n",
                            alpha, phi, fquad) ;
                }
                if ( cg_Wolfe (alpha, phi, dphi, Parm) )
                {
                    status = 0 ;
                    goto Exit ;
                }
            }
        }
        if ( phi <= Parm->fpert )
        {
            a = alpha ;
            dphia = dphi ;
        }
        else
        {

/* contraction phase */

            b = alpha ;
            while ( TRUE )
            {
                alpha = .5*(a+b) ;
                nshrink++ ;
                if ( nshrink > Parm->nexpand )
                {
                    status = 6 ;
                    goto Exit ;
                }
                cg_step (xtemp, x, d, alpha, n) ;
                cg_grad (gtemp, xtemp) ;
                Parm->ng++ ;
                dphi = cg_dot (gtemp, d, n) ;
                if ( dphi >= zero ) goto Secant ;
                phi = cg_value (xtemp) ;
                Parm->nf++ ;
                if ( Parm->PrintLevel )
                {
                    printf ("contract, a: %14.6e b: %14.6e alpha: %14.6e phi: "
                            "%14.6e dphi: %14.6e\n", a, b, alpha, phi, dphi) ;
                }
                if ( Parm->QuadOK && (phi <= fquad) )
                {
                    if ( cg_Wolfe (alpha, phi, dphi, Parm) )
                    {
                        status = 0 ;
                        goto Exit ;
                    }
                }
                if ( phi <= Parm->fpert )
                {
                    a = alpha ;
                    dphia = dphi ;
                }
                else
                {
                    b = alpha ;
                }
            }
        }

/* expansion phase */

        ngrow++ ;
        if ( ngrow > Parm->nexpand )
        {
            status = 3 ;
            goto Exit ;
        }
        alpha = Parm->rho*alpha ;
        cg_step (xtemp, x, d, alpha, n) ;
        cg_grad (gtemp, xtemp) ;
        Parm->ng++ ;
        dphi = cg_dot (gtemp, d, n) ;
        if ( Parm->PrintLevel )
        {
            printf ("expand,   a: %14.6e alpha: %14.6e phi: "
                     "%14.6e dphi: %14.6e\n", a, alpha, phi, dphi) ;
        }
    }

Secant:
    b = alpha ;
    dphib = dphi ;
    if ( Parm->QuadOK )
    {
        phi = cg_value (xtemp) ;
        Parm->nf++ ;
        if ( ngrow + nshrink == 0 ) fquad = MIN (phi, Parm->f0) ;
        if ( phi <= fquad )
        {
            if ( cg_Wolfe (alpha, phi, dphi, Parm) )
            {
                status = 0 ;
                goto Exit ;
            }
        }
    }
    nsecant = Parm->nsecant ;
    for (iter = 1; iter <= nsecant; iter++)
    {
        if ( Parm->PrintLevel )
        {
            printf ("secant, a: %14.6e b: %14.6e da: %14.6e db: %14.6e\n",
                     a, b, dphia, dphib) ;
        }
        width = Parm->gamma*(b - a) ;
        if ( -dphia <= dphib ) alpha = a - (a-b)*(dphia/(dphia-dphib)) ;
        else                   alpha = b - (a-b)*(dphib/(dphia-dphib)) ;
        c = alpha ;
        a0 = a ;
        b0 = b ;
        da0 = dphia ;
        db0 = dphib ;
        status = cg_update (&a, &dphia, &b, &dphib, &alpha, &phi,
                    &dphi, x, xtemp, d, gtemp, cg_value, cg_grad, Parm) ;
        if ( status >= 0 ) goto Exit ;
        else if ( status == -2 )
        {
            if ( c == a )
            {
                if ( dphi > da0 ) alpha = c - (c-a0)*(dphi/(dphi-da0)) ;
                else              alpha = a ;
            }
            else
            {
                if ( dphi < db0 ) alpha = c - (c-b0)*(dphi/(dphi-db0)) ;
                else              alpha = b ;
            }
            if ( (alpha > a) && (alpha < b) )
            {
                if ( Parm->PrintLevel ) printf ("2nd secant\n") ;
                status = cg_update (&a, &dphia, &b, &dphib, &alpha, &phi,
                           &dphi, x, xtemp, d, gtemp, cg_value, cg_grad, Parm) ;
                if ( status >= 0 ) goto Exit ;
            }
        }

/* bisection iteration */

        if ( b-a >= width )
        {
            alpha = .5*(b+a) ;
            if ( Parm->PrintLevel ) printf ("bisection\n") ;
            status = cg_update (&a, &dphia, &b, &dphib, &alpha, &phi,
                        &dphi, x, xtemp, d, gtemp, cg_value, cg_grad, Parm) ;
            if ( status >= 0 ) goto Exit ;
        }
        else if ( b <= a )
        {
            status = 7 ;
            goto Exit ;
        }
    }
    status = 4 ;

Exit:
    Parm->alpha = alpha ;
    Parm->f = phi ;
    Parm->df = dphi ;
    return (status) ;
}

/*
   ordinary Wolfe line search routine
   This routine is identical to cg_line except that the function
   psi [a] = phi [a] - phi [0] - a*delta*dphi [0] is minimized instead of
   the function phi
*/

int cg_lineW
(
    double       dphi0 , /* function derivative at starting point (alpha = 0) */
    double          *x , /* current iterate */
    double      *xtemp , /* x + alpha*d */
    double          *d , /* current search direction */
    double      *gtemp , /* gradient at x + alpha*d */
    double  (*cg_value)  /* user provided routine to return the function */
             (double *), /* value at x */
    void     (*cg_grad)  /* user provided routine, returns in g the */
    (double *, double*), /* gradient at x*/
    cg_parameter *Parm      /* cg parameters */
)
{
    double a, dpsia, b, dpsib, c, alpha, phi, dphi,
           a0, da0, b0, db0, width, fquad, zero, psi, dpsi,
           cg_dot (double *, double *, int) ;

    int n, nsecant, nshrink, ngrow, status, iter,
        cg_Wolfe (double, double, double, cg_parameter *),
        cg_updateW (double *, double *, double *, double *, double *, double *,
                    double *, double *, double *, double *, double *, double *,
                    double (double *), void (double *, double *),
                    cg_parameter *) ;

    alpha = Parm->alpha ;
    phi = Parm->f ;
    dphi = Parm->df ;
    n = Parm->n ;
    zero = 0. ;
    cg_step (xtemp, x, d, alpha, n) ;
    cg_grad (gtemp, xtemp) ;
    Parm->ng++ ;
    dphi = cg_dot (gtemp, d, n) ;
    dpsi = dphi - Parm->wolfe_hi ;
 
/*Find initial interval [a,b] such that dphia < 0, dphib >= 0,
         and phia <= phi0 + feps*fabs (phi0) */
 
    a = zero ;
    dpsia = dphi0 - Parm->wolfe_hi ;
    ngrow = 0 ;
    nshrink = 0 ;
    while ( dpsi < zero )
    {
        phi = cg_value (xtemp) ;
        psi = phi - alpha*Parm->wolfe_hi ;
        Parm->nf++ ;

/* if quadstep in effect and quadratic conditions hold, check wolfe condition*/

        if ( Parm->QuadOK )
        {
            if ( ngrow == 0 ) fquad = MIN (phi, Parm->f0) ;
            if ( phi <= fquad )
            {
                if ( Parm->PrintLevel )
                {
                    printf ("alpha: %14.6e phi: %14.6e fquad: %14.6e\n",
                            alpha, phi, fquad) ;
                }
                if ( cg_Wolfe (alpha, phi, dphi, Parm) )
                {
                    status = 0 ;
                    goto Exit ;
                }
            }
        }
        if ( psi <= Parm->fpert )
        {
            a = alpha ;
            dpsia = dphi ;
        }
        else
        {

/* contraction phase */

            b = alpha ;
            while ( TRUE )
            {
                alpha = .5*(a+b) ;
                nshrink++ ;
                if ( nshrink > Parm->nexpand )
                {
                    status = 6 ;
                    goto Exit ;
                }
                cg_step (xtemp, x, d, alpha, n) ;
                cg_grad (gtemp, xtemp) ;
                Parm->ng++ ;
                dphi = cg_dot (gtemp, d, n) ;
                dpsi = dphi - Parm->wolfe_hi ;
                if ( dpsi >= zero ) goto Secant ;
                phi = cg_value (xtemp) ;
                psi = phi - alpha*Parm->wolfe_hi ;
                Parm->nf++ ;
                if ( Parm->PrintLevel )
                {
                    printf ("contract, a: %14.6e b: %14.6e alpha: %14.6e phi: "
                            "%14.6e dphi: %14.6e\n", a, b, alpha, phi, dphi) ;
                }
                if ( Parm->QuadOK && (phi <= fquad) )
                {
                    if ( cg_Wolfe (alpha, phi, dphi, Parm) )
                    {
                        status = 0 ;
                        goto Exit ;
                    }
                }
                if ( psi <= Parm->fpert )
                {
                    a = alpha ;
                    dpsia = dpsi ;
                }
                else
                {
                    b = alpha ;
                }
            }
        }

/* expansion phase */

        ngrow++ ;
        if ( ngrow > Parm->nexpand )
        {
            status = 3 ;
            goto Exit ;
        }
        alpha = Parm->rho*alpha ;
        cg_step (xtemp, x, d, alpha, n) ;
        cg_grad (gtemp, xtemp) ;
        Parm->ng++ ;
        dphi = cg_dot (gtemp, d, n) ;
        dpsi = dphi - Parm->wolfe_hi ;
        if ( Parm->PrintLevel )
        {
            printf ("expand,   a: %14.6e alpha: %14.6e phi: "
                     "%14.6e dphi: %14.6e\n", a, alpha, phi, dphi) ;
        }
    }

Secant:
    b = alpha ;
    dpsib = dpsi ;
    if ( Parm->QuadOK )
    {
        phi = cg_value (xtemp) ;
        Parm->nf++ ;
        if ( ngrow + nshrink == 0 ) fquad = MIN (phi, Parm->f0) ;
        if ( phi <= fquad )
        {
            if ( cg_Wolfe (alpha, phi, dphi, Parm) )
            {
                status = 0 ;
                goto Exit ;
            }
        }
    }
    nsecant = Parm->nsecant ;
    for (iter = 1; iter <= nsecant; iter++)
    {
        if ( Parm->PrintLevel )
        {
            printf ("secant, a: %14.6e b: %14.6e da: %14.6e db: %14.6e\n",
                     a, b, dpsia, dpsib) ;
        }
        width = Parm->gamma*(b - a) ;
        if ( -dpsia <= dpsib ) alpha = a - (a-b)*(dpsia/(dpsia-dpsib)) ;
        else                   alpha = b - (a-b)*(dpsib/(dpsia-dpsib)) ;
        c = alpha ;
        a0 = a ;
        b0 = b ;
        da0 = dpsia ;
        db0 = dpsib ;
        status = cg_updateW (&a, &dpsia, &b, &dpsib, &alpha, &phi, &dphi,
                   &dpsi, x, xtemp, d, gtemp, cg_value, cg_grad, Parm) ;
        if ( status >= 0 ) goto Exit ;
        else if ( status == -2 )
        {
            if ( c == a )
            {
                if ( dpsi > da0 ) alpha = c - (c-a0)*(dpsi/(dpsi-da0)) ;
                else              alpha = a ;
            }
            else
            {
                if ( dpsi < db0 ) alpha = c - (c-b0)*(dpsi/(dpsi-db0)) ;
                else              alpha = b ;
            }
            if ( (alpha > a) && (alpha < b) )
            {
                if ( Parm->PrintLevel ) printf ("2nd secant\n") ;
                status = cg_updateW (&a, &dpsia, &b, &dpsib, &alpha, &phi,
                    &dphi, &dpsi, x, xtemp, d, gtemp, cg_value, cg_grad, Parm) ;
                if ( status >= 0 ) goto Exit ;
            }
        }

/* bisection iteration */

        if ( b-a >= width )
        {
            alpha = .5*(b+a) ;
            if ( Parm->PrintLevel ) printf ("bisection\n") ;
            status = cg_updateW (&a, &dpsia, &b, &dpsib, &alpha, &phi, &dphi,
                       &dpsi, x, xtemp, d, gtemp, cg_value, cg_grad, Parm) ;
            if ( status >= 0 ) goto Exit ;
        }
        else if ( b <= a )
        {
            status = 7 ;
            goto Exit ;
        }
    }
    status = 4 ;

Exit:
    Parm->alpha = alpha ;
    Parm->f = phi ;
    Parm->df = dphi ;
    return (status) ;
}

/* update returns: 8 if too many iterations
                   0 if Wolfe condition is satisfied
                  -1 if interval is updated and a search is done
                  -2 if the interval updated successfully */

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
    double          *d , /* current search direction */
    double      *gtemp , /* gradient at x + alpha*d */
    double    (*cg_value)  /* user provided routine to return the function */
               (double *), /* value at x */
    void       (*cg_grad)  /* user provided routine, returns in g the */
      (double *, double*), /* gradient at x*/
    cg_parameter *Parm   /* cg parameters */
)
{
    double zero, cg_dot (double *, double *, int) ;
    int n, nshrink, status, cg_Wolfe (double, double, double, cg_parameter *) ;

    zero = 0. ;
    n = Parm->n ;
    cg_step (xtemp, x, d, *alpha, n) ;
    *phi = cg_value (xtemp) ;
    Parm->nf++ ;
    cg_grad (gtemp, xtemp) ;
    Parm->ng++ ;
    *dphi = cg_dot (gtemp, d, n) ;
    if ( Parm->PrintLevel )
    {
        printf ("update alpha: %14.6e phi: %14.6e dphi: %14.6e\n",
                 *alpha, *phi, *dphi) ;
    }
    if ( cg_Wolfe (*alpha, *phi, *dphi, Parm) )
    {
        status = 0 ;
        goto Exit2 ;
    }
    status = -2 ;
    if ( *dphi >= zero )
    {
        *b = *alpha ;
        *dphib = *dphi ;
        goto Exit2 ;
    }
    else
    {
        if ( *phi <= Parm->fpert )
        {
            *a = *alpha ;
            *dphia = *dphi ;
            goto Exit2 ;
        }
    }
    nshrink = 0 ;
    *b = *alpha ;
    while ( TRUE )
    {
        *alpha = .5*(*a + *b) ;
        nshrink++ ;
        if ( nshrink > Parm->nexpand )
        {
            status = 8 ;
            goto Exit2 ;
        }
        cg_step (xtemp, x, d, *alpha, n) ;
        cg_grad (gtemp, xtemp) ;
        Parm->ng++ ;
        *dphi = cg_dot (gtemp, d, n) ;
        *phi = cg_value (xtemp) ;
        Parm->nf++ ;
        if ( Parm->PrintLevel )
        {
            printf ("contract, a: %14.6e alpha: %14.6e "
                    "phi: %14.6e dphi: %14.6e\n", *a, *alpha, *phi, *dphi) ;
        }
        if ( cg_Wolfe (*alpha, *phi, *dphi, Parm) )
        {
            status = 0 ;
            goto Exit2 ;
        }
        if ( *dphi >= zero )
        {
            *b = *alpha ;
            *dphib = *dphi ;
            goto Exit1 ;
        }
        if ( *phi <= Parm->fpert )
        {
            if ( Parm->PrintLevel )
            {
                printf ("updata a: %14.6e dphia: %14.6e\n", *alpha, *dphi) ;
            }
            *a = *alpha ;
            *dphia = *dphi ;
        }
        else *b = *alpha ;
    }
Exit1:
    status = -1 ;
Exit2:
    if ( Parm->PrintLevel )
    {
        printf ("UP a: %14.6e b: %14.6e da: %14.6e db: %14.6e status: %i\n",
                 *a, *b, *dphia, *dphib, status) ;
    }
    return (status) ;
}

/* This routine is identical to cg_update except that the function
   psi [a] = phi [a] - phi [0] - a*delta*dphi [0] is minimized instead of
   the function phi. The return int has the following meaning:
                   8 if too many iterations
                   0 if Wolfe condition is satisfied
                  -1 if interval is updated and a search is done
                  -2 if the interval updated successfully */

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
)
{
    double zero, psi, cg_dot (double *, double *, int) ;
    int n, nshrink, status, cg_Wolfe (double, double, double, cg_parameter *) ;

    zero = 0. ;
    n = Parm->n ;
    cg_step (xtemp, x, d, *alpha, n) ;
    *phi = cg_value (xtemp) ;
    psi = *phi - *alpha*Parm->wolfe_hi ;
    Parm->nf++ ;
    cg_grad (gtemp, xtemp) ;
    Parm->ng++ ;
    *dphi = cg_dot (gtemp, d, n) ;
    *dpsi = *dphi - Parm->wolfe_hi ;
    if ( Parm->PrintLevel )
    {
        printf ("update alpha: %14.6e psi: %14.6e dpsi: %14.6e\n",
                 *alpha, psi, *dpsi) ;
    }
    if ( cg_Wolfe (*alpha, *phi, *dphi, Parm) )
    {
        status = 0 ;
        goto Exit2 ;
    }
    status = -2 ;
    if ( *dpsi >= zero )
    {
        *b = *alpha ;
        *dpsib = *dpsi ;
        goto Exit2 ;
    }
    else
    {
        if ( psi <= Parm->fpert )
        {
            *a = *alpha ;
            *dpsia = *dpsi ;
            goto Exit2 ;
        }
    }
    nshrink = 0 ;
    *b = *alpha ;
    while ( TRUE )
    {
        *alpha = .5*(*a + *b) ;
        nshrink++ ;
        if ( nshrink > Parm->nexpand )
        {
            status = 8 ;
            goto Exit2 ;
        }
        cg_step (xtemp, x, d, *alpha, n) ;
        cg_grad (gtemp, xtemp) ;
        Parm->ng++ ;
        *dphi = cg_dot (gtemp, d, n) ;
        *dpsi = *dphi - Parm->wolfe_hi ;
        *phi = cg_value (xtemp) ;
        psi = *phi - *alpha*Parm->wolfe_hi ;
        Parm->nf++ ;
        if ( Parm->PrintLevel )
        {
            printf ("contract, a: %14.6e alpha: %14.6e "
                    "phi: %14.6e dphi: %14.6e\n", *a, *alpha, *phi, *dphi) ;
        }
        if ( cg_Wolfe (*alpha, *phi, *dphi, Parm) )
        {
            status = 0 ;
            goto Exit2 ;
        }
        if ( *dpsi >= zero )
        {
            *b = *alpha ;
            *dpsib = *dpsi ;
            goto Exit1 ;
        }
        if ( psi <= Parm->fpert )
        {
            if ( Parm->PrintLevel )
            {
                printf ("updata a: %14.6e dpsia: %14.6e\n", *alpha, *dpsi) ;
            }
            *a = *alpha ;
            *dpsia = *dpsi ;
        }
        else *b = *alpha ;
    }
Exit1:
    status = -1 ;
Exit2:
    if ( Parm->PrintLevel )
    {
        printf ("UP a: %14.6e b: %14.6e da: %14.6e db: %14.6e status: %i\n",
                 *a, *b, *dpsia, *dpsib, status) ;
    }
    return (status) ;
}

/*
Version 1.2 Change:
  1. The variable dpsi needs to be included in the argument list for
     subroutine cg_updateW (update of a Wolfe line search)
*/
