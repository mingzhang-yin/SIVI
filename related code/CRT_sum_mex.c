
#include <mex.h>
#include "cokus.c"
#define RAND_MAX_32 4294967295.0
/* L = CRT_sum_mex(x,r,rand(sum(x),1),max(x));   
 or L = CRT_sum_mex(x,r);   */
        

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    mwSize Lenx;
    mwIndex i, j, ij;
    double *x, *RND, *Lsum, *prob;
    double maxx, r;    
    Lenx = mxGetM(prhs[0])*mxGetN(prhs[0]);
    x = mxGetPr(prhs[0]);
    r = mxGetScalar(prhs[1]);
    
    
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    Lsum = mxGetPr(plhs[0]);
    
    if(nrhs>=3)
        RND = mxGetPr(prhs[2]);
    
    if(nrhs>=4)
        maxx =  mxGetScalar(prhs[3]);
    else
        for(i=0, maxx=0;i<Lenx;i++)
            if (maxx<x[i]) maxx = x[i];
    
    prob = (double *) mxCalloc(maxx, sizeof(double));
    
    for(i=0;i<maxx;i++)
        prob[i] = r/(r+i);
    
    for(ij=0, i=0, Lsum[0]=0;i<Lenx;i++)
        for(j=0;j<x[i];j++) {
        if(nrhs<3) {
            /*if ( ((double) randomMT() / (double) 4294967296.0) <prob[j])     Lsum[0]++;*/
            if  ((double) randomMT() <= prob[j]*RAND_MAX_32)     Lsum[0]++;
            /*if ( (double) rand() <= prob[j]*RAND_MAX)     Lsum[0]++;*/
        }
        else {
            if (RND[ij++]<=prob[j])     Lsum[0]++;
        }
        }
}
