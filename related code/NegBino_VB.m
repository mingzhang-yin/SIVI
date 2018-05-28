function [r,p,a_tilde,b_tilde,alpha_tilde,beta_tilde,out] = NegBino_VB(x, MaxIter, Calculate_ELBO)
%%
% Matlab code. 
% Variational Bayes inference for the negative binomial distribution.
% Input:
% x, a vector of counts
%%
% Assumming:
% x_i ~ NegBino(r,p), p ~ Beta(alpha, beta), r ~ Gamma(a, 1/b);
% Infer r and p with Variational Bayes:
%  a_tilde = a + \sum_i mu_L_i;  b_tilde = b - N*mu_log_1_p;
%  alpha_tilde = alpha + sum_i x_i;  beta_tilde = beta + N*mu_r;
%  mu_L_i = \sum_{n=1}^{x_i} exp(mu_log_r)/(n-1+exp(mu_log_r))
%  mu_log_1_p = psi(beta_tilde)-psi(alpha_tilde+beta_tilde);
%  mu_p = alpha_tilde./(alpha_tilde+beta_tilde);
%  mu_r = a_tilde/b_tilde;  mu_log_r = psi(a_tilde) - log(b_tilde);

%%
% If finding this code helpful, please cite one of the following papers:
% 
% [1] M. Zhou, L. Li, D. Dunson and L. Carin, "Lognormal and
%     Gamma Mixed Negative Binomial Regression," in ICML 2012.
%     
% [2] M. Zhou and L. Carin, "Augment-and-Conquer Negative Binomial
%     Processes," in NIPS 2012.
%
% [3] M. Zhou and L. Carin, "Negative Binomial Process Count
%     and Mixture Modeling,"  IEEE TPAMI, Feb. 2015.
%
%%
% Coded by Mingyuan Zhou
% 10/13/2012 first version.
% 11/28/2012 second version. 
% 08/16/2016 current version.
% http://mingyuanzhou.github.io/
% 
%   Copyright (C) 2016, Mingyuan Zhou.
%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 3, or (at your option)
%   any later version.
% 
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
% 
% 
%%
% Demo 1: Red mites on apple leaves
% Table 1 of Bliss, C. I. and Fisher, R. A. Fitting the negative binomial distribution to biological data. Biometrics, 1953.
% Figure 1 (c) of [1] Zhou et al., ICML 2012.
%
% x = [zeros(1,70),ones(1,38),2*ones(1,17),3*ones(1,10),4*ones(1,9),5*ones(1,3),6*ones(1,2),7*ones(1,1)];
% [r,p,a_tilde,b_tilde,alpha_tilde,beta_tilde] = NegBino_VB(x);
%%
% Demo 2: 
%
% r=0.3; p=0.8; x = nbinrnd(r,1-p,1,50);
% [r,p,a_tilde,b_tilde,alpha_tilde,beta_tilde] = NegBino_VB(x);
% 
% r=0.3; p=0.8; x = nbinrnd(r,1-p,1,200);
% [r,p,a_tilde,b_tilde,alpha_tilde,beta_tilde] = NegBino_VB(x);
% 
% r=0.3; p=0.8; x = nbinrnd(r,1-p,1,1000);
% [r,p,a_tilde,b_tilde,alpha_tilde,beta_tilde] = NegBino_VB(x);
if nargin<1
    x = [zeros(1,70),ones(1,38),2*ones(1,17),3*ones(1,10),4*ones(1,9),5*ones(1,3),6*ones(1,2),7*ones(1,1)];
end

if nargin<2
    MaxIter = 200;
end

if nargin<3
    Calculate_ELBO = true;
end

N = length(x);      

%Initialize r and p with Method of Moments
mu_ML = mean(x);
varx = sum((x-mu_ML).^2)/(N-1);
phi_MM = 1/mu_ML*(varx/mu_ML-1);

r = max(min(1/phi_MM,10),0.1);
p = mu_ML/(mu_ML+r);

if Calculate_ELBO == true
    %choose a worse initilization to show how the ELBO improves as the
    %interation increases
    r=r*2;
    p=p/2;
end
   
out =zeros(2,MaxIter);
ELBO = zeros(1,MaxIter);
figure
for iter=1:MaxIter
        if iter==1
            a0 = 1e-6;  b0=1e-6;    alpha0=1e-6;    beta0=1e-6;
            %a0 = 1e-2;  b0=1e-2;    alpha0=1e-2;    beta0=1e-2;            
            a_tilde = a0;   
            b_tilde = b0;
            mu_r = r;   
            mu_log_r = log(r);            
        else            
            a_tilde = a0 + mu_L; 
            b_tilde = b0 - N*mu_log_1_p;           
            mu_r = a_tilde/b_tilde;
            mu_log_r = psi(a_tilde) - log(b_tilde);
        end      

        alpha_tilde = alpha0 + sum(x);
        beta_tilde = beta0 + N*mu_r;
        
        mu_L = CRT_Expectation(x,exp(mu_log_r));

        mu_log_1_p = psi(beta_tilde)-psi(alpha_tilde+beta_tilde);
        mu_p = alpha_tilde./(alpha_tilde+beta_tilde);
        
        if iter>=10 && abs(mu_p-p)/p<1e-6 &&  abs(mu_r-r)/r<1e-6
            p = mu_p;
            r = mu_r;
            disp(['converged in ', num2str(iter), ' steps']);
            break
        end        
        p = mu_p;
        r = mu_r;        

        out(:,iter)=[r;p];
        
        if Calculate_ELBO == true
            mu_log_p = psi(alpha_tilde)-psi(alpha_tilde+beta_tilde);
            Lbound = N*mu_r*mu_log_1_p + mu_log_p*sum(x)-sum(gammaln(x+1)); %+N*(mu_log_r);
            Lbound = Lbound+ a0*log(b0) + (a0-1)*mu_log_r - b0 * mu_r - gammaln(a0);
            Lbound = Lbound - (a_tilde*log(b_tilde) + (a_tilde-1)*mu_log_r - b_tilde * mu_r - gammaln(a_tilde));
            Lbound = Lbound+ (alpha0-1)*mu_log_p + (beta0-1)*mu_log_1_p - betaln(alpha0,beta0);
            Lbound = Lbound- ((alpha_tilde-1)*mu_log_p + (beta_tilde-1)*mu_log_1_p - betaln(alpha_tilde,beta_tilde));
            %rr = gamrnd(a_tilde,1/b_tilde,10000,1);
            rr = gamrnd(a_tilde,1/b_tilde,100000,1);
            for i=1:N
                Lbound  = Lbound + (sum(gammaln(x(i)+rr)-gammaln(rr)))/length(rr);
            end
            ELBO(iter) = Lbound;
        end

        if mod(iter,10)==0
            subplot(4,1,1);plot(x,'.'); title('count data');
            subplot(4,1,2);plot(out(1,1:iter)); title('r');
            subplot(4,1,3);plot(out(2,1:iter)); title('p');
            subplot(4,1,4);plot(ELBO(1:iter)); title('LowerBound');
            drawnow;            
        end        
end
out = out(:,1:iter);
disp(['E[r]= ', num2str(r)]);
disp(['E[p]= ', num2str(p)]);
figure; 
subplot(1,2,1); plot(max(r-1,0):0.01:r+1,gampdf(max(r-1,0):0.01:r+1,a_tilde,1/b_tilde)); title('Q(r)');
subplot(1,2,2); plot(0:0.01:1,betapdf(0:0.01:1,alpha_tilde,beta_tilde)); title('Q(p)');

function LsumE = CRT_Expectation(x,r)
%  Using Corollary A.2 in 
%  [3] M. Zhou and L. Carin, "Negative Binomial Process Count
%     and Mixture Modeling,"  arXiv:1209.3442, Sept. 2012.
%  to calculate \sum_{i=1}^N <L_i> in Equation (30) of 
%  [1] M. Zhou, L. Li, D. Dunson and L. Carin, "Lognormal and
%     Gamma Mixed Negative Binomial Regression," in ICML 2012.
LsumE = r*sum(psi(nonzeros(x)+r)-psi(r));
% LsumE=0;
% Prob = r./(r+(0:(max(x)-1)));
% for i=1:length(x);
%     if x(i)>0
%         LsumE = LsumE + sum(Prob(1:x(i)));
%     end
% end
