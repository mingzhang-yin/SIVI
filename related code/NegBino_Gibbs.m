function [samples, rmean, pmean,rmedian,pmedian,mu] = NegBino_Gibbs(x, Burnin, Collection, CollectionStep, a_0, b_0, e_0, f_0)
%%
% Matlab code. 
% Gibbs sampling for the negative binomial distribution.
% Input:
% x, a vector of counts
% Burnin, number of samples that would be discarded
% Collection, number of samples after Burnin
% CollectionStep, collect one sample in every CollectionStep samples
%%
% Assumming:
% x_i ~ NegBino(r,p), p ~ Beta(a_0, b_0), r ~ Gamma(e_0, 1/f_0);
% Infer r and p with Gibbs sampling:
% p ~ Beta(a_0 + sum_i x_i, b_0 + N*r);
% ell_i ~ CRT(x_i, r);
% r ~ Gamma(e_0 + \sum_i ell_i, 1/(f_0 - N * log(1-p) ));
%
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
%     and Mixture Modeling,"  arXiv:1209.3442, Sept. 2012.
%
%%
% Coded by Mingyuan Zhou, mingyuan.zhou@duke.edu, 10/12/2012 version.
% http://people.ee.duke.edu/~mz1/
% 
%   Copyright (C) 2012, Mingyuan Zhou.
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
% Figure 1 (a)-(b) of [1] Zhou et al., ICML 2012.
%
% x = [zeros(1,70),ones(1,38),2*ones(1,17),3*ones(1,10),4*ones(1,9),5*ones(1,3),6*ones(1,2),7*ones(1,1)];
% [samples, rmean, pmean,rmedian,pmedian,mu] = NegBino_Gibbs(x);
%%
% Demo 2: 
%
% r=0.3; p=0.7; x = nbinrnd(r,1-p,1,100);
% [samples, rmean, pmean,rmedian,pmedian,mu] = NegBino_Gibbs(x);

if nargin<2
    Burnin = 10000;    Collection = 10000;
end
if nargin<4
    CollectionStep = 5;
end

if nargin<5
    a_0 = 1e-2;    b_0 = 1e-2;     e_0 = 1e-2;    f_0 = 1e-2;
end

samples= zeros(2,Burnin+Collection);

N = length(x);

r=1;
figure;

for iter=1:Burnin+Collection
    p = betarnd(sum(x)+a_0,N*r+b_0);  
    L = CRT_sum(x,r);  
    r = gamrnd(e_0 + L, 1./(f_0-N*log(max(1-p,realmin))));  
    samples(:,iter)=[r;p];
    if mod(iter,100)==0
        subplot(3,1,1);plot(x,'.');title('count data')
        subplot(3,1,2);plot(samples(1,:)); title('r')
        subplot(3,1,3);plot(samples(2,:)); title('p')
        drawnow
    end
end

samples = samples(:,Burnin+1:CollectionStep:end);

rmean = mean(samples(1,1:end));
pmean = mean(samples(2,1:end));
rmedian = median(samples(1,1:end));
pmedian = median(samples(2,1:end));
mu = mean(samples(1,:).*samples(2,:)./(1-samples(2,:)));

figure;
subplot(2,2,1); hist(samples(1,:)); title('r, histogram')
subplot(2,2,2); stem(autocorr(samples(1,:)),'.');  title('r, autocorr')
subplot(2,2,3); hist(samples(2,:));  title('p, histogram')
subplot(2,2,4); stem(autocorr(samples(2,:)),'.');  title('p, autocorr')
disp(['rmean: ', num2str(rmean)])
disp(['pmean: ', num2str(pmean)])
disp(['rmedian: ', num2str(rmedian)])
disp(['pmedian: ', num2str(pmedian)])
disp(['mu: ', num2str(mu)])

function Lsum= CRT_sum(x,r)
Lsum=0;
RND = r./(r+(0:(max(x)-1)));
for i=1:length(x);
    if x(i)>0
        Lsum = Lsum + sum(rand(1,x(i))<=RND(1:x(i)));
    end
end