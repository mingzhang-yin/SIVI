function [ML,sample,LogLike_iter] =  logistic_reg_Gibbs(Xcovariate,y,PolyaGammaTruncation,Burnin,Collection,CollectionStep,Is_fix_alpha)
if ~exist('Is_fix_alpha','var')
    Is_fix_alpha = false;
end


% %
%
% % IsParfor = false;
% %
% % if ~exist('qbound','var')
% %     qbound = 10^(-10);
% % end
% % if ~exist('PolyaGammaTruncation','var')
% %     PolyaGammaTruncation = 4;
% % end
% % if ~exist('Burnin','var')
% %     Burnin = 200;
% % end
% % if ~exist('Collection','var')
% %     Collection = 200;
% % end
% % if ~exist('CollectionStep','var')
% %     CollectionStep = 50;
% % end
% % if ~exist('PruneIdx_K','var')
% %     PruneIdx_K = [];
% % end
% % if ~exist('IsPlot','var')
% %     IsPlot = true;
% % end
% % if ~exist('dataname','var')
% %     dataname= 'unkown';
% % end
%
%  if ~exist('K_strar','var')
%      K_star= 1;
%  end

%Input:
%feautres: N by V matrix, N is the number of data points, V is the feature
%dimension
%species: data label
%idx: idx.train is the taining data indices, idx.test is the testing data
%indices
%K: truncation level of the gamma process, maximum number of experts
%T0: number of layers (number of criteria of an expert)
%PolyaGammaTruncation: the number of gamma random variables used to
%approximate a Polya-Gamma random variable
%Burnin: number of burnin samples
%Collection: number of samples after burnin
%CollectionStep: collect one sample per every CollectionStep interations
%PruneIdx_K: the set of iterations at which all experts are first activated
%and the experts with zero counts are then deactived
%IsPlot: plot results during MCMC iterations if IsPlot is true

%output:
%Error
%ML, maximum likelihood sample
%sample,  collected MCMC samples

%sum-deep-softplus (SDS) regression
%sum-softplus regression if T=1
%deep-softplus regression if K=1
%softplus regression if K=T=1
%logistic regression if K=T=1 and r=1

%Version 1: (sum-softplus and deep-softplus regressions), March, 2015
%Version 2: (SDS regression), December, 2015
%Version 3: (SDS regression with adaptive tuncation), April, 2016
%Copyright: Mingyuan Zhou, 2016

%% data preparation

%eps=1e-6;


% if IsPlot
%     %figure
% %     Plot_Decision_Boundary = true;
% %     if V==2 && Plot_Decision_Boundary
% %         %plot classificaiton probability map for two dimensional data
% %         temp1 = (max(features(:,1)) - min(features(:,1)));
% %         temp2 = (max(features(:,2)) - min(features(:,2)));
% %         [x1,x2] = meshgrid(min(features(:,1))-temp1/5:temp1/100:max(features(:,1))+temp1/5,...
% %             min(features(:,2))-temp2/5:temp2/100:max(features(:,2))+temp2/5);
% %         xs1 = [x1(:),x2(:)];
% %         xs1 = xs1';
% %         XLIM = [(min(x1(:))),(max(x1(:)))];
% %         YLIM = [(min(x2(:))),(max(x2(:)))];
% %     end
%     
%     if V==2
%         UU=eye(2);
%     else
%         %plot the two-dimensional projections of high dimensional data
%         [UU,SS,VV]=svds(X.train,2);
%         UU = UU*SS;
%     end
%     Colors = {'r-','g-.','b:','m--','c-','k-.'};
% end

%% Initilization
text = [];
fprintf('\n Iteration: ');

Vcovariate = size(Xcovariate,1);
N = size(Xcovariate,2);
DiagIdx = sparse(1:Vcovariate,1:Vcovariate,true);


Beta = zeros(Vcovariate,1);
   

maxIter = Burnin + Collection;
sample.Beta = cell(0);
ML.LogLike = zeros(1,maxIter);


LogLike_iter = zeros(1,maxIter);

%% Gibbs sampling
hyper_a=1e-6;
hyper_b=1e-6;
PSI=Xcovariate'*Beta;
for iter=1:maxIter
    
    if iter>1
        alpha = randg(hyper_a+1/2*ones(Vcovariate,1))./(hyper_b+Beta.^2/2);
        %hyper_b = randg(1e-0+ hyper_a*Vcovariate)./(1e-0+ sum(alpha));
    else
        alpha = 10;
    end
    if Is_fix_alpha == true
        alpha = 0.01;
    end
    
    omega = PolyaGamRnd_Gam(ones(N,1),PSI,PolyaGammaTruncation);
    
    cov_Xt=Xcovariate*sparse(1:N,1:N,omega)*Xcovariate';
    cov_Xt(DiagIdx) = cov_Xt(DiagIdx) + max(alpha,1e-3);
    
    [invchol,errMSG] = chol(cov_Xt);
    count_0=0;
    while errMSG~=0
        %cov_Xt =  nearestSPD(cov_Xt);
        cov_Xt(DiagIdx) =  cov_Xt(DiagIdx) + 10^(count_0-6);
        [invchol,errMSG] = chol(cov_Xt);
        count_0 = count_0+1
        flag=1
    end
    %inverse cholesky of the covariance matrix
    invchol = invchol\speye(Vcovariate);
    
    mu = Xcovariate*(y - 1/2);
    
    Beta= invchol*(randn(Vcovariate,1) + invchol'*mu);
    PSI = Xcovariate'*Beta;
    
    
    %% Calcluate training classification probabilities and loglikelihoods
    
    %class_prob_train = exp(PSI-logOnePlusExp(PSI));
    
    LogLike = mean(y.*PSI-logOnePlusExp(PSI));
    LogLike_iter(iter)=LogLike;
    
    ML.LogLike(iter) = LogLike;
    % combine.LogLike(iter) = mean(sum(Y.*log(min(max(class_prob_combine',realmin),1-realmin)),1));
    
    if mod(iter,100)==0 %IsPlot
        fprintf(repmat('\b',1,length(text)));
        text = sprintf('%d',iter);
        fprintf(text, iter);
        drawnow
    end
    
    if iter==1
        ML.class_log_prob_max = -inf;
        ML.Beta=Beta;
    end
    
    if ML.LogLike(iter) >=ML.class_log_prob_max
        ML.class_log_prob_max = LogLike;
        ML.Beta = Beta;
        ML.omega = omega;
    end
    
    
    if  iter>Burnin && mod(iter,CollectionStep)==0
        sample.Beta{end+1}=Beta;
    end
    
end

