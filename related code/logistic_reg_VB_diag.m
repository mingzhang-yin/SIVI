function [mu_Beta,Sig_Beta,chol_Sig_Beta,mu_alpha,c_v,d_v] =  logistic_reg_VB_diag(Xcovariate,y,maxIter,Is_fix_alpha)
% %

if ~exist('Is_fix_alpha','var')
    Is_fix_alpha = false;
end

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

text = [];
fprintf('\n Iteration: ');

Vcovariate = size(Xcovariate,1);
N = size(Xcovariate,2);
DiagIdx = sparse(1:Vcovariate,1:Vcovariate,true);


Beta = zeros(Vcovariate,1);


sample.Beta = cell(0);
ML.LogLike = zeros(1,maxIter);


LogLike_iter = zeros(1,maxIter);


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


DiagIdx = sparse(1:Vcovariate,1:Vcovariate,true);





sample.Beta = cell(0);
ML.LogLike = zeros(1,maxIter);


LogLike_iter = zeros(1,maxIter);

%% Gibbs sampling
c0=1e-6;
d0=1e-6;

chol_Beta = 0;

for iter=1:maxIter
    
    %     if iter==1
    %
    %
    %         b_tilde = a0 + b0;
    %         g_tilde = g0;
    %         mu_h = b_tilde/g_tilde;
    %         mu_Psi = randn(N,1);
    %         a_tilde = a0 + sum(X);
    %         h_tilde = b_tilde/g_tilde + sum(log(1+exp(mu_Psi)));
    %         Sig_Beta = eye(P);
    %         mu_Beta = rand(P,1);
    %         mu_r = 100;
    %         Eomega = (y+mu_r).*(tanh(mu_Psi/2)./mu_Psi/2);
    %         mu_phi = 1;
    %         mu_log_r =log(mu_r);
    %         mu_alpha = ones(P,1);
    %         EBetaBetaT = mu_Beta*mu_Beta' + Sig_Beta;
    %         Sig_Psi = 1./(mu_phi + Eomega);
    %         mu_Psi = Sig_Psi.*((y-mu_r)/2 + mu_phi*(X*mu_Beta+BiasTerm));
    %         EPsiTPsi = sum(Sig_Psi + mu_Psi.*mu_Psi);
    %     else
    %         b_tilde = a0 + b0;
    %         g_tilde = g0 + mu_r;
    %     end
    
    
    
    if iter==1
        mu_alpha = 10*ones(Vcovariate,1);
        mu_Beta = rand(Vcovariate,1);
        Sig_Beta = eye(Vcovariate);
        EBetaBetaT = mu_Beta*mu_Beta'+Sig_Beta;
    else
        
        %        alpha = randg(hyper_a+1/2*ones(Vcovariate,1))./(hyper_b+Beta.^2/2);
        %  hyper_b = randg(1e-0+ hyper_a*Vcovariate)./(1e-0+ sum(alpha));
        
        c_v = c0 + 1/2;
        %d_v = d0 + 1/2*(diag(Sig_Beta) + mu_Beta.*mu_Beta);
        d_v = d0 + 1/2*diag(EBetaBetaT);
        mu_alpha = c_v./d_v;
        
    end
    
    if Is_fix_alpha == true
        mu_alpha = 0.01;
    end
    
    
    
    E_PSI = sqrt(sum((Xcovariate'*EBetaBetaT).*(Xcovariate'),2));
    Eomega = tanh(E_PSI/2)./(E_PSI*2);
    
    %     if 0
    %         Sig_Beta = (Xcovariate*sparse(1:N,1:N,Eomega)*Xcovariate' + diag(mu_alpha))\speye(Vcovariate);
    %         mu_Beta = Sig_Beta*(Xcovariate*(y - 1/2));
    %
    %     else
    %         cov_Xt=Xcovariate*sparse(1:N,1:N,Eomega)*Xcovariate';
    %         cov_Xt(DiagIdx) = cov_Xt(DiagIdx) + max(mu_alpha,1e-3);
    %         [invchol,errMSG] = chol(cov_Xt);
    %         count_0=0;
    %         while errMSG~=0
    %             %cov_Xt =  nearestSPD(cov_Xt);
    %             cov_Xt(DiagIdx) =  cov_Xt(DiagIdx) + 10^(count_0-6);
    %             [invchol,errMSG] = chol(cov_Xt);
    %             count_0 = count_0+1
    %             flag=1
    %         end
    %         %inverse cholesky of the covariance matrix
    %         chol_Sig_Beta = invchol\speye(Vcovariate);
    %         Sig_Beta = chol_Sig_Beta*chol_Sig_Beta';
    %         mu_Beta = Sig_Beta*(Xcovariate*(y - 1/2));
    %     end
    
    Sig_Beta = 1./(sum(sparse(1:N,1:N,Eomega)*(Xcovariate.^2)',1)'+mu_alpha);
    for v=1:Vcovariate
        mu_Beta(v) = Sig_Beta(v)* Xcovariate(v,:)*(y-1/2-Eomega.*(Xcovariate'*mu_Beta-Xcovariate(v,:)'*mu_Beta(v)));
    end
    %%%%mu_Beta = Sig_Beta.*sum(Xcovariate'.*bsxfun(@minus,(y-1/2),Eomega.*(Xcovariate'*mu_Beta- bsxfun(@times,Xcovariate', mu_Beta'))),1)';
    
    
    EBetaBetaT = mu_Beta*mu_Beta'+diag(Sig_Beta);
    
    
    %% Calcluate training classification probabilities and loglikelihoods
    
    %class_prob_train = exp(PSI-logOnePlusExp(PSI));
    
    PSI = Xcovariate'*mu_Beta;
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
        ML.mu_Beta=mu_Beta;
    end
    
    if ML.LogLike(iter) >=ML.class_log_prob_max
        ML.class_log_prob_max = LogLike;
        ML.mu_Beta = mu_Beta;
        ML.Sig_Beta = Sig_Beta;
        ML.Eomega = Eomega;
    end
    
    
    %     if  iter>Burnin && mod(iter,CollectionStep)==0
    %         sample.Beta{end+1}=mu_Beta;
    %     end
    
end

if ~exist('chol_Sig_Beta','var')
    chol_Sig_Beta = chol(diag(Sig_Beta));
end
%outPara.c_v=c_v;
%outPara.d_v=d_v;


