function [psi_hat,w_T,A_T,y_hat,G,logL,iter] = EM(psi,y,X,DistMat,exit_tolerance,max_iterations,estimate_mu,verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Input arguments        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%psi:               model parameter starting values - [data structure]
%y:                 observed data - [N x 1 vector]
%X:                 covariates - [N x b matrix]
%DistMat:           distance matrix - [N x N matrix]
%exit_tolerance:    exit condition of the EM algorithm - [scalar > 0]
%max_iterations:    number of maximum iterations of the EM algorithm - [scalar > 0]
%estimate_mu:       if 1 estimate the mu parameter - [boolean]
%verbose:           if 1 details are printed during the EM computation - [boolean]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Output arguments       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%psi_hat:           estimated model parameters - [data structure]
%w_T:               estimated w variable - [N x 1 vector]
%A_T:               variance of w - [N x N matrix]
%y_hat:             estimated y (NaN filled) - [N x 1 vector]
%G:                 estimated G matrix - [N x N matrix]
%logL:              marginal log-likelihood - [scalar]
%iter:              EM iterations executed - [scalar]

%input arguments check
if nargin<4
    error('Not enough input arguments');
end

if nargin<5
    exit_tolerance=0.0001;
    warning(['An exit tolerance equal to ',num2str(exit_tolerance),' is considered']);
end

if nargin<6
    max_iterations=500;
    warning(['The maximum number of EM iterations is ',num2str(max_iterations)]);
end

if nargin<7
    verbose=1;
end

if nargin>=7
    if isempty(exit_tolerance)
        exit_tolerance=0.0001;
        warning(['An exit tolerance equal to ',num2str(exit_tolerance),' is considered']);
    end
    if isempty(max_iterations)
        max_iterations=500;
        warning(['The maximum number of EM iterations is ',num2str(max_iterations)]);
    end
    if isempty(verbose)
        verbose=1;
    end
end

if exit_tolerance<=0
    error('The exit tolerance must be > 0');
end
if max_iterations<=0
    error('The number of maximum iterations must be > 0');
end
if not(verbose==0 || verbose==1)
    error('verbose must be either 0 or 1');
end

if not(isempty(X))
    if not(size(y,1)==size(X,1))
        error('y and X must have the same number of rows');
    end
end

if not(size(DistMat,1)==size(DistMat,2))
    error('DistMat must be a square matrix');
end

if not(size(y,1)==size(DistMat,1))
    error('y and DistMat must have the same number of rows');
end

if not(isempty(X))
    if not(length(psi.beta)==size(X,2))
        error('The number of columns of X must be equal to the number of elements of psi.beta');
    end
end

psi_vec=get_psi_vec(psi);
psi_vec_last=psi_vec*0;
iter=0;
L=not(isnan(y));
N=length(y);
one_vec=ones(N,1);

%EM iterations
disp('***********************************************************');
disp('EM estimation started...');
ct1=clock;
while (norm(psi_vec-psi_vec_last)/norm(psi_vec)>exit_tolerance)&&(iter<max_iterations)
    if verbose
        if not(isempty(X))
            disp(['Iteration ',num2str(iter),' - Norm: ',num2str(norm(psi_vec-psi_vec_last)/norm(psi_vec)),' mu: ',num2str(psi.mu),' beta: ',num2str(psi.beta'),' Sigma_eps: ',num2str(psi.sigma_eps),' Gamma: ',num2str(psi.gamma),' Theta: ',num2str(psi.theta),' Phi: ',num2str(psi.phi)])
        else
            disp(['Iteration ',num2str(iter),' - Norm: ',num2str(norm(psi_vec-psi_vec_last)/norm(psi_vec)),' mu: ',num2str(psi.mu),' Sigma_eps: ',num2str(psi.sigma_eps),' Gamma: ',num2str(psi.gamma),' Theta: ',num2str(psi.theta),' Phi: ',num2str(psi.phi)])
        end
    end
    G=get_G_matrix(DistMat,psi.phi);
    g=diag(G);
    g_inv=1./g;
    Sigma_eps=eye(N)*psi.sigma_eps;
    Sigma_w=exp(-DistMat/psi.theta);
    Sigma_y=(g*g').*(psi.gamma^2*Sigma_w)+diag(g.^2*psi.sigma_eps);
    Sigma_wy=psi.gamma*Sigma_w*G';
    
    %E-step
    if not(isempty(X))
        w_T=Sigma_wy(:,L)*(Sigma_y(L,L)\(y(L)-g(L).*(psi.mu*one_vec(L)+X(L,:)*psi.beta)));
    else
        w_T=Sigma_wy(:,L)*(Sigma_y(L,L)\(y(L)-g(L).*(psi.mu*one_vec(L))));
    end
    A_T=Sigma_w-(Sigma_wy(:,L)/Sigma_y(L,L))*Sigma_wy(:,L)';
    temp=zeros(N);
    if not(isempty(X))
        res_T=g_inv(L).*y(L)-psi.mu*one_vec(L)-X(L,:)*psi.beta-psi.gamma*w_T(L);
    else
        res_T=g_inv(L).*y(L)-psi.mu*one_vec(L)-psi.gamma*w_T(L);
    end
    
    %M-step
    temp(L,L)=res_T*res_T'+psi.gamma^2*A_T(L,L);
    temp(~L,~L)=Sigma_eps(~L,~L);
    psi_temp.sigma_eps=trace(temp)/N;
    if not(isempty(X))
        psi_temp.beta=((X(L,:)'*X(L,:))\X(L,:)')*(res_T+X(L,:)*psi.beta);
    else
        psi_temp.beta=[];
    end
    if estimate_mu
        if not(isempty(X))
            res_T=g_inv(L).*y(L)-X(L,:)*psi_temp.beta-psi.gamma*w_T(L);
        else
            res_T=g_inv(L).*y(L)-psi.gamma*w_T(L);
        end
        psi_temp.mu=trace(res_T*one_vec(L)')/sum(L);
    else
        psi_temp.mu=psi.mu;
    end
    if not(isempty(X))
        res_T=g_inv(L).*y(L)-psi_temp.mu*one_vec(L)-X(L,:)*psi_temp.beta;
    else
        res_T=g_inv(L).*y(L)-psi_temp.mu*one_vec(L);
    end
    m2=w_T*w_T'+A_T;
    psi_temp.gamma=trace(res_T*w_T(L)')/trace(m2(L,L));
    psi_temp.theta=exp(fminsearch(@(x) geo_function(x,DistMat,m2),log(psi.theta),optimset('MaxIter',100,'TolX',1e-4)));
    if not(isempty(X))
        res=one_vec*psi_temp.mu+X*psi_temp.beta+psi_temp.gamma*w_T;
    else
        res=one_vec*psi_temp.mu+psi_temp.gamma*w_T;
    end
    Omega=psi_temp.gamma^2*A_T;
    min_result = fminsearch(@(x) potential_function(x,y,DistMat,res,Omega,psi_temp.sigma_eps),log(psi.phi),optimset('MaxIter',100,'TolX',1e-4));
    psi_temp.phi=exp(min_result(1));
    psi_temp.alpha=1;
    
    psi_vec_last=get_psi_vec(psi);
    psi=psi_temp;
    
    Sigma = psi.gamma^2*(g(L)*g(L)').*Sigma_w(L,L)+diag(g(L).^2)*psi.sigma_eps;
    if not(isempty(X))
        eta = y(L)-g(L).*(psi.mu*one_vec(L)+X(L,:)*psi.beta);
    else
        eta = y(L)-g(L).*(psi.mu*one_vec(L));
    end
    logL = sum(log(eig(Sigma)))+eta'*(Sigma\eta);
    if verbose
        disp(['Marginal logL: ',num2str(logL)]);
    end
    psi_vec=get_psi_vec(psi);
    iter=iter+1;
end
psi_hat=psi;

%y_hat computation
if not(isempty(X))
    y_hat=G*(X*psi.beta+one_vec*psi.mu+psi.gamma*w_T);
else
    y_hat=G*(one_vec*psi.mu+psi.gamma*w_T);
end
ct2=clock;
disp(['EM estimation ended after ',num2str(iter),' iterations']);
disp(['Computation time: ',computation_time(etime(ct2,ct1))]);
disp(['Marginal log-likelihood: ',num2str(logL)]);
disp('');
disp('Estimated model parameters: ');
disp(['    mu       : ',num2str(psi.mu)]);
if not(isempty(psi.beta))
    disp(['    beta     : ',num2str(psi.beta')]);
end
disp(['    sigma_esp: ',num2str(psi.sigma_eps)]);
disp(['    gamma    : ',num2str(psi.gamma)]);
disp(['    theta    : ',num2str(psi.theta),' m']);
disp(['    phi      : ',num2str(psi.phi),' m']);
disp('');
disp('***********************************************************');
end