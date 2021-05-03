% ES Niching with (1,lambda)-CMA - with Mahalanobis distance metric.
% -------------------------------------------------------------------------------------
% Niching CMA-ES: Dynamic Niching with Covariance Matrix Adaptation ES with Mahalanobis
% distance metric for nonlinear function multimodal maximization.
% To be used under the terms of the GNU General Public License:
% http://www.gnu.org/copyleft/gpl.html
% CMA-ES implementation is based on Hansen's code:
% http://www.icos.ethz.ch/software/evolutionary_computation/cma
%
% Author: Ofer M. Shir, 2007. e-mail: oshir@liacs.nl
% Reference: 
% http://www.liacs.nl/~oshir/
% -----------------------------------------------------------------------------------

function [X,mpr_q] = mahalanobis_nichingcma(strfitnessfct,N,X_a,X_b,q,q_eff,rho,...
    kappa,co_sigma,MAX_EVAL);
close all;
% Strategy parameter setting: Selection
lambda=10; mu=1; weights = ones(mu,1);
mueff=sum(weights)^2/sum(weights.^2); % variance-effective size of mu
weights = weights/sum(weights);     % normalize recombination weights array

%Strategy parameter setting: Adaptation
cc = 4/(N+4);    % time constant for cumulation for covariance matrix
cs = (mueff+2)/(N+mueff+3); % t-const for cumulation for sigma control
mucov = mueff;   % size of mu used for calculating learning rate ccov
ccov = (1/mucov) * 2/(N+1.4)^2 + (1-1/mucov) * ...  % learning rate for
    ((2*mueff-1)/((N+2)^2+2*mueff));             % covariance matrix
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma
chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of

%Data-structures
X = (X_b-X_a)*rand(N,q_eff) + X_a; %decision parameters to be optimized.
Y = zeros(N,q_eff*lambda); %temporary DB for offspring.
P = zeros(1,q_eff*lambda); %Parents indices
% Initialize dynamic (internal) strategy parameters and constants
for i=1:q_eff,
    [sigma{i},pc{i},ps{i},B{i},D{i},C{i},counteval{i}] = init_cma(N,co_sigma);
end

gen = 0;
global_eval = 0;
arfitness = inf*ones(1,q_eff*lambda);

MAX_GENERATIONS = ceil(MAX_EVAL/(q_eff*lambda));
stat = zeros(1,MAX_GENERATIONS);
mpr_q = zeros(q,MAX_GENERATIONS);
% -------------------- Generation Loop --------------------------------
while global_eval < MAX_EVAL
    arz = randn(N,q_eff*lambda);  % array of normally distributed r.v.
    for k=1:q_eff*lambda,
        parent = ceil(k/lambda);
        Y(:,k) = X(:,parent) + sigma{parent}*(B{parent}*...
            D{parent}*arz(:,k)); % add mutation
        counteval{parent} = counteval{parent}+1;
        P(1,k) = parent;
    end

%    Boundary Conditions: May be treated here
    
    % Fitness evaluation + sorting
    arfitness(:) = feval (strfitnessfct,Y(:,:));
    global_eval = global_eval + size(Y,2);
    [arfitness, arindex] = sort(arfitness,2,'ascend'); %  M I N I M I Z A T I O N
    Y = Y(:,arindex); % Decision+Strategy parameters are now sorted!
    arz = arz(:,arindex);
    P = P(:,arindex);

    stat(1,gen+1) = arfitness(1,1);
    MX=zeros(1,q);
    
    %Dynamic Peak Identification
    [DPS,pop_niche] = DPI (Y(:,:),B,D,P,lambda*q_eff,q,rho);
    %(1,lambda) Selection for each niche
    for i=1:q,
        j=DPS(1,i);
        if (j~=0)
            parent = P(1,j); %the original parent!
            X(:,i) = Y(:,j);
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},new_C{i},new_counteval{i}] = ...
                cma_adapt(sigma,pc,ps,B,D,C,counteval,parent,arz(:,j),N,cs,cc,ccov,chiN,damps,mucov,weights,lambda);
        else
            X(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},new_C{i},...
                new_counteval{i}] = init_cma(N,co_sigma);
        end
    end
    if (mod(gen,kappa)==0)
        for i=q+1:q_eff,
            X(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},...
                new_C{i},new_counteval{i}] = init_cma(N,co_sigma);
        end
    end
    sigma = new_sigma;
    pc = new_pc;
    ps = new_ps;
    B = new_B;
    C = new_C;
    D = new_D;
    counteval = new_counteval;
    MX = feval(strfitnessfct,X(:,1:q));
    mpr_q(:,gen+1) = MX;%1./(1+abs(MX(:,:)'));
    gen = gen + 1;
end
X = X(:,1:q);
disp([num2str(gen) ': ' num2str(MX(:,:))]);

%--------------------------------------------------------------------------
function[sigma,pc,ps,B,D,C,counteval] = init_cma(N,co_sigma);
pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma
B = eye(N);                         % B defines the coordinate system
D = eye(N);                         % diagonal matrix D defines the scaling
C = B*D*(B*D)';                     % covariance matrix
sigma = co_sigma;
counteval = 0;

%--------------------------------------------------------------------------
function[new_sigma,new_pc,new_ps,new_B,new_D,new_C,new_counteval] = ...
    cma_adapt(sigma,pc,ps,B,D,C,counteval,parent,Z,N,cs,cc,ccov,chiN,damps,mucov,weights,lambda);
new_ps = (1-cs)*ps{parent} + sqrt(cs*(2-cs)) * (B{parent} * Z);
hsig = norm(new_ps)/sqrt(1-(1-cs)^(2*counteval{parent}/lambda))/chiN...
    < 1.5 + 1/(N+1);
new_pc = (1-cc)*pc{parent} ...
    + hsig * sqrt(cc*(2-cc)) * (B{parent} * D{parent} * Z);

% Adapt covariance matrix C
new_C = (1-ccov) * C{parent} ...                    % regard old matrix
    + ccov * (1/mucov) * (new_pc*new_pc' ...   % plus rank one update
    + (1-hsig) * cc*(2-cc) * C{parent}) ...
    + ccov * (1-1/mucov) ...           % plus rank mu update
    * (B{parent}*D{parent}*Z) ...
    *  diag(weights) * (B{parent}*D{parent}*Z)';

% Adapt step size sigma
new_sigma = sigma{parent} * exp((cs/damps)*(norm(new_ps)/chiN - 1));

% Update B and D from C
new_C=triu(new_C)+triu(new_C,1)'; % enforce symmetry
[new_B,new_D] = eig(new_C);       % eigen decomposition, B==normalized eigenvectors
new_D(new_D<1E-10)=1E-10; 
if max(diag(new_D)) > 1E14*min(diag(new_D))
    tmp = max(diag(new_D))/1E14 - min(diag(new_D));
	new_C = new_C + tmp*eye(N); new_D = new_D + tmp*eye(N); 
end
new_D = abs(diag(sqrt(diag(new_D)))); % D contains standard deviations now
new_counteval = counteval{parent};
%--------------------------------------------------------------------------
function [DPS,pop_niche] = DPI (Y,U,L,P,psize,q,rho);
DPS = zeros(1,q); %Dynamic Peak Set.
pop_niche = zeros(1,psize); %The classification of each individual to a niche; zero is "non-peak" domain.
Num_Peaks = 1;
niche_count = zeros(1,q);
DPS(1,1) = 1;
niche_count(1,1) = 1;
pop_niche(1,1) = 1;

for k=2:psize,
    assign = 0;
    for j=1:Num_Peaks,
        B=U{P(DPS(1,j))};
        D=L{P(DPS(1,j))};
        SIGMA=B * diag(1./diag(D).^2) * (B');
        d_pi = (Y(:,DPS(1,j))-Y(:,k))' * SIGMA * (Y(:,DPS(1,j))-Y(:,k));
        if (sqrt(d_pi) < rho)
            niche_count(1,j) = niche_count(1,j)+1;
            pop_niche(1,k) = j;
            assign = 1;
            break;
        end
    end
    if ((Num_Peaks<q)&&(assign==0))
        Num_Peaks = Num_Peaks + 1;
        DPS(1,Num_Peaks) = k;
        niche_count(1,Num_Peaks) = 1;
        pop_niche(1,k) = Num_Peaks;
    end
end

pop_niche(find(pop_niche==0)) = q+1;
%--------------------------------------------------------------------------
