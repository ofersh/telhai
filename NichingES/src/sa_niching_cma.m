% Self-Adaptive ES Niching with (1,lambda)-CMA.
% -----------------------------------------------------------------------------------
% Niching CMA-ES: Dynamic Self-Adaptive Niche Radius Niching with Covariance Matrix
% Adaptation ES for nonlinear function multimodal maximization.
% To be used under the terms of the GNU General Public License:
% http://www.gnu.org/copyleft/gpl.html
% CMA-ES implementation is based on Hansen's code:
% http://www.icos.ethz.ch/software/evolutionary_computation/cma
%
% Author: Ofer M. Shir, 2006. e-mail: oshir@liacs.nl
% Reference: PPSN-2006, Iceland; Springer.
% http://www.liacs.nl/~oshir/
% -----------------------------------------------------------------------------------

function [X,mpr_q,rho_stat] = sa_niching_cma(bnf,N,X_a,X_b,...
    q,q_eff,rho,kappa,co_sigma,NEC,delta,R_min,R_max);
close all;
strfitnessfct = 'benchmark_func';
global initial_flag;
initial_flag=0;

%Strategy parameter setting: Selection
lambda=10; mu=1; weights = ones(mu,1);
mueff=sum(weights)^2/sum(weights.^2); % variance-effective size of mu
weights = weights/sum(weights);     % normalize recombination weights array

% Niching Parameters.
% R_max = sqrt(N*((X_b-X_a)^2))/(2);
% R_min = R_max/(N*q);
x_rho = rho * ones(1,q_eff); %R_max * rand(1,q_eff);
y_rho = rho * ones(1,lambda*q_eff); %R_max * rand(1,lambda*q_eff);

% Strategy parameter setting: Adaptation
cc = 4/(N+4);    % time constant for cumulation for covariance matrix
cs = (mueff+2)/(N+mueff+3); % t-const for cumulation for sigma control
mucov = mueff;   % size of mu used for calculating learning rate ccov
ccov = (1/mucov) * 2/(N+1.4)^2 + (1-1/mucov) * ...  % learning rate for
    ((2*mueff-1)/((N+2)^2+2*mueff));             % covariance matrix
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma
chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  

X = (X_b-X_a)*rand(N,q_eff) + X_a; %decision parameters to be optimized.
Y = zeros(N,q_eff*lambda); %temporary DB for offspring.
P = zeros(1,q_eff*lambda); %Parents indices.

% Initialize dynamic (internal) strategy parameters and constants
for i=1:q_eff,
    [sigma{i},pc{i},ps{i},B{i},D{i},C{i},counteval{i},ds{i}] = init_cma(N,co_sigma);
end

gen = 0;
global_eval = 0;
arfitness = zeros(1,q_eff*lambda);
%best = zeros(N+1,q);

out = 100;
MAX_EVAL = q*NEC;
MAX_GENERATIONS = ceil(MAX_EVAL/(q_eff*lambda));
rho_stat = zeros(q,MAX_GENERATIONS);
mpr_q = zeros(q,MAX_GENERATIONS);

% -------------------- Generation Loop --------------------------------
while global_eval < MAX_EVAL
    arz = randn(N,q_eff*lambda); % array of normally distributed r.v.
    for k=1:q_eff*lambda,
        parent = ceil(k/lambda);
        Y(:,k) = X(:,parent) + sigma{parent}*(B{parent}*...
            D{parent}*arz(:,k)); % mutation
        counteval{parent} = counteval{parent}+1;
        P(1,k) = parent;
        c_rho = 0.2*(1-exp((delta)*ds{parent}));
        y_rho(1,k) = (1-c_rho)*x_rho(1,parent) + c_rho*sqrt(N)*sigma{parent};
    end

    %Keeping every rho in the interval [R_min,R_max]:
    if ((sum(y_rho(:,:) > R_max) > 0)||(sum(y_rho(:,:) < R_min) > 0))
        y_rho(:,:) = y_rho(:,:).*(y_rho(:,:) >= R_min).*(y_rho(:,:) <= R_max) +...
                R_min*(y_rho(:,:) < R_min) + R_max*(y_rho(:,:) > R_max);
    end

    %Periodic Boundary Conditions - Let us keep X in the interval [X_a,X_b]:
    if ((sum(sum(Y(:,:) < X_a)) > 0) || (sum(sum(Y(:,:) > X_b)) > 0))
        Y(:,:) = Y(:,:).*(Y(:,:) >= X_a).*(Y(:,:) <= X_b) + X_a*(Y(:,:) < X_a) + X_b*(Y(:,:) > X_b);
    end

    % Fitness evaluation + sorting
    arfitness(:) = -(feval (strfitnessfct,Y(:,:)',bnf))'; % M I N U S   S I G N
    [arfitness, arindex] = sort(arfitness,2,'descend'); %  maximization
    global_eval = global_eval + size(Y,2);

    Y = Y(:,arindex); % Decision+Strategy parameters are now sorted!
    arz = arz(:,arindex);
    P = P(:,arindex);
    y_rho = y_rho(:,arindex);

    %Dynamic Peak Identification
    [DPS,pop_niche,M,niche_count] = DPI (Y(:,:),y_rho,q,lambda);
    F_SH = arfitness./M;
    %(1,lambda) Selection for each niche
    for i=1:q,
        j=DPS(1,i);
        if (j~=0)
            k=find(max(F_SH(find(pop_niche==i)))==F_SH);
            if (size(k,2)>1)
                for l=k,
                    if (pop_niche(l)==i)
                        k=l;
                        break;
                    end
                end
            end
            X(:,i) = Y(:,k);
            x_rho(1,i) = y_rho(1,k); %index j is crucial.
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},new_C{i},new_counteval{i}] = ...
                cma_adapt(sigma,pc,ps,B,D,C,counteval,P(1,k),arz(:,k),N,cs,cc,ccov,chiN,damps,mucov,weights,lambda);
            ds{i} = abs(sigma{P(1,k)} - new_sigma{i});
        else
            X(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            x_rho(1,i) = rho;
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},...
                new_C{i},new_counteval{i},ds{i}] = init_cma(N,co_sigma);
        end
    end
    if (mod(gen,kappa)==0)
        for i=q+1:q_eff,
            X(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            x_rho(1,i) = rho; %R_max * rand(1,1);
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},...
                new_C{i},new_counteval{i},ds{i}] = init_cma(N,co_sigma);
        end
    end
    sigma = new_sigma;
    pc = new_pc;
    ps = new_ps;
    B = new_B;
    C = new_C;
    D = new_D;
    counteval = new_counteval;
    MX = -(feval(strfitnessfct,X(:,1:q)',bnf))'; %M I N U S   S I G N 

    mpr_q(:,gen+1) = -MX;%1./(1+abs(MX(:,:)'));
    %%Output
%             if (mod(gen,out)==0)
%                 disp([num2str(gen) ': ' num2str(-MX(:,:))]);
%              end

    rho_stat(:,gen+1) = x_rho(1,1:q)';
    gen = gen + 1;
end

X = X(:,1:q);
%best = -MX;
%[best(N+1,:), arindex] = sort(best(N+1,:),2,'descend'); % maximization
%best(1:N,:) = best(1:N,arindex);

disp([num2str(gen) ': ' num2str(-MX(:,:))]);

% For the GAS/Vincent functions only:
% figure;
% y=[0:0.01:10];
% plot(y,Shekel(y));
% hold on;
% plot (X,MX,'k.');
% hold off;
% axis([0.1 10 -1 1.2]);

end

%--------------------------------------------------------------------------
function[sigma,pc,ps,B,D,C,counteval,ds] = init_cma(N,co_sigma);
pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma
B = eye(N);                         % B defines the coordinate system
D = eye(N);                         % diagonal matrix D defines the scaling
C = B*D*(B*D)';                     % covariance matrix
sigma = co_sigma;
counteval = 0;
ds = 0;
end
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
[new_B,new_D] = eig(new_C);       % eigen decomposition, B==normalized eigenvectors
new_D = abs(diag(sqrt(diag(new_D)))); % D contains standard deviations now
new_counteval = counteval{parent};
end
%--------------------------------------------------------------------------
function [DPS,pop_niche,M,niche_count] = DPI (Y,y_rho,q,lambda);
N = size(Y,1);
neighbour_factor = 1;
psize = size(Y,2);
DPS = zeros(1,q); %Dynamic Peak Set.
pop_niche = zeros(1,psize); %The classification of each individual to a niche; zero is "non-peak" domain.
Num_Peaks = 1;
niche_count = zeros(1,q);
DPS(1,1) = 1;
niche_count(1,1) = 1;
pop_niche(1,1) = 1;
Euc_D = zeros(psize,psize);
for i=1:psize,
    for j=i+1:psize,
        Euc_D(i,j) = norm(Y(:,i)-Y(:,j));
        Euc_D(j,i) = Euc_D(i,j);
    end
end
d_min = sort(Euc_D,2);
for k=2:psize,
    assign = 0;
    for j=1:Num_Peaks,
        d_pi = norm(Y(:,k) - Y(:,DPS(1,j)));
        if ((d_pi < y_rho(1,DPS(1,j)))||(d_pi<neighbour_factor*d_min(DPS(1,j),2)))
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

alpha = 1; %/lambda;
beta = 1/(lambda);

Theta_sh = Euc_D < repmat(y_rho,psize,1);
M = sum(Theta_sh,1);
for i=find(pop_niche~=(q+1)),
    M(1,i) = niche_count(1,pop_niche(1,i));
end
M = ones(1,psize) + (M > lambda).*(alpha*(M-lambda).^2); + ...
    (M < lambda).*(beta*(M-lambda).^2);
end