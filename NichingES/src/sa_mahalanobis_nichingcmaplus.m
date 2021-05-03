%Self-Adaptive ES Niching with (1+lambda)-CMA.
% -----------------------------------------------------------------------------------
% Niching CMA-ES: Dynamic Self-Adaptive Niche Radius Niching with Covariance Matrix
% Adaptation ES for nonlinear function multimodal maximization.
% To be used under the terms of the GNU General Public License:
% http://www.gnu.org/copyleft/gpl.html
%
% Author: Ofer M. Shir, 2007. e-mail: oshir@liacs.nl
% http://www.liacs.nl/~oshir/
% -----------------------------------------------------------------------------------

function [X,mpr_q,rho_stat] = sa_mahalanobis_nichingcmaplus(bnf,N,X_a,X_b,...
    q,q_eff,rho,kappa,co_sigma,NEC,delta,R_min,R_max);
close all;
strfitnessfct = 'benchmark_func';
global initial_flag;
initial_flag=0;

%Strategy parameter setting: Selection
mu = 1; lambda=10;
rech = 0.8;
%Strategy parameter setting: Adaptation
ptarget = (2/11); pts = (ptarget/(1-ptarget));
pthresh = 0.44;
cc = 2/(N+2);
ccov = 2/(6+N^2);
d = 1+(N/2);
cp = (1/12);

% Niching Parameters.
% R_max = sqrt(N*((X_b-X_a)^2))/(2);
% R_min = R_max/(N*q);
x_rho = rho * ones(1,q_eff); %R_max * rand(1,q_eff);
%y_rho = rho * ones(1,lambda*q_eff); %R_max * rand(1,lambda*q_eff);

%Data-structures
X = (X_b-X_a)*rand(N,q_eff) + X_a; %decision parameters to be optimized.
Y = zeros(N,q_eff*lambda); %temporary DB for offspring.
P = zeros(1,q_eff*lambda); %Parents indices
% Initialize dynamic (internal) strategy parameters and constants
for i=1:q_eff,
    [sigma{i},pc{i},ps{i},B{i},D{i},C{i},counteval{i},ds{i}] = ...
        init_cma(N,co_sigma,ptarget);
end

gen = 0;
global_eval = 0;
arfitness = zeros(1,q_eff*(lambda+1)); %Offspring Fitness
arP = -(feval (strfitnessfct,X(:,:)',bnf))'; %Parents' Fitness
best = zeros(N+1,q);

out = 100;
MAX_EVAL = q*NEC;
MAX_GENERATIONS = ceil(MAX_EVAL/(q_eff*lambda));
rho_stat = zeros(q,MAX_GENERATIONS);
mpr_q = zeros(q,MAX_GENERATIONS);

% -------------------- Generation Loop --------------------------------
while global_eval < MAX_EVAL
    Y = zeros(N,q_eff*(lambda+1)); %temporary DB for offspring.
    P = zeros(1,q_eff*(lambda+1)); %Parents indices
    y_rho = zeros(1,q_eff*(lambda+1));
    arz = randn(N,q_eff*lambda);  % array of normally distributed r.v.
    for k=1:q_eff*lambda,
        parent = ceil(k/lambda);
        Y(:,k) = X(:,parent) + sigma{parent}*(B{parent}*...
            D{parent}*arz(:,k)); % add mutation
        if ((sum(Y(:,k) < X_a) > 0) || (sum(Y(:,k) > X_b) > 0))
            Y(:,k) = Y(:,k).*(Y(:,k) >= X_a).*(Y(:,k) <= X_b) +...
                X_a*(Y(:,k) < X_a) + X_b*(Y(:,k) > X_b);
        end
        arfitness(k) = -feval (strfitnessfct,Y(:,k)',bnf);% M I N U S   S I G N
        global_eval = global_eval + 1;
        counteval{parent} = counteval{parent}+1;
        P(1,k) = parent;
        c_rho = rech*(1-exp((delta)*ds{parent}));
        y_rho(1,k) = (1-c_rho)*x_rho(1,parent) + c_rho*sqrt(N)*sigma{parent};
    end

    %Concatenation of the parents to the DB:
    arz = [arz,zeros(N,q_eff)];
    arfitness(:,q_eff*lambda+1:end) = arP;
    P(:,q_eff*lambda+1:end) = [1:1:q_eff];
    Y(:,q_eff*lambda+1:end) = X(:,:);
    y_rho(:,q_eff*lambda+1:end) = x_rho(:,:);

    %Keeping every rho in the interval [R_min,R_max]:
    if ((sum(y_rho(:,:) > R_max) > 0)||(sum(y_rho(:,:) < R_min) > 0))
        y_rho(:,:) = y_rho(:,:).*(y_rho(:,:) >= R_min).*(y_rho(:,:) <= R_max) +...
                R_min*(y_rho(:,:) < R_min) + R_max*(y_rho(:,:) > R_max);
    end

    % Fitness sorting
    [arfitness, arindex] = sort(arfitness,2,'descend'); %maximization
    Y = Y(:,arindex); % Decision+Strategy parameters are now sorted!
    arz = arz(:,arindex);
    P = P(:,arindex);
    y_rho = y_rho(:,arindex);

    %Dynamic Peak Identification
    [DPS,pop_niche,M,niche_count] = DPI (Y(:,:),B,D,P,y_rho,q,lambda);
    F_SH = arfitness./M;

    %(1+lambda) Selection for each niche
    FitP = -inf*ones(1,q_eff);
    XP = zeros(N,q_eff);
    p_rho = zeros(1,q_eff);
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
            parent = P(1,k); %the original parent!
            XP(:,i) = Y(:,k);
            FitP(1,i) = arfitness(1,k);
            p_rho(1,i) = y_rho(1,k);
            lsucc = (arfitness(1,k) > arP(1,parent)); %Recent Mutation Success
            [new_sigma{i},new_ps{i},new_counteval{i}] = ...
                updateStepSize(sigma,ps,counteval,lsucc,cp,d,pts,parent);
            ds{i} = abs(sigma{P(1,k)} - new_sigma{i});
            if (lsucc)
                [new_pc{i},new_B{i},new_D{i},new_C{i}] = ...
                    updateCov(arz(:,k),C,B,D,pc,new_ps{i},parent,cc,ccov,pthresh);
            else
                new_pc{i} = pc{parent};
                new_B{i} = B{parent};
                new_D{i} = D{parent};
                new_C{i} = C{parent};
            end
        else
            XP(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},new_C{i},...
                new_counteval{i},ds{i}] = init_cma(N,co_sigma,ptarget);
            FitP(1,i) = -feval (strfitnessfct,XP(:,i)',bnf);% M I N U S   S I G N
            p_rho(1,i) = rho;
        end
    end
    if (mod(gen,kappa)==0)
        for i=q+1:q_eff,
            XP(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},...
                new_C{i},new_counteval{i},ds{i}] = init_cma(N,co_sigma,ptarget);
            FitP(1,i) = -feval (strfitnessfct,XP(:,i)',bnf); % M I N U S   S I G N
            p_rho(1,i) = rho;
        end
    end
    sigma = new_sigma;
    pc = new_pc;
    ps = new_ps;
    B = new_B;
    C = new_C;
    D = new_D;
    counteval = new_counteval;
    X = XP;
    arP = FitP;
    x_rho = p_rho;

    mpr_q(:,gen+1) = -arP(:,1:q);
    % Output
%     if (mod(gen,out)==0)
%         disp([num2str(gen) ': ' num2str(-arP(:,1:q))]);
%     end
    rho_stat(:,gen+1) = x_rho(1,1:q)';
    gen = gen + 1;
end

X = X(:,1:q);
[best(N+1,:), arindex] = sort(best(N+1,:),2,'descend'); % maximization
best(1:N,:) = best(1:N,arindex);

disp([num2str(gen) ': ' num2str(-arP(:,1:q))]);
end
%--------------------------------------------------------------------------
function[new_sigma,new_ps,new_counteval] = updateStepSize(sigma,ps,counteval,lsucc,cp,d,pts,parent);
new_ps = (1-cp)*ps{parent} + cp*lsucc;
new_sigma = sigma{parent}*exp((1/d)*(new_ps-(pts*(1-new_ps))));
new_counteval = counteval{parent};
end
%--------------------------------------------------------------------------
function[new_pc,new_B,new_D,new_C] = updateCov(Z,C,B,D,pc,ps,parent,cc,ccov,pthresh);
if (ps < pthresh)
    new_pc = (1-cc)*pc{parent} + sqrt(cc*(2-cc))*(B{parent}*D{parent}*Z);
    new_C = (1-ccov)*C{parent} + ccov*(new_pc*new_pc');
else
    new_pc = (1-cc)*pc{parent};
    new_C = (1-ccov)*C{parent} + ccov*(new_pc*new_pc' + cc*(2-cc)*C{parent});
end
% Update B and D from C
N = size(D,1);
new_C=triu(new_C)+triu(new_C,1)'; % enforce symmetry
[new_B,new_D] = eig(new_C);       % eigen decomposition, B==normalized eigenvectors
new_D(new_D<1E-10)=1E-10; 
if max(diag(new_D)) > 1E14*min(diag(new_D))
    tmp = max(diag(new_D))/1E14 - min(diag(new_D));
	new_C = new_C + tmp*eye(N); new_D = new_D + tmp*eye(N); 
end
new_D = abs(diag(sqrt(diag(new_D)))); % D contains standard deviations now
end
%--------------------------------------------------------------------------
function[sigma,pc,ps,B,D,C,counteval,ds] = init_cma(N,co_sigma,ptarget);
pc = zeros(N,1); ps = ptarget;   % evolution paths for C and sigma
B = eye(N);                         % B defines the coordinate system
D = eye(N);                         % diagonal matrix D defines the scaling
C = B*D*(B*D)';                     % covariance matrix
sigma = co_sigma;
counteval = 0;
ds = 0;
end
%--------------------------------------------------------------------------
function [DPS,pop_niche,M,niche_count] = DPI (Y,U,L,P,y_rho,q,lambda);
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
M_D = zeros(psize,psize);
for i=1:psize,
    B = U{P(i)}; D = L{P(i)};
    SIGMA=B * diag(1./diag(D).^2) * (B');
    for j=1:psize,
        M_D(i,j) = sqrt((Y(:,i)-Y(:,j))' * SIGMA * (Y(:,i)-Y(:,j)));
    end
end
%d_min = sort(Euc_D,2);
for k=2:psize,
    assign = 0;
    for j=1:Num_Peaks,
        B=U{P(DPS(1,j))};
        D=L{P(DPS(1,j))};
        SIGMA=B * diag(1./diag(D).^2) * (B');
        d_pi = (Y(:,DPS(1,j))-Y(:,k))' * SIGMA * (Y(:,DPS(1,j))-Y(:,k));
        if (sqrt(d_pi) < y_rho(1,DPS(1,j)))
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

Theta_sh = M_D < repmat(y_rho,psize,1);
M = sum(Theta_sh,1);
for i=find(pop_niche~=(q+1)),
    M(1,i) = niche_count(1,pop_niche(1,i));
end
M = ones(1,psize) + (M > lambda).*(alpha*(M-lambda).^2); + ...
    (M < lambda).*(beta*(M-lambda).^2);
end
%--------------------------------------------------------------------------