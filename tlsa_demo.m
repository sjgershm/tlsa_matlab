% This script demonstrates how to create and analyze a data set with TLSA

% Parameters of the synthetic data
N = 80;     % number of observations
C = 2;      % number of covariates
K = 10;     % number of latent sources
D = 4;      % number of features (3 spatial dimensions and 1 time dimension)
M = D+2;    % number of parameters per source
S = 2;      % number of subjects
tau = 1;    % noise precision

% TLSA options (missing fields get set to defaults)
opts.mapfun = @(theta,R) map_st_rbf(theta,R);    % mapping function (spatiotemporal RBF)
opts.K = K;
opts.beta = 0.01;  % set to 0 to fit each subject independently

% Create the synthetic data set
[r1 r2 r3 r4] = ndgrid(linspace(0,1,5)');
R = [r1(:) r2(:) r3(:) r4(:)];  % location matrix
omega = randn(K,M);             % source parameters (here each subject has the same parameters)

for s = 1:S
    data(s).X = randn(N,C);     % design matrix
    W = randn(C,K);             % weight matrix 
    data(s).R = R;      
    F = tlsa_map(opts.mapfun,omega,data(s).R);  % basis images
    data(s).Y = normrnd(data(s).X*W*F,sqrt(1/tau));          % neural data
    
    % generate some test data
    testdata(s).X = randn(N,C);
    testdata(s).R = R;
    testdata(s).Y = normrnd(testdata(s).X*W*F,sqrt(1/tau));
end

% run variational expectation-maximization algorithm
results = tlsa_EM(data,opts);

% decode covariates for test data
mu = tlsa_decode_gaussian(data,testdata,results);

% show inferred and ground truth covariates for a single subject
figure;
scatter(testdata(1).X(:),mu{1}(:)); lsline
xlabel('Ground truth covariates','FontSize',15);
ylabel('Decoded covariates','FontSize',15);