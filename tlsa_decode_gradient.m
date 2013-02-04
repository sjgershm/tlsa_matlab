function mu = tlsa_decode_gradient(traindata,testdata,results,logp,lb,ub)
    
    % Use gradient descent to decode covariates with an arbitrary prior.
    %
    % USAGE: mu = tlsa_decode_gradient(traindata,testdata,results,logp,[lb],[ub])
    %
    % INPUTS:
    %   traindata - training data
    %   testdata - test data
    %   results - results structure
    %   logp - prior function handle, example: logp = @(x) moglik(x,traindata(1).X,0.001);
    %   lb, ub (optional) - lower/upper bounds (default: [-inf, inf])
    %
    % OUTPUTS:
    %   mu - cell array, where m{s}(n,:) represents the decoded covariate
    %   for trial n in subject s
    
    % optimization options
    opts = optimset('GradObj','on','display','off','DerivativeCheck','off');
    
    % lower and upper bounds
    if nargin < 5 || isempty(lb); lb = -inf; end
    if nargin < 6 || isempty(ub); ub = inf; end
    
    C = size(traindata(1).X,2);
    if length(lb)==1; lb = lb*ones(1,C); end
    if length(ub)==1; ub = ub*ones(1,C); end 
    
    % loop over subjects
    for s = 1:length(traindata)
        disp(num2str(s));
        q = results.q(s);   % this structure contains the fitted posterior
        
        % precompute some stuff
        tau = q.rho*q.nu;   % noise precision
        F = tlsa_map(results.opts.mapfun,q.omega,traindata(s).R);   % basis images
        A = q.W*F;  % weights x basis images
        
        mu{s} = testdata(s).X;
        for n = 1:size(testdata(s).X,1) % loop over data points
            x0 = logp([]);              % draw a sample from the prior as initialization
            y = testdata(s).Y(n,:);     % neural data
            f = @(x) decode_likfun(x,y,A,tau,logp);
            mu{s}(n,:) = fmincon(f,x0,[],[],[],[],lb,ub,[],opts);   % run bounded optimization
        end
    end
    
end

%-------------------%

function [lik d] = decode_likfun(x,y,A,tau,logp)
    % negative log posterior over covariates
    
    res = y - x*A;                  % residuals
    lik = 0.5*tau*(res*res');       % negative log-likelihood
    d = -tau*res*A';                % gradient
    [likp dp] = logp(x);            % add prior
    lik = lik - likp;
    d = d - dp;
end