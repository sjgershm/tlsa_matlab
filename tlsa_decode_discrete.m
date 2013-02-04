function post = tlsa_decode_discrete(testdata,results,xvals,prior)
    
    % Decode discrete covariates from brain images.
    %
    % USAGE: post = tlsa_decode_discrete(testdata,results,xvals,[prior])
    %
    % INPUTS:
    %   testdata - test data structure
    %   results - results data structure
    %   xvals - [I x C] possible covariate vectors, where I is the number
    %           of unique vectors
    %   prior (optional) - [1 x I] vector of prior probabilities (default:
    %                      uniform).
    %
    % OUTPUTS:
    %   post - [1 x S] cell array, where each cell (corresponding to a subject) 
    %          contains a [N x I] matrix of log posterior probabilities
    %
    % Sam Gershman, Sep 2011
    
    %uniform prior by default
    if nargin < 4 || isempty(prior)
        prior = ones(1,size(xvals,1))./size(xvals,1);
    end
    
    for s = 1:length(testdata)
        q = results.q(s);
        tau = q.rho*q.nu;         %estimated precision
        N = size(testdata(s).X,1);
        F = tlsa_map(results.opts.mapfun,q.omega,testdata(s).R);
        y = xvals*q.W*F;
        for n = 1:N
            P = log(prior);
            for j = 1:size(xvals,1)
                res = testdata(s).Y(n,:) - y(j,:);         % residuals
                P(j) = P(j) - 0.5*tau*(res*res');       % log-likelihood
            end
            post{s}(n,:) = P-logsumexp(P,2);
        end
    end
    
end

%############################################

function s = logsumexp(x,dim)
    
    % Returns log(sum(exp(x),dim)) while avoiding numerical underflow.
    %
    % USAGE: s = logsumexp(x,[dim])
    %
    % Default is dim = 1 (columns).
    
    if nargin == 1
        % Determine which dimension sum will use
        dim = find(size(x)~=1,1);
        if isempty(dim), dim = 1; end
    end
    
    % subtract the largest in each column
    y = max(x,[],dim);
    x = bsxfun(@minus,x,y);
    s = y + log(sum(exp(x),dim));
    i = find(~isfinite(y));
    if ~isempty(i)
        s(i) = y(i);
    end
end