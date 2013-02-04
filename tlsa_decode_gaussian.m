function [mu iS] = tlsa_decode_gaussian(data,testdata,results,mu0,iS0)
    
    % Decode continuous covariates from brain images under Gaussian assumptions.
    % If no prior parameters are provided, an empirical prior is constructed using
    % the sample mean and precision from the training data.
    %
    % USAGE: [mu iS] = tlsa_decode_gaussian(data,testdata,results,[mu0],[iS0])
    %
    % INPUTS:
    %   data - [1 x S] training data structure
    %   testdata - [1 x S] test data structure
    %   results - results structure
    %   mu0 (optional) - prior mean
    %   iS0 (optional) - prior precision matrix
    %
    % OUTPUTS:
    %   mu - [1 x S] cell array, where each cell contains the [N x C] posterior
    %        mean covariate vector for each datapoint
    %   iS - [1 x S] cell array, where each cell contains the [C x C] posterior precision matrix
    %
    % Sam Gershman, Oct 2012
    
    for s = 1:length(data)
        
        % if no mean & precision given, estimate empirical (MLE) prior
        if nargin < 4
            [mu0 iS0] = tlsa_empirical_prior(data(s));
        end
        
        q = results.q(s);
        tau = q.rho*q.nu;
        F = tlsa_map(results.opts.mapfun,q.omega,data(s).R);
        A = q.W*F;
        iS{s} = iS0 + tau*(A*A');
        mu{s} = bsxfun(@plus,mu0*iS0,tau*testdata(s).Y*A')/iS{s};
    end
    
end

function [mu iS] = tlsa_empirical_prior(data,diagonalize)
    
    % Constructs an empirical prior for the covariates based on the training data.
    %
    % This prior is equivalent to the maximum likelihood estimates of
    % the mean and covariance matrix of a Gaussian.
    %
    % USAGE: [mu iS] = tlsa_empirical_prior(data,[diagonalize])
    %
    % INPUTS:
    %   data - data structure
    %   diagonalize (optional) - if 1, uses diagonal approximation of the
    %                           precision matrix (default: 1)
    %
    % OUTPUTS:
    %   mu - [1 x C] mean vector
    %   iS - [C x C] precision matrix
    %
    % Sam Gershman, Sep 2011
    
    if nargin < 2 || isempty(diagonalize); diagonalize = 1; end
    
    X = data.X;
    mu = mean(X);
    r = bsxfun(@minus,X,mu);
    Sigma = (r'*r)/size(X,1);
    if diagonalize
        iS = diag(1./(eps+diag(Sigma)));
    else
        iS = inv(Sigma);
    end
end