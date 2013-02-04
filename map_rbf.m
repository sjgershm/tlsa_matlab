function [f df] = map_rbf(omega,R)
    
    % Mapping function for radial basis function.
    %
    % This function assumes that that the parameters are defined over the entire real line, and then
    % uses a logistic sigmoid transformation to map them into [0,1].
    %
    % USAGE: [f df] = map_rbf(omega,R)
    %
    % INPUTS:
    %   omega = [mu lambda], where mu is a [1 x D] vector specifying the
    %           source center, and lambda is a scalar width parameter
    %   R - [V x D] feature locations
    %
    % OUTPUTS:
    %   f - [1 x V] basis image
    %   df - [M x V] matrix of partial derivatives
    %
    % Sam Gershman, June 2011
    
    omega = 1./(1+exp(-omega));
    
    M = length(omega);
    D = M - 1;
    mu = omega(1:D);
    lambda = omega(M);
    V = size(R,1);
    
    g = bsxfun(@minus,mu,R);
    h = sum(g.^2,2);
    f = exp(-h/lambda)';
    
    % derivatives
    if nargout > 1
        df = zeros(M,V);
        a = -2*f/lambda;
        for d = 1:D
            df(d,:) = a.*g(:,d)';
        end
        df(M,:) = f.*h'./(lambda^2);
        df = bsxfun(@plus,df,(omega.*(1-omega))');
    end