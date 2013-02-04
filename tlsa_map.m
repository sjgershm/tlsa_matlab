function [F df] = tlsa_map(mapfun,omega,R,F,df)
    
    % Construct basis images from source parameters.
    %
    % USAGE: [F df] = tlsa_map(mapfun,omega,R)
    %
    % INPUTS:
    %   mapfun - function handle for source mapping function
    %   omega - [K x M] matrix of source parameters
    %   R - [V x D] feature locations
    %
    % OUTPUTS:
    %   F - [K x V] basis images
    %   df - [K x 1] cell array of [M x V] partial derivative matrices
    %
    % Sam Gershman, Jun 2011
    
    V = size(R,1);
    K = size(omega,1);
    if nargin < 4
        F = zeros(K,V);
    end
    if nargin < 5 && nargout > 1
        df = cell(1,K);
    end
    
    for k = 1:K
        if nargout > 1
            [F(k,:) df{k}] = mapfun(omega(k,:),R);
        else
            F(k,:) = mapfun(omega(k,:),R);
        end
    end