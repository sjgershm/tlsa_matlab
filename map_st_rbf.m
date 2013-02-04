function [f df] = map_st_rbf(omega,R)
    
    % Mapping function for spatiotemporal radial basis function.
    %
    % This function assumes that that the parameters are defined over the entire real line, and then
    % uses a logistic sigmoid transformation to map them into [0,1].
    %
    % USAGE: [f df] = map_st_rbf(omega,R)
    %
    % INPUTS:
    %   omega = [mu_s lambda_s mu_t lambda_t], where mu_s is a [1 x D-1] vector specifying the
    %           source center, and lambda_s is a scalar width parameter,
    %           mu_t is the temporal center and lambda_t its width
    %   R - [V x D] feature locations; the last dimension is time
    %
    % OUTPUTS:
    %   f - [1 x V] basis image
    %   df - [M x V] matrix of partial derivatives, where M is the number of parameters
    %
    % Sam Gershman, Jun 2011
    
    omega = 1./(1+exp(-omega));
    
    M = length(omega);
    Ds = M-3; % number of spatial dimensions
    mu_s = omega(1:Ds);
    lambda_s = omega(Ds+1);
    mu_t = omega(Ds+2);
    lambda_t = omega(Ds+3);
    
    g_s = bsxfun(@minus,mu_s,R(:,1:Ds));
    h_s = sum(g_s.^2,2);
    g_t = mu_t - R(:,Ds+1);
    h_t = g_t.^2;
    f = exp(-h_s/lambda_s - h_t/lambda_t)';
    
    % derivatives
    if nargout > 1
        V = size(R,1);
        df = zeros(M,V);
        a = -2*f/lambda_s;
        for Ds = 1:Ds
            df(Ds,:) = a.*g_s(:,Ds)';
        end
        df(Ds+1,:) = f.*h_s'./(lambda_s^2);
        df(Ds+2,:) = -2.*f'.*g_t./lambda_t;
        df(Ds+3,:) = f.*h_t'./(lambda_t^2);
        df = bsxfun(@plus,df,(omega.*(1-omega))');
    end