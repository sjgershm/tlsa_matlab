function [stat con] = tlsa_linear_contrast(results,eta,data,threshold)
    
    % Test a linear contrast.
    %
    % USAGE: [stat con] = tlsa_linear_contrast(results,eta,[data],[threshold])
    %
    % INPUTS:
    %   results - structure containing fitted model
    %   eta - [C x 1] contrast vector
    %   data (optional) - [1 x S] subject data (required to compute the contrast image)
    %   threshold (optional) - p-value threshold for contrast image (default: 0.05)
    %
    % OUTPUTS:
    %   stat - structure containing information about the t-tests (one test
    %          for each source)
    %   con - [1 x V] contrast image, averaged across subjects
    %
    % Sam Gershman, Oct 2012
    
    for s = 1:length(results.q)
        C(s,:) = eta'*results.q(s).W;
    end
    
    [~,p,~,stat] = ttest(C);
    stat.p = p;
    
    if nargout > 1
        if nargin < 4; threshold = 0.05; end
        ix = stat.p < threshold;
        for s = 1:length(results.q)
            F = tlsa_map(results.opts.mapfun,results.q(s).omega,data(s).R);
            con(s,:) = C(s,ix)*F(ix,:);
        end
        con = mean(con);
    end