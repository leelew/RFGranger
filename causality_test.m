%% Granger causality test for SM-P feedback
% granger causality test[1] apply for X as predict variables, Y as target
% variables to find whether X granger cause Y.

% X must as lagged time series, full model is Y(t) = X(t-1)+...X(t-m)+c, and
% baseline model is Y(t) = c. Thus contrast the performance of two regression 
% above could see the impact of X(t-m). 

% Example: 
% 1. linear granger causality test
%    pass = causality_test(X,Y,1);
% 2. nonlinear granger causality test
%    pass = causality_test(X,Y,2);

% Referrence:
% [1]:

% attention: X,Y must pass the station series test,such as ADF test.
% Copyright(c): Li Lu, 2019

function [pass,R2_full,R2_baseline,varagout] = causality_test(X,Y,type)

model = models;
switch type   
    case 'linear'
        'Need improved'
    case 'nonlinear'
        % use random forest to apply nonlinear granger causality test.
        const = ones(numel(Y),1);
        [R2_full,~] = model.run_models([X,const],Y,3);
        [R2_baseline,~] = model.run_models(const,Y,3);

        if R2_baseline < R2_full
            pass = 1;
        else
            pass = 0;
        end
end

if nargout > 1; varagout{1} = R2_full; end 
if nargout > 2; varagout{2} = R2_baseline; end
end
