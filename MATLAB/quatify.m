% This quaitfy method is from Tuttle and Savicci,2016[1].
% This method derived from three steps:
% (1) regression Y = beta*X+c, and got p value.
% attention: this part remove the bootstrap method raised by [1] to save time.
% (2) Use GLM method to seperate dry days and wet days, if anomaly of target day
% is larger than seasonaly anomaly, set as wet day, vice versa.
% (3) Use mean of prediction of full model and baseline model(full model set 
% coefficient of predict variables as 0) to get the qualify value of X-Y
% feedback.

function [p_value,impact,varargout] = quatify(X_predict,Y_predict,...
                                    X,Y,...
                                    season_anomaly,...
                                    type)
% Paramters:
% _________
% - X_predict,Y_predict:
% - X,Y:
% - season_anomaly:
% - type:
%
% Attributes:
% __________
% - impact:
% - p_value:
          
% time length                        
N = numel(Y);
% regression of POCC residual = beta*S residual+c.
one = ones(N,1);    %constant term in regression
switch type
    
%
case 'linear'
% transform x residual got from best model to offset of GLM.
% random forest changed to glm method, need to set down the offsetR
offsetR = norminv(X_predict);
% if x residual close to 0,corresponding offset value close to +-inf.
% thus this phenomenen should be exclude.
offsetR(find(offsetR<-1000|offsetR>1000)) = NaN; 
%
try  
    [b,~,stats] = glmfit([one,X-X_predict],Y,...
                         'binomial',...
                         'link','probit',...
                         'estdisp','on',...
                         'constant','off',...
                         'offset',offsetR);
    % dispersion
    dispsave = stats.sfit(end);    
    %coefficient significance
    p_value = stats.p(2);   
catch Wrong    
    disp('error in unrestrict, uncorrected offset model')
    dispsave = NaN; 
    b([1,end-4:end]) = nan; 
    p_value = NaN;
end
% use predict value of restricted model/full model to qualify S-POCC impact.
if ~isnan(b(2))
    %
    PNaN = X.*0;
    % predictions from unrestricted model - 
    % all coefficients corrected for bias
    predU_all = normcdf(1.*offsetR + ...
                       (squeeze(b)'*[one,X-X_predict]')'+...
                       (PNaN),...
                       0,dispsave);
    %predictions from unrestricted model, but with the soil
    %moisture term set to zero - all coefficients corrected for bias
    predU_baseline = normcdf(1.*offsetR + ...
                            (squeeze(b)'*[one,(X-X_predict).*0]')'+...
                            (PNaN),...
                            0,dispsave);
    % 
    IJ = season_anomaly + PNaN - nanmean(season_anomaly+PNaN);
    %
    quants = quantile(IJ,[0.1 0.25 0.5 0.75 0.9]);    
    % mean S impact in bottom/top 50% of S anomaly
    % (dividing predU_all by predU_all_sp0 works because slagperts
    % has a mean of zero, so it does not change the mean of the prediction)
    II = find(IJ < quants(ceil(length(quants)/2)));
    impact_dry = nanmean(predU_all(II))./nanmean(predU_baseline(II));
    %
    II = find(IJ > quants(ceil(length(quants)/2)));
    impact_wet = nanmean(predU_all(II))./nanmean(predU_baseline(II));
    %save mean S impact in bottom/top 50% of S anomaly
    impact = [impact_dry,impact_wet];
else   
    p_value = NaN;
    impact = nan(2,1);
end

%
case 'nonlinear'
    %
    model = models();
    [~,resid] = model.RF([one,X-X_predict],Y-Y_predict);
    predU_all = Y-Y_predict-resid;
    [~,resid] = model.RF(X-X_predict,Y-Y_predict);
    predU_baseline = Y-Y_predict-resid;
    % 
    IJ = season_anomaly + PNaN - nanmean(season_anomaly+PNaN);
    %
    quants = quantile(IJ,[0.1 0.25 0.5 0.75 0.9]); 
    %
    II = find(IJ < quants(ceil(length(quants)/2)));
    impact_dry = nanmean(predU_all(II))./nanmean(predU_baseline(II));
    %
    II = find(IJ > quants(ceil(length(quants)/2)));
    impact_wet = nanmean(predU_all(II))./nanmean(predU_baseline(II));
    %save mean S impact in bottom/top 50% of S anomaly
    impact = [impact_dry,impact_wet]; 
    p_value = nan;
end

if nargout > 2; varargout{1} = predU_all; end
if nargout > 3; varargout{2} = predU_baseline; end

end


