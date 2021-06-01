%% Main code for specific SM-P feedback
% This main.m file is apply nonlinear granger causality framework(NGCF) on
% identify surface soil moisture-precipitation feedback, including both sign 
% and pattern distribution.

% Procedure:
% 1. Use get_terms.m to get independent terms, as period terms(i.e., inter-
% annual cycle, seasonal cycle); lagged terms(i.e., lagged P and press); spatial
% terms(i.e., lagged P and press over selected square indicated spatial impact)
%
% 2. Run random forest for P occurrence and surface soil moisture as dependent 
% terms, and independent terms metioned in 1., Addtionally, by using hybrid
% selection feature method to find the 'best' regression(to avoid overfitting 
% in some content). After applying these regression, get the residual value of 
% both surface soil moisture and precipitation.
%
% 3. Then, fit "slagperts" to residual of POCC and "offset", using only S 
% with no P on the previous day, and adding lagged atmospheric pressure as 
% independent variable. 
%
% 4. Calculate S-POCC impacts by dividing the unrestricted model by the re-
% stricted model, plotted against the seasonal S anomaly, and taking the mean 
% above and below the median of Sclim.

%[We all applied bias-corrected random forest regression method in our research
% to remove endogeneity in some extent.]
%
% Paramters:
% _________
% - P,S,press:
% - lon,lat:
% - startDate:
% - nAnnual,nSeasonal
% - day_lag:
% - Slen:
% - pbcrit:
%
% Attributes:
% __________
% - impact:
% - p_value:

function mains = main()
mains.get_sp_impact = @get_sp_impact;
mains.get_sp_terms = @get_sp_terms;
end

function [p_value,impact,varargout] = get_sp_impact...
                                      (P,S,press,...
                                       lon,lat,...
                                       startDate,...
                                       nAnnual,nSeasonal,day_lag,...
                                       Slen,...
                                       pbcrit,...
                                       fun,...
                                       independ_type) 
%                                      
% ***attention: only for S-P feedback and target dataset.                                  
%% class
model = models();
terms = get_terms();
%*** lag_window dayLag = 5:5:25
%% get independent and dependent terms
[POCC,S,independ_terms,seasTerm,annualTerm,lagged_P_terms,lagged_press_terms,depend_P_terms] = get_sp_terms...
                                              (P,S,press,...
                                               lon,lat,...
                                               startDate,...
                                               nAnnual,nSeasonal,day_lag,...
                                               Slen,...
                                               pbcrit,...
                                               independ_type); 
valid = length(find(~isnan(S)==1))>= round(0.1*length(S)) && ...
        sum(POCC)>0 && sum(S)~=0;
% models
if valid 
    try
        if fun==4
            % remove independent terms impact
            [R2P,residP,endogenity_P] = model.run_models...
                                        (independ_terms,POCC,fun,'',0.5,0.5);
            [R2S,residS,endogenity_S] = model.run_models...
                                        (independ_terms,S,fun,'',0.5,0.5);
        elseif fun==3
            [R2P,residP] = model.run_models...
                           (independ_terms,POCC,fun,'',0.5,0.5);
            [R2S,residS] = model.run_models...
                           (independ_terms,S,fun,'',0.5,0.5);
            endogenity_P = nan;
            endogenity_S = nan;
        end
        % causality test
        [pass,R2_full,R2_baseline] = causality_test...
                                     (residS,residP,'nonlinear');
        % quatify   
        season_anomaly = terms.get_season_anomaly(nSeasonal,seasTerm,S,2);
        [p_value,impact,predU_all,predU_baseline] = quatify...
                                                    (S-residS,POCC-residP,...
                                                     S,POCC,...
                                                     season_anomaly,...
                                                     'linear');
    catch
        impact=[nan,nan];p_value=nan; R2P=nan;R2S=nan;residP=nan;residS=nan;
        season_anomaly=nan;pass=nan;R2_full=nan;R2_baseline=nan;
        endogenity_P=nan;endogenity_S=nan;predU_all=nan;predU_baseline=nan;
    end
else
    impact=[nan,nan];p_value=nan; R2P=nan;R2S=nan;residP=nan;residS=nan;
    season_anomaly=nan;pass=nan;R2_full=nan;R2_baseline=nan;
    endogenity_P=nan;endogenity_S=nan;predU_all=nan;predU_baseline=nan;
end
% optional output
if nargout>2; varargout{1} = R2P; end
if nargout>3; varargout{2} = R2S; end
if nargout>4; varargout{3} = [R2_full,R2_baseline]; end
if nargout>5; varargout{4} = [endogenity_P,endogenity_S]; end
if nargout>6; varargout{5} = season_anomaly; end
if nargout>7; varargout{6} = POCC; end
if nargout>8; varargout{7} = S; end
if nargout>9; varargout{8} = independ_terms; end
if nargout>10; varargout{9} = residP; end
if nargout>11; varargout{10} = residS; end
if nargout>12; varargout{11} = seasTerm; end
if nargout>13; varargout{12} = annualTerm; end
if nargout>14; varargout{13} = predU_all; end
if nargout>15; varargout{14} = predU_baseline; end
if nargout>16; varargout{15} = depend_P_terms; end


end

function [POCC,S,independ_terms,varargout] = get_sp_terms...
                                             (P,S,press,...
                                              lon,lat,...
                                              startDate,...
                                              nAnnual,nSeasonal,day_lag,...
                                              Slen,...
                                              pbcrit,...
                                              independ_type)
%% class
terms = get_terms();
%*** lag_window dayLag = 5:5:25
%% get independent and dependent terms
% P
[depend_P_terms,lagged_P_terms,~,spatial_P_terms,annualTerm,seasTerm,~,~] = ...
terms.get_all_terms(P,startDate,lat,lon,Slen,nAnnual,nSeasonal,day_lag);
% press
[~,lagged_press_terms,~,spatial_press_terms] = ...
terms.get_all_terms(press,startDate,lat,lon,Slen,nAnnual,nSeasonal,day_lag);
% get dependent terms             
POCC = 0.*depend_P_terms;
POCC(find(depend_P_terms>pbcrit))=1;
%*** POCC(find(depend_P_terms>0.1 & depend_P_terms<1))=1; % light/heavy rain
[S,~,spatial_S_terms] = ...
terms.get_all_terms(S,startDate,lat,lon,Slen,nAnnual,nSeasonal,day_lag);
% get independent terms
if independ_type==0
    independ_terms = [annualTerm,seasTerm];
elseif independ_type==1
    independ_terms = [annualTerm,seasTerm,...
                      lagged_P_terms,lagged_press_terms];
elseif independ_type==2
    independ_terms = [annualTerm,seasTerm,...
                      lagged_P_terms,lagged_press_terms,...
                      spatial_P_terms,spatial_press_terms];
elseif independ_type==3
    independ_terms = [annualTerm,seasTerm,...
                      lagged_P_terms,lagged_press_terms,...
                      spatial_P_terms,spatial_press_terms,...
                      spatial_S_terms];
end

if nargout>3; varargout{1} = seasTerm; end
if nargout>4; varargout{2} = annualTerm; end
if nargout>5; varargout{3} = lagged_P_terms; end
if nargout>6; varargout{4} = lagged_press_terms; end
if nargout>7; varargout{5} = depend_P_terms; end

%if nargout>7; varargout{5} = lagged_S_terms; end


end
