%% get_terms
%
% In this module, we get all the terms used for estabilishing regression.
% including lagged terms (account for short-term persistence effect), period
% terms (account for long-term periodicty,i.e.,seasonality and annual cycle
% and trends),spatial terms(account for spatial impact of persistence term),
% and dependent terms.

% Copyright(c) Li Lu, 2019

function term = get_terms

term.get_all_terms = @get_all_terms;

term.get_depend_terms = @get_depend_terms;

term.get_lagged_terms = @get_lagged_terms;
term.get_period_terms = @get_period_terms;
term.get_spatial_terms = @get_spatial_terms;
term.get_extreme_terms = @get_extreme_terms;

term.get_season_anomaly = @get_season_anomaly;

end

function [depend_terms,...
          lagged_terms,...
          period_terms,...
          spatial_terms,...
          varargout] = get_all_terms(data_all,...
                                     startDate,...
                                     lat,lon,...
                                     Slen,...
                                     nAnnual,nSeasonal,day_lag)
                         
% """exclude leap day of leap year
 
% Arguments:
%    data_all (N_time,Nlon,Nlat) -- input dataset
%    startDate -- start date of input dataset
%    lat/lon -- latitude and longitude index
%    Slen -- index for spatial terms, which was constructed 
%            by using data in Slen x Slen square
%    nAnnual,nSeasonal -- index for annual/season terms
%    day_lag -- 

% Returns:
%    varargout: annual_terms --
%               season_terms --
%               unleap_day -- index of unleap day of year.
%               jd -- index of day of year with excluding leap day.
% """
    
% set data
[data_pixel,unleap_day,jd] = remove_leap_day(data_all,startDate,lat,lon);
% get dependent terms
depend_terms = get_depend_terms(data_pixel,day_lag);
% get lagged terms
lagged_terms = get_lagged_terms(data_pixel,day_lag);
% get period terms
[period_terms,annual_terms,season_terms] = get_period_terms...
                                           (data_pixel,...
                                           jd,...
                                           nAnnual,nSeasonal,day_lag);
% get spatial terms
spatial_terms = get_spatial_terms(data_all,Slen,lat,lon,jd,day_lag);
%optional additional outputs
if nargout>4; varargout{1} = annual_terms; end
if nargout>5; varargout{2} = season_terms; end
if nargout>6; varargout{3} = unleap_day; end
if nargout>7; varargout{4} = jd; end
end

function depend_terms = get_depend_terms(data,day_lag)

% """get dependent terms of input array
% ***attention: only use for our dataframe.(see main.m)
% """

depend_terms = data(day_lag+1:end);
end

function [data_,unleap_day,jd] = ...
          remove_leap_day(data,startDate,lat,lon)

% """exclude leap day of leap year
% 
% Returns:
%   data_ -- data with excluding leap day.
%   unleap_day -- index of unleap day of year.
%   jd -- index of day of year with excluding leap day.

% ***attention: only for Date type like 'yyyy-mm-dd'
% """

% specific data
data = data(:,lon,lat);
% Convert date and time to serial date number
startDateNum = datenum(startDate,'yyyy-mm-dd'); 
% serial date number array
v = (startDateNum:(startDateNum+length(data)-1))'; 
% Convert date and time to vector of components
d = datevec(v); 
%day of year corresponding to data
jd = v - datenum(d(:,1), 1,0);
% day of un-leap year
unleap_day = find(jd~=366); 
N = length(unleap_day);
jd = jd(unleap_day);
data_ = data(unleap_day);

end

function lagged_terms = ...
         get_lagged_terms(data,day_lag)

% """use extreme index, spatial homeheterogeneity, surface pressure etc.
% to construct X(t-1),X(t-2),...,X(t-dayLag);"""

% size of input data
[~,y] = size(data);
% loop for 1:day_lag days
for i = 0:day_lag-1
    % if input data only have one feature, i.e.,soil moisture
    if y==1
        lagged_terms(:,y*i+1:y*(i+1)) = data(i+1:end-(day_lag-i));
    % if input data have more than one feature, i.e.,[soil moisture,press,...]
    else
        lagged_terms(:,y*i+1:y*(i+1)) = data(i+1:end-(day_lag-i),:);
    end
end
end

function [period_terms,annualTerm,seasTerm] = ...
         get_period_terms(data,...
                          jd,...
                          nAnnual,nSeasonal,day_lag)
    
% """Construct periodic terms using Fourier series.
% use method from Tuttle and Savicci,2016,Science."""

% size
[N,~] = size(data);
% construct periodic terms varies on annual cycle(18-1.8 year).
annualTerm = zeros(N,nAnnual*2);
d = (1:1:N)';
%for first yrmod term(s), use half sinusoidal cycles 
%(for trend over the 9-yr study period)
for ii = 1
    annualTerm(:,1+(ii-1)*2:ii*2) = [sin(d*(ii)*pi./N),cos(d*(ii)*pi./N)];
end
%for other yrmod terms, use full sinusoidal cycles, 
%starting at 1 cycle per study period
for ii = 1:nAnnual-1
    annualTerm(:,1+((ii+1)-1)*2:(ii+1)*2) = ...
        [sin(d*2*(ii)*pi./N),cos(d*2*(ii)*pi./N)];
end
 
% construct periodic terms varies on seasonal cycle(1year-2.4month).
seasTerm= zeros(N,nSeasonal*2);
for jj = 1:nSeasonal
    seasTerm(:,(jj-1)*2+1:jj*2)=[sin(jd*2*jj*pi./365),cos(jd*2*jj*pi./365)];
end

% % construct lagged P terms using method from Tuttle,2016.
% pLagTerm = zeros(N,2^nPlag-1);
% indexPlag = cell(nPlag+1,1);
% % lag 0 days
% indexPlag(1) = {1:2*(nAnnual+nSeasonal)};
% %need to do 4 times, once for each different # of P lags
% for ii = 1:nPlag
%     
%     pLagLen=2^ii-1; % number of lagged P terms by each day (i.e.,1-4 days)
%     % psub is index of lagged P array
%     %(e.g., 1 for lagged 1 day, 3 for lagged 2 days)
%     if ii == 1
%         psub = 1:pLagLen;
%     elseif ii > 1
%         psub = max(psub)+1:(pLagLen+max(psub));
%     end
%     % index for select independent variable matrix
%     indexPlag(ii+1) = {[1:2*(nAnnual+nSeasonal),2*(nAnnual+nSeasonal)+ psub]};   
%     
%     lagmod = zeros(N,ii);    
%     for jj = 1:ii
%         lagmod(:,jj) = (circshift(POCC,jj));
%     end 
%     
%     % Convert binary number to decimal number, the same decimal number means 
%     % the same condition, give 1 in that particular day, 
%     % +1 for condition that norain in all days
%     JJ = bin2dec(num2str(lagmod))+1; 
%     
%     newlagmod = zeros(N,2^ii);    
%     for mm = 1:N
%         newlagmod(mm,JJ(mm))=1;
%     end
%     
%     % the first column indicate the condition that no rain in all days. 
%     % exclude it
%     pLagTerm(:,psub) = newlagmod(:,2:end);    
% end

% combine period terms and lagged P terms.
period_terms = [annualTerm,seasTerm];
% corresponding to the length of lagged_terms.
period_terms(end-day_lag+1:end,:) = [];
annualTerm(end-day_lag+1:end,:) = [];
seasTerm(1:day_lag,:) = [];
end

function spatial_terms = ...
         get_spatial_terms(data,Slen,lat,lon,jd,day_lag)

% """get parameters of around pixels, named square. 
% square must be single number"""

if mod(Slen,2)==1
    % get size of data, as [time,lon,lat]
    [Ntime, Nlon, Nlat] = size(data); 
    % initial spatial terms.
    spatial_terms = nan(Ntime,Slen*Slen+1);
    % half of selected square-pixels.
    Hlen = (Slen-1)/2;
    % spatial square range
    lat_range = lat-Hlen:lat+Hlen; 
    lon_range = lon-Hlen:lon+Hlen;
    % numbers of pixels in square 
    N = numel(lat_range)*numel(lon_range);
    % if lat,lon is out of the range
    valid = sum(lat_range<1)+sum(lon_range<1)==0 && ...
            sum(lat_range-112>0)+sum(lon_range-232>0)==0;
    %
    if valid 
        % original parameter of pixel around
        data_ = data(:,lon_range,lat_range);
        data_around = reshape(data_,[Ntime,N]);
        data_pixel = data_around(:,(N-1)/2);
        data_around(:,(N-1)/2) = [];
        data_all = [data_pixel,data_around];
        % construct spatial terms
        spatial_terms = [...
                         data_around,...
                         max(data_all,[],2)-min(data_all,[],2),...
                         sum((data_around-data_pixel).^2,2)...
                         ];
        %              
        spatial_terms = spatial_terms(jd,:);
        spatial_terms(end-day_lag+1:end,:) = [];
    else
        spatial_terms = [];
    end
else
    'square of spatial terms must be odd number'
end 
end

function extreme_terms = ...
         get_extreme_terms(data,lat,lon)
     
disp(TODO:'Need improved')
     
data_pixel = data(:,lon,lat);
data_index = quantile(data_pixel,[0.01,0.05,0.95,0.99]);

% JJ = data_pixel(ii*24+1:ii*24+24);                 
% if length(find(JJ<0.1))>1 & length(find(JJ>0.1))>1
%     extremeIndex(ii,:) = [max(JJ),min(JJ),max(JJ)-min(JJ),std(JJ)...
%         ,length(find(JJ>AA(3))),length(find(JJ>AA(4)))...
%         max(diff([1,find(JJ<0.1)',24]))-1,max([1,find(JJ>0.1)',24])-1];
% end

end

function [season_anomaly,varargout] = get_season_anomaly...
                                      (maxseasterms,seasmod,X,AICPEN)

% """seasonal anomaly by method from Tuttle and Savincci[1]
%
% Arguments:
%   maxseasterms -- 
%   seasmod -- seasonal terms constructed by get_period_terms()
%   X -- target input array
%   AICPEN --
%
% Returns:
%   season_anomaly -- input array removed seasonal average,
% """

N = numel(X);
% index for all possible regression
maxind_S = 2.^(maxseasterms)-1;
index_S = dec2bin(1:maxind_S);
index_S = index_S == '1';
% map index, each time remove sinx and cosx.
MAP_IND_S = nan(1,2*maxseasterms);
%
for jj = 1:maxseasterms
    MAP_IND_S((jj-1)*2+1:(jj-1)*2+2) = jj;
end
%
aicyrandseas_S = nan(maxind_S,1);
%
one = ones(N,1);
%
for kk = 1:maxind_S
    clear II PP
    %
    II = find(index_S(kk,:)==1);
    %
    PP = zeros(1,length(MAP_IND_S));
    for ii = 1:length(II)
        JK = find(MAP_IND_S==II(ii)); %+1 is b/c of constant
        PP(JK) = 1;
    end
    PP = find(PP==1);
    %
    [~,dev,~] = glmfit([one,seasmod(:,PP)],X,...
                        'normal',...
                        'link','identity',...
                        'estdisp','on',...
                        'constant','off');
    %
    indicator = index();
    aicyrandseas_S(kk,1) = indicator.AIC(1+length(PP),dev,AICPEN);
end
%find min aic
[~,bind_S]=nanmin(aicyrandseas_S);
%
clear II PP
%
II = find(index_S(bind_S,:)==1);
PP = zeros(1,length(MAP_IND_S));
for ii = 1:length(II)
    JK = find(MAP_IND_S==II(ii));
    PP(JK) = 1;
end
PP = find(PP==1);
% fit best model
[b_Sseas,~,~] = glmfit([one,seasmod(:,PP)],X,...
                        'normal',...
                        'link','identity',...
                        'estdisp','on',...
                        'constant','off');
% climatology as determined by best AIC seasonal model of S
sclim_9yr_aic = [ones(length(seasmod(:,1)),1),seasmod(:,PP)]*b_Sseas;
% subtract mean S climatology from S
season_anomaly = X-sclim_9yr_aic;
% index of dry and wet days.
IJ = season_anomaly - nanmean(season_anomaly);
quants = quantile(IJ,[0.1 0.25 0.5 0.75 0.9]);          
index_dryday = find(IJ < quants(ceil(length(quants)/2)));
index_wetday = find(IJ > quants(ceil(length(quants)/2)));
%optional additional outputs
if nargout>1; varargout{1} = index_dryday; end
if nargout>2; varargout{2} = index_wetday; end

end

