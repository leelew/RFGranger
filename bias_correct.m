%% bias-correct
% TODO: need improved
% bias-correct of VAR regression models.
% bias-correction of random forest regression.
% [1] Bias-corrected random forests in regression.
%     Journal of Applied Statistics

% Block Bootstrap of linear Regressions
function bias_correct()
    %
    if isempty(find(isnan(X),1))
        %estimate block boot length, using automated code from Andrew Patton
        [aa]=opt_block_length_REV_dec07(slagperts_filled);
    else
        aa=[90;NaN];
    end
if isnan(aa(1))
    aa(1)=90;  %if NaN, set block boot length to 90
end
bbpar=1./aa(1);
clear II
%
if bbpar<=0.2
    coefsboot=nan(enshere,length(bcoefsave(:,1)));
    pvalsboot=nan(enshere,length(bcoefsave(:,1)));
    Rind_save=nan(enshere,1);
    
    ct=0;
    done=0;
    good_boot=0;
    
    jk=1;
    jkA=1;
    
    done_offset=0;
    
    while done==0
        
        ct=ct+1;
        NN=length(JJ);  %NN is the length of the data set
        II=[];
        
        while length(II)<NN
            btlength=geornd(bbpar,1);
            t1=ceil(rand*NN);
            JJ2=[t1:t1+btlength];
            KK2=find(JJ2<=NN);
            JJ2=JJ2(KK2);
            II=[II,JJ2];
        end
        II=II(1:1:NN);
        
        %             if length(find(~isnan(POCC(II))))>0.95*length(find(~isnan(POCC))) ...
        %                     && length(find(~isnan(POCC(II))))<1.05*length(find(~isnan(POCC))) ...
        %                     && length(find(~isnan(S(II))))>0.95*length(find(~isnan(S))) ...
        %                     && length(find(~isnan(S(II))))<1.05*length(find(~isnan(S)))
        
        if done_offset==0
            try
                MM=[one(JJ),slagperts(JJ),...
                    Press_lag1(JJ),Press_lag2(JJ),Press_lag3(JJ),Press_lag4(JJ)];
                NN2 = offsetR(JJ);
                NN3 = POCC(JJ);
                [b,dev2,stats,iter,iter_flag,rank_flag,scale_flag]=...
                    glmfit_itersave(MM(II,:),...
                    NN3(II),'binomial','link',...
                    'probit','estdisp','on','constant','off','offset',NN2(II));
                
                coefsboot(jkA,:)=b;    %boot coefs
                pvalsboot(jkA,:)=stats.p;    %boot p values
                
                if any([iter_flag,scale_flag])>0
                    coefsboot(jkA,:)=NaN;
                    pvalsboot(jkA,:)=NaN;
                end
            catch
                coefsboot(jkA,:)=NaN;
                pvalsboot(jkA,:)=NaN;
            end
            
            %require that non-NaN values obtained before
            %moving on to next boot index
            if ~isnan(coefsboot(jkA,:)) & ...
                    all(coefsboot(jkA,:)>typrangemin(UUU)) & ...
                    all(coefsboot(jkA,:)<typrangemax(UUU))
                %save index of restricted model corresponding to jkA
                Rind_save(jkA,1)=jk;
                jkA=jkA+1;
                good_boot=1;
            end
            
            %if enough bootstraps found, exit loop
            if jkA==enshere+1
                done_offset=1;
            end
        end
        
        %if the offset provided a successful bootstrap,
        %increase index by 1
        if good_boot==1
            jk=jk+1;
            good_boot=0;
        end
        %             end
        
        %if boot index>enshere or iteration>50*enshere, stop loop
        if done_offset==1
            'complete set of bootstraps'
            bootstrap_complete=done_offset;    %boostrap finished successfully
            bootstrap_num=jkA-1;   %number of successful bootstrap samples
            done=1;
        elseif ct==50*enshere
            'not enough successful bootstraps'
            bootstrap_complete=done_offset;    %boostrap did not finish
            bootstrap_num=jkA-1;    %number of successful bootstrap samples
            failure=5;
            done=1;
        end
    end
    
    
    %% Unbias Coefficients and Calculate p Values
    
    %%%Unrestricted, Offset Model
    
    coefmean=squeeze(nanmean(coefsboot))';
    
    coefbest=2*squeeze(bcoefsave(UUU))-coefmean;
    
    %correct entire histogram of S coefficient values
    benshereC=squeeze(coefsboot(:,2))-coefmean(2)+coefbest(2);
    
    if ~isnan(coefbest(2))
        
        II=find(~isnan(benshereC));
        
        if nanmedian(benshereC(II))>0
            pval=2*(ksdensity(benshereC(II),0,'function','cdf'));
        end
        if nanmedian(benshereC(II))<0
            pval=2*(1-ksdensity(benshereC(II),0,'function','cdf'));
        end
        pval_boots_ksdens=pval;
        bvals_U_all(UUU)=coefbest;
    else
        pval_boots_ksdens = NaN;
    end
    
else
    'Estimated block boot length is less than 1'
    failure=4;
    S_impact_allR = nan(2,1);
    pval_boots_ksdens = NaN;
end
end

function xlag = mlag(x,n,init)
% """generates a matrix of n lags from a matrix (or vector)
% containing a set of vectors (For use in var routines)

% Arguments:
%   x -- a matrix (or vector), nobs x nvar
%   nlag -- # of contiguous lags for each vector in x
%   init -- (optional) scalar value to feed initial missing values
%             (default = 0)

% Returns:
%   xlag -- a matrix of lags (nobs x nvar*nlag)
%           x1(t-1), x1(t-2), ... x1(t-nlag), x2(t-1), ... x2(t-nlag) ...

% SEE ALSO: lag() which works more conventionally
% """

if nargin ==1 
    n = 1; % default value
    init = 0;
elseif nargin == 2
    init = 0;
end

if nargin > 3
    error('mlag: Wrong # of input arguments');
end

[nobs, nvar] = size(x);

xlag = ones(nobs,nvar*n)*init;
icnt = 0;
for i=1:nvar
    for j=1:n
        xlag(j+1:nobs,icnt+j) = x(1:nobs-j,i);
    end
    icnt = icnt+n;
end
end

function Bstar = opt_block_length_REV_dec07(data)

% """This is a function to select the optimal (in the sense
% of minimising the MSE of the estimator of the long-run
% variance) block length for the stationary bootstrap or circular bootstrap.
% Code follows Politis and White, 2001, "Automatic Block-Length
% Selection for the Dependent Bootstrap".

%  NOTE: The optimal average block length for the stationary bootstrap, 
%        and it does not need to be an integer.The optimal block length for 
%        the circular bootstrap should be an integer. Politis and White suggest 
%        rounding the output UP to the nearest integer.

% Returns:
%   Bstar, a 2xk vector of optimal bootstrap block lengths, [BstarSB;BstarCB]
% """

[n,k] = size(data);

% these are optional in opt_block_length_full.m, 
% but fixed at default values here
KN = max(5,sqrt(log10(n)));
% adding KN extra lags to employ Politis' (2002) suggestion 
% for finding largest signif m
mmax = ceil(sqrt(n))+KN;    
% dec07: new idea for rule-of-thumb to 
% put upper bound on estimated optimal block length
Bmax = ceil(min(3*sqrt(n),n/3));  

c=2;
origdata=data;
Bstar_final=[];

for i=1:k
   data=origdata(:,i);
   
   % FIRST STEP: finding mhat-> the largest lag for which the autocorrelation is still significant.
   temp = mlag(data,mmax);
   temp = temp(mmax+1:end,:);	% dropping the first mmax rows, as they're filled with zeros
   temp = corrcoef([data(mmax+1:end),temp]);
   temp = temp(2:end,1);
   
   % We follow the empirical rule suggested in Politis, 2002, "Adaptive Bandwidth Choice".
   % as suggested in Remark 2.3, setting c=2, KN=5
   temp2 = [mlag(temp,KN)',temp(end-KN+1:end)];		% looking at vectors of autocorrels, from lag mhat to lag mhat+KN
   temp2 = temp2(:,KN+1:end);		% dropping the first KN-1, as the vectors have empty cells
   temp2 = (abs(temp2)<(c*sqrt(log10(n)/n)*ones(KN,mmax-KN+1)));	% checking which are less than the critical value
   temp2 = sum(temp2)';		% this counts the number of insignif autocorrels
   temp3 = [(1:1:length(temp2))',temp2];
   temp3 = temp3(find(temp2==KN),:);	% selecting all rows where ALL KN autocorrels are not signif
   if isempty(temp3)
      mhat = max(find(abs(temp) > (c*sqrt(log10(n)/n)) )); % this means that NO collection of KN autocorrels were all insignif, so pick largest significant lag
   else
      mhat = temp3(1,1);	% if more than one collection is possible, choose the smallest m
   end
   if 2*mhat>mmax
      M = mmax;
      trunc1=1;
   else
      M = 2*mhat;
   end
   clear temp temp2 temp3;
   
   
   % SECOND STEP: computing the inputs to the function for Bstar
   kk = (-M:1:M)';
   
   if M>0
      temp = mlag(data,M);
      temp = temp(M+1:end,:);	% dropping the first mmax rows, as they're filled with zeros
      temp = cov([data(M+1:end),temp]);
      acv = temp(:,1);			% autocovariances
      acv2 = [-(1:1:M)',acv(2:end)];
      if size(acv2,1)>1
         acv2 = sortrows(acv2,1);
      end
      acv = [acv2(:,2);acv];			% autocovariances from -M to M
      clear acv2;
      Ghat = sum(lam(kk/M).*abs(kk).*acv);
      DCBhat = 4/3*sum(lam(kk/M).*acv)^2;
 
% OLD nov07      
%      DSBhat = 2/pi*quadl('opt_block_length_calc',-pi,pi,[],[],kk,acv,lam(kk/M));
%      DSBhat = DSBhat + 4*sum(lam(kk/M).*acv)^2;	% first part of DSBhat (note cos(0)=1)

% NEW dec07
      DSBhat = 2*(sum(lam(kk/M).*acv)^2);	% first part of DSBhat (note cos(0)=1)
      
      % FINAL STEP: constructing the optimal block length estimator
      Bstar = ((2*(Ghat^2)/DSBhat)^(1/3))*(n^(1/3));
      if Bstar>Bmax
         Bstar = Bmax;
      end
      BstarCB = ((2*(Ghat^2)/DCBhat)^(1/3))*(n^(1/3));
      
      if BstarCB>Bmax
         BstarCB = Bmax;
      end      
      Bstar = [Bstar;BstarCB];
   else
      Ghat = 0;
      % FINAL STEP: constructing the optimal block length estimator
      Bstar = [1;1];
   end
   Bstar_final=[Bstar_final Bstar];
end
Bstar=Bstar_final;

%%%%%%%%%%%%%%%%%%%%%%%%
function lam=lam(kk)
%Helper function, calculates the flattop kernel weights
lam = (abs(kk)>=0).*(abs(kk)<0.5)+2*(1-abs(kk)).*(abs(kk)>=0.5).*(abs(kk)<=1);
end
end
