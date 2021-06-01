%% Figure for special around
% Figure method of output dataframe in NGCF
% the result dataframe is [impact,p_value,varargout]x[nLon,nLat]

function fig = figure_()
fig.run_figure = @run_figure;
fig.sign = @sign;
end

function fig = run_figure(X,color,title_,axis_range)

lat = (25:0.25:52.75)';
lon = (-125:0.25:-67.25)';

m_proj('Lambert Conformal Conic',...
       'lat',[25 50],...
       'lon',[-125 -65]);
h = m_pcolor(lon,lat,X');
m_coast('color',[0 0 0],'linewidth',1);
m_gshhs('hb1','linewidth',1,'color','k'); 
hold on;
set(h,'edgecolor','none');
m_grid('box','on',...
       'xtick',7,...
       'ytick',6,...
       'fontsize',12,...
       'fontname','Times New Roman',...
       'gridcolor','none');
%m_colmap(color)
colormap(color);
caxis(axis_range);
%colorbar('location','SouthOutside','FontSize',11); 
title(title_,'Fontsize',13,'FontName','Times New Roman');

end

function [impact_dry,impact_wet,varargout] = sign(result)

% """divide the result got from nonlinear causal inference model"""

%
p_value = squeeze(result(1,:,:));
impact_dry = squeeze(result(2,:,:));
impact_wet = squeeze(result(3,:,:));
R2_P = squeeze(result(4,:,:));
R2_S = squeeze(result(5,:,:));

if numel(result(:,1,1))>5
    R2_full = squeeze(result(6,:,:));
    R2_baseline = squeeze(result(7,:,:));
    endog_P = squeeze(result(8,:,:));
    endog_S = squeeze(result(9,:,:));
end

%
[nLon,nLat] = size(p_value);
%
impact_dry(find(p_value>0.05))= nan;
impact_wet(find(p_value>0.05))= nan;
%
if nargout>2; varargout{1} = R2_P; end
if nargout>3; varargout{2} = R_full_baseline; end
if nargout>4; varargout{3} = endogeneity; end
end
