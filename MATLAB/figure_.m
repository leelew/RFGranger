% Figure method of output dataframe in NGCF
% the result dataframe is [impact,p_value,varargout]x[nLon,nLat]
%
%
function fig = figure_()
fig.run_figure = @run_figure;
fig.sign = @sign;
fig.run = @run;
end

function run(result)
figure
[impact_dry,impact_wet,R2] = sign(result);
subplot(1,2,1)
%subplot('Position',[0.01 0.51 0.48 0.45]);
run_figure(impact_dry,'jet','Dry',[0.8,1.2])
%subplot('Position',[0.01 0.51 0.48 0.45]);
subplot(1,2,2)
run_figure(impact_wet,'jet','Wet',[0.8,1.2])
%subplot(3,1,3)
%run_figure(R2,'jet','R2',[0.6,0.9])
end

function fig = run_figure(X,color,title_,axis_range)

lat = [25:0.25:52.75]';
lon = [-125:0.25:-67.25]';

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
%m_colmap('jet')
colormap(color);
caxis(axis_range);
colorbar('location','SouthOutside','FontSize',11); 
title(title_,'Fontsize',20);

end

function [impact_dry,impact_wet,R2] = sign(result)
% give p_value shape as (nLon,nLat)

%
impact_dry = squeeze(result(2,:,:));
impact_wet = squeeze(result(3,:,:));
p_value = squeeze(result(1,:,:));
R2 = squeeze(result(4,:,:));
%
[nLon,nLat] = size(p_value);
%
impact_dry(find(p_value>0.05))= nan;
impact_wet(find(p_value>0.05))= nan;

end
