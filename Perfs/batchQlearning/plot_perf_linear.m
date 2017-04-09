perfs_05     = readtable('perf_linear_0.5.csv');
perfs_01     = readtable('perf_linear_0.1.csv');
perfs_001    = readtable('perf_linear_0.01.csv');
perfs_0001   = readtable('perf_linear_0.001.csv');
perfs_00001  = readtable('perf_linear_0.0001.csv');
perfs_000001 = readtable('perf_linear_1e-05.csv');
xaxis = perfs_05{:,1};
perfs_05     = perfs_05{:,:};
perfs_01     = perfs_01{:,:};
perfs_001    = perfs_001{:,:};
perfs_0001   = perfs_0001{:,:};
perfs_00001  = perfs_00001{:,:};
perfs_000001 = perfs_000001{:,:};

losses_05     = readtable('losses_linear_0.5.csv');
losses_01     = readtable('losses_linear_0.1.csv');
losses_001    = readtable('losses_linear_0.01.csv');
losses_0001   = readtable('losses_linear_0.001.csv');
losses_00001  = readtable('losses_linear_0.0001.csv');
losses_000001 = readtable('losses_linear_1e-05.csv');
xaxisloss = losses_05{:,1};
losses_05     = log(losses_05{:,:});
losses_01     = log(losses_01{:,:});
losses_001    = log(losses_001{:,:});
losses_0001   = log(losses_0001{:,:});
losses_00001  = log(losses_00001{:,:});
losses_000001 = log(losses_000001{:,:});


%one_perf = ones(80,1);
col1 = [0 0 1]; %blue
col2 = [1 0.8 0.2]; %orange
col3 = [0 1 1]; %turquoise
col4 = [0 0.6 0.6]; %dark green
col5 = [0 1 0]; %clear green
col6 = [1 0 0]; %red

f1 = figure;
plot(xaxis,perfs_05(:,2),'-','color',col1,'LineWidth',.8) 
title('Mean episodes lentgh')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
%ylim([5 162])
%legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','southeast')
hold on;
plot(xaxis,perfs_01(:,2),'-','color',col2,'LineWidth',.8) 
hold on;
plot(xaxis,perfs_001(:,2),'-','color',col3,'LineWidth',.8) 
hold on;
plot(xaxis,perfs_0001(:,2),'-','color',col4,'LineWidth',.8) 
hold on;
plot(xaxis,perfs_00001(:,2),'-','color',col5,'LineWidth',.8) 
hold on;
plot(xaxis,perfs_000001(:,2),'-','color',col6,'LineWidth',.8) 
hold off;
saveas(f1,'plots/len_linear.jpg')

f2 = figure
plot(xaxis,perfs_05(:,4),'-','color',col1,'LineWidth',.8) 
title('Mean episodes returns')
xlabel('epochs') % x-axis label
ylabel('episode returns') % y-axis label
%ylim([-1 -0.22])
%legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','southeast')
hold on;
plot(xaxis,perfs_01(:,4),'-','color',col2,'LineWidth',.8) 
hold on;
plot(xaxis,perfs_001(:,4),'-','color',col3,'LineWidth',.8) 
hold on;
plot(xaxis,perfs_0001(:,4),'-','color',col4,'LineWidth',.8) 
hold on;
plot(xaxis,perfs_00001(:,4),'-','color',col5,'LineWidth',.8) 
hold on;
plot(xaxis,perfs_000001(:,4),'-','color',col6,'LineWidth',.8) 
hold off;
saveas(f2,'plots/return_linear.jpg')

f3 = figure
plot(xaxisloss,losses_05(:,2),'-','color',col1,'LineWidth',.8) 
title('Training loss')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
%ylim([1.25 4])
%legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','southeast')
hold on;
plot(xaxisloss,losses_01(:,2),'-','color',col2,'LineWidth',.8) 
hold on;
plot(xaxisloss,losses_001(:,2),'-','color',col3,'LineWidth',.8) 
hold on;
plot(xaxisloss,losses_0001(:,2),'-','color',col4,'LineWidth',.8) 
hold on;
plot(xaxisloss,losses_00001(:,2),'-','color',col5,'LineWidth',.8) 
hold on;
plot(xaxisloss,losses_000001(:,2),'-','color',col6,'LineWidth',.8) 
hold off;
saveas(f3,'plots/loss_linear.jpg')

f4 = figure
plot(perfs_05(:,1),perfs_05(:,2),'-',perfs_05(:,1),perfs_05(:,2)+perfs_05(:,3),':',...
                                           perfs_05(:,1),perfs_05(:,2)-perfs_05(:,3),':','color',col1,'LineWidth',.9) 
title('Mean episodes lentgh for lr=0.5')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f4,'plots/linear_meanl_5.jpg')

f5 = figure
plot(perfs_01(:,1),perfs_01(:,2),'-',perfs_01(:,1),perfs_01(:,2)+perfs_01(:,3),':',...
                                           perfs_01(:,1),perfs_01(:,2)-perfs_01(:,3),':','color',col2,'LineWidth',.9) 
title('Mean episodes lentgh for lr=0.1')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f5,'plots/linear_meanl_1.jpg')

f6 = figure
plot(perfs_001(:,1),perfs_001(:,2),'-',perfs_001(:,1),perfs_001(:,2)+perfs_001(:,3),':',...
                                           perfs_001(:,1),perfs_001(:,2)-perfs_001(:,3),':','color',col3,'LineWidth',.9) 
title('Mean episodes lentgh for lr=0.01')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f6,'plots/linear_meanl_01.jpg')

f7 = figure
plot(perfs_0001(:,1),perfs_0001(:,2),'-',perfs_0001(:,1),perfs_0001(:,2)+perfs_0001(:,3),':',...
                                           perfs_0001(:,1),perfs_0001(:,2)-perfs_0001(:,3),':','color',col4,'LineWidth',.9) 
title('Mean episodes lentgh for lr=0.001')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f7,'plots/linear_meanl_001.jpg')

f8 = figure
plot(perfs_00001(:,1),perfs_00001(:,2),'-',perfs_00001(:,1),perfs_00001(:,2)+perfs_00001(:,3),':',...
                                           perfs_00001(:,1),perfs_00001(:,2)-perfs_00001(:,3),':','color',col5,'LineWidth',.9) 
title('Mean episodes lentgh for lr=0.0001')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f8,'plots/linear_meanl_0001.jpg')

f9 = figure
plot(perfs_000001(:,1),perfs_000001(:,2),'-',perfs_000001(:,1),perfs_000001(:,2)+perfs_000001(:,3),':',...
                                           perfs_000001(:,1),perfs_000001(:,2)-perfs_000001(:,3),':','color',col6,'LineWidth',.9) 
title('Mean episodes lentgh for lr=0.00001')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f9,'plots/linear_meanl_00001.jpg')

close all