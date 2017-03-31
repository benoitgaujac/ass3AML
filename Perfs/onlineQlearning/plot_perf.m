perfs_30     = readtable('Qlearning_30_perf.csv');
perfs_100     = readtable('Qlearning_100_perf.csv');
perfs_1000     = readtable('Qlearning_1000_perf.csv');
perfs_replay   = readtable('replay_perf.csv');
perfs_target  = readtable('target_perf.csv');
perfs_double = readtable('doubleQ_perf.csv');
xaxis = perfs_30{:,1};
perfs_30     = perfs_30{:,:};
mov_perfs_30 = tsmovavg(perfs_30,'s',5,1);
perfs_100    = perfs_100{:,:};
mov_perfs_100 = tsmovavg(perfs_100,'s',5,1);
perfs_1000   = perfs_1000{:,:};
mov_perfs_1000 = tsmovavg(perfs_1000,'s',5,1);
perfs_replay = perfs_replay{:,:};
mov_perfs_replay = tsmovavg(perfs_replay,'s',5,1);
perfs_target = perfs_target{:,:};
mov_perfs_target = tsmovavg(perfs_target,'s',5,1);
perfs_double = perfs_double{:,:};
mov_perfs_double = tsmovavg(perfs_double,'s',5,1);

losses_30     = readtable('Qlearning_30_losses.csv');
losses_100    = readtable('Qlearning_100_losses.csv');
losses_1000   = readtable('Qlearning_1000_losses.csv');
losses_replay = readtable('replay_losses.csv');
losses_target = readtable('target_losses.csv');
losses_double = readtable('doubleQ_losses.csv');
xaxisloss = losses_30{:,1};
%losses_30     = losses_30{:,:};
%losses_100    = losses_100{:,:};
%losses_1000   = losses_1000{:,:};
%losses_replay = losses_replay{:,:};
%losses_target = losses_target{:,:};
%losses_double = losses_double{:,:};
%losses_30     = log(losses_30{:,:});
%losses_100    = log(losses_100{:,:});
%losses_1000   = log(losses_1000{:,:});
%losses_replay = log(losses_replay{:,:});
%losses_target = log(losses_target{:,:});
%losses_double = log(losses_double{:,:});
losses_30     = 2*sqrt(losses_30{:,:});
losses_100    = 2*sqrt(losses_100{:,:});
losses_1000   = 2*sqrt(losses_1000{:,:});
losses_replay = 2*sqrt(losses_replay{:,:});
losses_target = 2*sqrt(losses_target{:,:});
losses_double = 2*sqrt(losses_double{:,:});


col1 = [0 0 1]; %blue
col2 = [1 0.8 0.2]; %orange
col3 = [0 1 1]; %turquoise
col4 = [0 0.6 0.6]; %dark green
col5 = [0 1 0]; %clear green
col6 = [1 0 0]; %red

f1 = figure
plot(xaxis,mov_perfs_30(:,2),'-',xaxis,perfs_30(:,2),':','color',col1,'LineWidth',.9) 
title('Mean episodes lentgh')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
%ylim([5 162])
%legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','southeast')
hold on;
plot(xaxis,mov_perfs_100(:,2),'-',xaxis,perfs_100(:,2),':','color',col2,'LineWidth',.9) 
hold on;
plot(xaxis,mov_perfs_1000(:,2),'-',xaxis,perfs_1000(:,2),':','color',col3,'LineWidth',.9) 
hold on;
plot(xaxis,mov_perfs_replay(:,2),'-',xaxis,perfs_replay(:,2),':','color',col4,'LineWidth',.9) 
hold on;
plot(xaxis,mov_perfs_target(:,2),'-',xaxis,perfs_target(:,2),':','color',col5,'LineWidth',.9) 
hold on;
plot(xaxis,mov_perfs_double(:,2),'-',xaxis,perfs_double(:,2),':','color',col6,'LineWidth',.9) 
hold off;
saveas(f1,'plots/meanlen.jpg')

f2 = figure
plot(xaxis,mov_perfs_30(:,4),'-',xaxis,perfs_30(:,4),':','color',col1,'LineWidth',.9) 
title('Mean episodes returns')
xlabel('epochs') % x-axis label
ylabel('episode returns') % y-axis label
%ylim([-1 -0.22])
%legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','southeast')
hold on;
plot(xaxis,mov_perfs_100(:,4),'-',xaxis,perfs_100(:,4),':','color',col2,'LineWidth',.9) 
hold on;
plot(xaxis,mov_perfs_1000(:,4),'-',xaxis,perfs_1000(:,4),':','color',col3,'LineWidth',.9) 
hold on;
plot(xaxis,mov_perfs_replay(:,4),'-',xaxis,perfs_replay(:,4),':','color',col4,'LineWidth',.9) 
hold on;
plot(xaxis,mov_perfs_target(:,4),'-',xaxis,perfs_target(:,4),':','color',col5,'LineWidth',.9) 
hold on;
plot(xaxis,mov_perfs_double(:,4),'-',xaxis,perfs_double(:,4),':','color',col6,'LineWidth',.9) 
hold off;
saveas(f2,'plots/meanreturns.jpg')

f3 = figure
plot(xaxisloss,losses_30(:,2),'-','color',col1,'LineWidth',.9) 
title('Training loss')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
%ylim([0.0 8.0])
%legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','southeast')
hold on;
plot(xaxisloss,losses_100(:,2),'-','color',col2,'LineWidth',.9) 
hold on;
plot(xaxisloss,losses_1000(:,2),'-','color',col3,'LineWidth',.9) 
hold on;
plot(xaxisloss,losses_replay(:,2),'-','color',col4,'LineWidth',.9) 
hold on;
plot(xaxisloss,losses_target(:,2),'-','color',col5,'LineWidth',.9) 
hold on;
plot(xaxisloss,losses_double(:,2),'-','color',col6,'LineWidth',.9) 
hold off;
saveas(f3,'plots/losses.jpg')

f4 = figure
plot(perfs_30(:,1),perfs_30(:,2),'-',perfs_30(:,1),perfs_30(:,2)+perfs_30(:,3),':',...
                                           perfs_30(:,1),perfs_30(:,2)-perfs_30(:,3),':','color',col1,'LineWidth',.9) 
title('Mean episodes lentgh for nunits=30')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f4,'plots/Q30.jpg')

f5 = figure
plot(perfs_100(:,1),perfs_100(:,2),'-',perfs_100(:,1),perfs_100(:,2)+perfs_100(:,3),':',...
                                           perfs_100(:,1),perfs_100(:,2)-perfs_100(:,3),':','color',col2,'LineWidth',.9) 
title('Mean episodes lentgh for nunits=100')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f5,'plots/Q100.jpg')

f6 = figure
plot(perfs_1000(:,1),perfs_1000(:,2),'-',perfs_1000(:,1),perfs_1000(:,2)+perfs_1000(:,3),':',...
                                           perfs_1000(:,1),perfs_1000(:,2)-perfs_1000(:,3),':','color',col3,'LineWidth',.9) 
title('Mean episodes lentgh for nunits=1000')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f6,'plots/Q1000.jpg')

f7 = figure
plot(perfs_replay(:,1),perfs_replay(:,2),'-',perfs_replay(:,1),perfs_replay(:,2)+perfs_replay(:,3),':',...
                                           perfs_replay(:,1),perfs_replay(:,2)-perfs_replay(:,3),':','color',col4,'LineWidth',.9) 
title('Mean episodes lentgh for Qlearning with experience replay buffer')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f7,'plots/replay.jpg')

f8 = figure
plot(perfs_target(:,1),perfs_target(:,2),'-',perfs_target(:,1),perfs_target(:,2)+perfs_target(:,3),':',...
                                           perfs_target(:,1),perfs_target(:,2)-perfs_target(:,3),':','color',col5,'LineWidth',.9) 
title('Mean episodes lentgh for Qlearning with experience replay buffer and target network')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f8,'plots/target.jpg')

f9 = figure
plot(mov_perfs_double(:,1),mov_perfs_double(:,2),'-',mov_perfs_double(:,1),mov_perfs_double(:,2)+mov_perfs_double(:,3),':',...
                                           mov_perfs_double(:,1),mov_perfs_double(:,2)-mov_perfs_double(:,3),':','color',col6,'LineWidth',.9) 
title('Mean episodes lentgh for double Qlearning')
xlabel('epochs') % x-axis label
ylabel('episode lenght') % y-axis label
legend('Mean lentgh', 'std lentgh','std lentgh', 'Location','best')
saveas(f9,'plots/double.jpg')

close all