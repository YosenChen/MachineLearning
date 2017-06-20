clear all; close all;
kNN_mse = [0.008575, 0.008366, 0.008310, 0.008285, 0.008275, 0.008269];
k = [3, 6, 9, 12, 15, 18];
figure;
plot(k, kNN_mse/4, 'bo-', 'LineWidth',3);
hold on;
plot(k, [1, 1, 1, 1, 1, 1]*0.008392/4, 'r-', 'LineWidth', 3);
legend({'kNN MSE', 'Bilinear MSE'}, 'FontSize',18, 'FontWeight','bold', 'Location', 'best');
set(gca,'fontsize',18)
grid on;
legend({'kNN MSE', 'Bilinear MSE'}, 'FontSize',18, 'FontWeight','bold', 'Location', 'best');
title('Parameter Search for kNN', 'FontSize',18,'FontWeight','bold','Color','b');
xlabel('k: Number of Neighbors', 'FontSize',18,'FontWeight','bold','Color','k')
ylabel('MSE (test set)', 'FontSize',18,'FontWeight','bold','Color','k')
print('kNN_PS.png', '-dpng');


kNN_mse = [0.029874, 0.030044, 0.030437];
ps = [8, 12, 16, 20];
figure;
plot(ps(1:(end-1)), kNN_mse, 'bo-', 'LineWidth',3);
hold on;
plot(ps(1:(end-1)), [1, 1, 1]*0.032727, 'r-', 'LineWidth', 3);
axis([8 16 0.0295 0.033])
set(gca,'fontsize',18)
grid on;
legend({'kNN MSE', 'Bilinear MSE'}, 'FontSize',18, 'FontWeight','bold', 'Location', 'best');
title('Patch Size Search for kNN', 'FontSize',18,'FontWeight','bold','Color','b');
xlabel('Patch Size', 'FontSize',18,'FontWeight','bold','Color','k')
ylabel('Total MSE (24 images)', 'FontSize',18,'FontWeight','bold','Color','k')
print('kNN_PS_patch_size.png', '-dpng');


reg_mse = [0.030132, 0.030327, 0.030131, 0.030444];
figure;
plot(ps+1, reg_mse, 'bo-', 'LineWidth',3);
hold on;
plot(ps+1, [1, 1, 1, 1]*0.032727, 'r-', 'LineWidth', 3);
set(gca,'fontsize',18)
axis([9 21 0.0295 0.033])
grid on;
legend({'LinReg MSE', 'Bilinear MSE'}, 'FontSize',18, 'FontWeight','bold', 'Location', 'best');
title('Patch Size Search for Linear Regression', 'FontSize',18,'FontWeight','bold','Color','b');
xlabel('Patch Size', 'FontSize',18,'FontWeight','bold','Color','k')
ylabel('Total MSE (24 images)', 'FontSize',18,'FontWeight','bold','Color','k')
print('Reg_PS_patch_size.png', '-dpng');


