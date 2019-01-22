% HW 2, #3, Kalman Filter - 2-D Car tracking probelm with constant velocity
clear all

dt = 1; % minute
num_meas=7;

H = [1 0; 0 1];
F = [1 0; 0 1];

Ra = [0.6 -0.6; -0.6 1.2];
Rb = [0.6 0.6; 0.6 1.2];
ya = [1 1 1 1 1 1 1; 0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.5];
yb = [4 4 4 4 4 4 4; 0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.5];

Hstack = [H;H];
Rstack = [Ra, zeros(2,2); zeros(2,2), Rb];
ystack = [ya; yb];

B0 = [1.2 0.6; 0.6 0.6];
x0 = [0; 0];

% intialize
xpost = x0;
Bpost = B0;

% Implement KF
% xas, yoa, yob, xb0
xb0 = [0;0];
xas=zeros(2,7);
for i=1:num_meas
    
    if i == 1
        xprior = xpost;
    else
        xprior = xpost+[0.085;-0.085]*dt;
        
    end
    Bprior = Bpost;
    K = Bprior*Hstack'*inv(Hstack*Bprior*Hstack'+Rstack);
    xpost = xprior + K*(ystack(:,i)-Hstack*xprior);
    Bpost = (eye(2)-K*Hstack)*Bprior;
    xas(1,i) = xpost(1);  xas(2,i)= xpost(2);
    plot(xpost(1),xpost(2), 'k*'); hold on; title('Car Location')
end
xpost 
Bpost

figure;
plot(xb0(1),xb0(2), 'b*'); hold on;
plot(xas(1,:), xas(2,:), 'r*');
plot(ya(1,:), ya(2,:), 'g*');
plot(yb(1,:), yb(2,:), 'c*');

% Compute probability that car is at specific location
probability = mvncdf([2-0.1;-2-0.1],[2+0.1;-2+0.1],xpost,Bpost) 