clear;
options.NeighborMode = 'KNN';
options.k = 10; 
options.WeightMode = 'HeatKernel'; % HeatKernel
% options.WeightMode = 'Binary'; % BinaryKernel
load NUS-WIDE.mat
dataname = 'NUW-WIDE';

X1 = X{1};
X1 = double(X1);
G1 = constructW(X1,options);
G1 = G1;

X2 = X{2};
X2 = double(X2);
G2 = constructW(X2,options);
G2 = G2;

X3 = X{3};
X3 = double(X3);
G3 = constructW(X3,options);
G3 = G3;

X4 = X{4};
X4 = double(X4);
G4 = constructW(X4,options);
G4 = G4;

X5 = X{5};
X5 = double(X5);
G5 = constructW(X5,options);
G5 = G5;


X6 = X{6};
X6 = double(X6);
G6 = constructW(X6,options);
G6 = G6;

NUSWIDEMGW = {G1, G2, G3, G4, G5, G6};
save('NUSWIDEMGW','NUSWIDEMGW');

