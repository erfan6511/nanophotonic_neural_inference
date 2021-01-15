
%cross-entropy with normalized output
clear;
clc;

Nstart = [481 389 1];%these should be odd
Res = [10^-7,10^-7,10^-7];
NPML = {[20 20],[20 20],[0 0]};
bc = [BC.p BC.p BC.p];

ft = FT.e;
ge = GT.prim;
eq = EquationType(ft,ge);

%building the system    
    
L0 = 10^-8; % this is the length everything is measured against
lam0 = 100; %wavelength. here is 100 micro meters
unit = PhysUnit(L0);
osc = Oscillation(lam0, unit);


%generateing the grid

%***generating lprim
lprim_cell = cell(1, Axis.count);
Npml = NaN(Axis.count, Sign.count);

for w = Axis.elems
	dl_intended = Res(w)/L0;
    N = Nstart(w);
    if N~=1
        Nm = floor(N/2)+1;% index of the middle point
        Nw = round(((N-Nm)*dl_intended+(N-Nm)*dl_intended)/dl_intended);
        lprim =linspace(-(N-Nm)*dl_intended,(N-Nm)*dl_intended,Nw+1);
    else
        Nw = 1;
        lprim = linspace(0,1,Nw+1);
    end
	Npml(w,Sign.n) = NPML{w}(Sign.n);
	Npml(w,Sign.p) = NPML{w}(Sign.p);
	lprim_cell{w} = lprim;
end

%***generating grid3d
grid3d = Grid3d(osc.unit, lprim_cell, Npml, bc);


%***creating the structures
eps_cell = {ones(grid3d.N), ones(grid3d.N), ones(grid3d.N)};
mu_cell = {ones(grid3d.N), ones(grid3d.N), ones(grid3d.N)};


%for PML
pml = PML.sc;
R_pml = exp(-16);  % target reflection coefficient
deg_pml = 4;  % polynomial degree
s_factor_cell = generate_s_factor(osc.in_omega0(), grid3d, deg_pml, R_pml);

eps_node_cell = cell(1,Axis.count);
mu_node_cell = cell(1,Axis.count);

%applying boundary condition
if bc(1) == 0
    ERxx([1,end],:,:) = -inf;ERyy([1,end],:,:) = -inf;ERzz([1,end],:,:) = -inf;
elseif bc(1) == -1
    URxx([1,end],:,:) = -inf;URyy([1,end],:,:) = -inf;URzz([1,end],:,:) = -inf;
end
if bc(2) == 0
    ERxx(:,[1,end],:) = -inf;ERyy(:,[1,end],:) = -inf;ERzz(:,[1,end],:) = -inf;
elseif bc(2) == -1
    URxx(:,[1,end],:) = -inf;URyy(:,[1,end],:) = -inf;URzz(:,[1,end],:) = -inf;
end
if bc(3) == 0
    ERxx(:,:,[1,end]) = -inf;ERyy(:,:,[1,end]) = -inf;ERzz(:,:,[1,end]) = -inf;
elseif bc(3) == -1
    URxx(:,:,[1,end]) = -inf;URyy(:,:,[1,end]) = -inf;URzz(:,:,[1,end]) = -inf;
end



J_cell = cell(1, Axis.count);
M_cell = cell(1, Axis.count);
for w = Axis.elems
	J_cell{w} = zeros(grid3d.N);
    M_cell{w} = zeros(grid3d.N);
end

N = grid3d.N;

Nx = N(Axis.x);
Ny = N(Axis.y);
Nz = N(Axis.z);

%setting up the receivers
sigma = lam0*L0/2;
oposx = floor(linspace(81,400,10));%position of the receivers in x_axis
gamma = cell(1,10);
for ii = 1:10
    gamma{ii} = zeros(N);
end

%It has been set for 2 dimensions
for ii = 1:10
    for jj = 1:Nx
        for kk = 1:Ny
            gamma{ii}(jj,kk,:) = exp((-(jj*Res(1)-oposx(ii)*Res(1))^2 - (kk*Res(2)-(floor(Ny/2)+162)*Res(2))^2)/(2*sigma^2));
        end
    end
end
%{
mask = zeros(N);
for ii = -210:210
    for jj = [-153:-3,3:153]
        mask(floor(Nx/2)+ii,floor(Ny/2)+jj,:) = 1;
    end
end
%}

%The training and testing data
load('ex3data1.mat'); % training data stored in arrays X, y
Y = zeros(5000,10);
for ii = 1:5000
    ind = round(mod(y(ii),10));
    Y(ii,ind+1)=1;
end
rand_order = randperm(5000);
X = X(rand_order,:);
Y = Y(rand_order,:);

trainx = X(1:4000,:);
trainy = Y(1:4000,:);

testx = X(4001:end,:);
testy = Y(4001:end,:);

clear X y Y

outj = zeros(1,10);
%training level

gdl = (grid3d.dl{Axis.x,GT.prim}(floor(Nx/2)));
omega0 = osc.in_omega0();


epsilon = 10^-4;
eps_cell_p = eps_cell;
eps_cell_m = eps_cell;
grad2 = zeros(N);


eps_sio2 = 2.1629;
eps_si = 13.316;

mask = zeros(N);

eps_cell{Axis.z}(floor(Nx/2)+(-210:210),floor(Ny/2)+(-160:153)) =  eps_sio2;
mask(floor(Nx/2)+(-210:210),floor(Ny/2)+(-153:153)) = 1;

for i = 1:10
    eps_cell{Axis.z}(oposx(i)+(-10:10),floor(Ny/2)+(153:173)) = eps_sio2;
end

eps_cell{3}(floor(Nx/2)+(-210:210),floor(Ny/2)+(-2:2)) = eps_si-1000i;
mask(floor(Nx/2)+(-210:210),floor(Ny/2)+(-2:2)) = 0;
middle_indx = floor(linspace(-208,208,10));

for i = middle_indx
    eps_cell{3}(floor(Nx/2)+i+(-10:10),floor(Ny/2)+(-2:2))=eps_sio2;
end

batch_size = 100;
%}
for counter = 1:1000

    cost = 0;
    index_array = randi([1,4000],1,batch_size);
    grad_tempor = zeros(3*prod(N),1);
    equ = MatrixEquation(eq,pml, omega0, eps_cell, mu_cell, s_factor_cell, {zeros(N),zeros(N),zeros(N)}, M_cell, grid3d);
    [A,~] = equ.matrix_op();
    [L,U,P,Q] = lu(A);
    order = equ.r;
    parfor m =1:batch_size
		indx = index_array(m);
		V1 = zeros(Nx,1);
		V2 = zeros(N);
		for ii = 1:400
		V1(40+(ii-1)*1) = trainx(indx,ii)/(gdl*gdl);
		end
		V2(:,floor(Ny/2)-159,1) = V1;
		srcj = -1i*omega0*[zeros(prod(N),1);zeros(prod(N),1);V2(:)];
		srcj = srcj(order);
		ej_new = Q*(U\(L\(P*srcj)));
		
		ej_tempor = reshape(ej_new, Axis.count, prod(N));
		Ezj = ej_tempor(int(Axis.z), :);
		Ezj = reshape(Ezj,N);
		V3 = zeros(1,10);
		for ii = 1:10
			%outj_mat = (Ezj.*gamma{ii});
			outj_mat = (Ezj.*conj(Ezj).*gamma{ii});
			V3(ii) = sum(outj_mat(:))*(gdl*gdl);
		end
		outj = V3/sum(V3);
		
		cost = cost + sum(trainy(indx,:).*log(outj)+(1-trainy(indx,:)).*log(1-outj));
		
		
		% the first term in the gradient
		
		gam_sum = zeros(N);
		sog_mat = zeros(N);
		
		for ii=1:10
		gam_sum = gam_sum + gamma{ii};
		end
		
		for ii = 1:10
		sog_mat=sog_mat + ((trainy(indx,ii)-outj(ii))/V3(ii))*...
			((gamma{ii} - outj(ii)*gam_sum)/(1-outj(ii))).*conj(Ezj);
		end
		
		sog_zero = zeros(prod(N),1);
		tot_sog = [sog_zero;sog_zero;sog_mat(:)];
		tot_sog = tot_sog(order);
		lam_mat = (P.')*((L.')\((U.')\((Q.')*tot_sog)));
		grad_tempor = (lam_mat.*ej_new) + grad_tempor;
		fprintf('%d\n',m);
    end
    cost = -cost/batch_size;
    fprintf('iteration: %04d , cost: %f \n',counter,cost);


    grad_tempor = (-(omega0^2)/batch_size)*real(grad_tempor)*gdl*gdl;
	
    grad_tempor = reshape(grad_tempor, Axis.count, prod(N));
    grad_tempor = grad_tempor(int(Axis.z), :);
    grad_tempor = reshape(grad_tempor,N);
 
    lr = 30;

    grad = lr*grad_tempor.*mask;

    eps_cell{Axis.z} = eps_cell{Axis.z} - grad;
    
end