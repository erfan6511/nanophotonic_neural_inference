
clear;
clc;
addpath(genpath('.\extra_files'))

Nstart = [881 281 1];%these should be odd
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
oposx = floor(linspace(81,800,10));%position of the receivers in x_axis
gamma = cell(1,10);
for ii = 1:10
    gamma{ii} = zeros(N);
end

%It has been set for 2 dimensions
for ii = 1:10
    for jj = 1:Nx
        for kk = 1:Ny
            gamma{ii}(jj,kk,:) = exp((-(jj*Res(1)-oposx(ii)*Res(1))^2 - (kk*Res(2)-(floor(Ny/2)+110)*Res(2))^2)/(2*sigma^2));
        end
    end
end

mask = zeros(N);
for ii = -410:410
    for jj = -100:100
        mask(floor(Nx/2)+ii,floor(Ny/2)+jj,:) = 1;
    end
end

masknl = zeros(N);


xs1 = randi([45 255],1,20);
xs2 = randi([225 430],1,20);
xs3 = randi([400 600],1,20);
xs4 = randi([570 830],1,20);
ys1 = randi([80,200],1,20);
for count = 1:5
    r = 3;
    for i = -5*r:5*r
        for j = -r:r
            if((i*i/(25*r^2))+(j*j/r^2)<=1)
                masknl(xs1(count)+i,ys1(count)+j) = 1;
                masknl(xs2(count)+i,ys1(count)+j) = 1;
                masknl(xs3(count)+i,ys1(count)+j) = 1;
                masknl(xs4(count)+i,ys1(count)+j) = 1;
            end
        end
    end

end

masknl = mask.*masknl;

mask = mask - masknl;


%The training and testing data
load('data.mat'); % training data stored in arrays X, y
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
eps_sio2 = 2.1629;
eps_cell{Axis.z}(floor(Nx/2)+(-410:410),floor(Ny/2)+(-106:100)) = eps_sio2;

xs = randi([35 845],1,2500);
ys = randi([45 235],1,2500);

for i = 1:2500
    eps_cell{Axis.z}(xs(i)+(-2:2),ys(i)+(-2:2),:) = 1;
end

eps_cell{3}(masknl==1) = eps_sio2;

const0 = 2;
k0 = 2;
%initializing phi and del_t
temp_mat = eps_cell{Axis.z};
phi = zeros(size(eps_cell{Axis.z}));
phi(temp_mat==eps_sio2)= -const0;
phi(temp_mat==1)= const0;

%sog_zero = zeros(N);
gdl = (grid3d.dl{Axis.x,GT.prim}(floor(Nx/2)));
omega0 = osc.in_omega0();


fileID = fopen('cost.txt','w');

grad_test = zeros(N);
eps_cell_p = eps_cell;
eps_cell_m = eps_cell;
epsilon = 1e-4;

batch_size = 100;

for counter = 1:150

    
    index_array = randi([1,4000],1,batch_size);
    index_array_test = randi([1,1000],1,batch_size);
    
    test_cost = 0;
    cost = 0;
    grad_tempor = zeros(3*prod(N),1);
    equ = MatrixEquation(eq,pml, omega0, eps_cell, mu_cell, s_factor_cell, {zeros(N),zeros(N),zeros(N)}, M_cell, grid3d);
	[A_init,~] = equ.matrix_op();
    order = equ.r;
    V1 = zeros(N);
	V1(41:2:840,floor(Ny/2)-105,1) = trainx(index_array(1),:)/(gdl*gdl);
	srcj = -1i*omega0*[zeros(prod(N),1);zeros(prod(N),1);V1(:)];
	srcj = srcj(order);
    ej_init = A_init\srcj;
    ej_init = reshape(ej_init,3,prod(N));
    Ezj_init = ej_init(3,:);
    Ezj_init = reshape(Ezj_init,N);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Operation on training data
    
    parfor m =1:batch_size
		indx = index_array(m);
		V1 = zeros(N);
		V1(41:2:840,floor(Ny/2)-105,1) = trainx(indx,:)/(gdl*gdl);
		srcj = -1i*omega0*[zeros(prod(N),1);zeros(prod(N),1);V1(:)];
		srcj = srcj(order);
		eps_cellh = eps_cell;
		%the loop for the nonlinearity
        
        Ezj = Ezj_init;
        wow = 10*ones(N);
        count = 1;
        while 1
            sigmanl = -((1e-6)*1i./(1e-20+Ezj.*conj(Ezj)));
        
            eps_cellh{Axis.z} = eps_cellh{Axis.z} + sigmanl.*masknl;
            equ2 = MatrixEquation(eq,pml, omega0, eps_cellh, mu_cell, s_factor_cell, J_cell, M_cell, grid3d);
            [A,~] = equ2.matrix_op();
        
            ej_new = A\srcj;
            ej_new2 = reshape(ej_new,3,prod(N));
            Ezj = ej_new2(3,:);
            Ezj = reshape(Ezj,N);

            eps_cellh{Axis.z} = eps_cellh{Axis.z} - sigmanl.*masknl;
			%figure();imagesc(abs(Ezj));
            if (abs(sum(sum(real(wow-Ezj)))/sum(sum(real(Ezj))))+abs(sum(sum(imag(wow-Ezj)))/sum(sum(imag(Ezj))))<0.05)||(count>=3)
                sigmanl = -((1e-6)*1i./(1e-20+Ezj.*conj(Ezj)));
                eps_cellh{Axis.z} = eps_cellh{Axis.z} + sigmanl.*masknl;
                equ2 = MatrixEquation(eq,pml, omega0, eps_cellh, mu_cell, s_factor_cell, J_cell, M_cell, grid3d);
                [A,~] = equ2.matrix_op();
                break;
            end
            wow = Ezj;
            fprintf('batch_number:%d, iner_counter:%d\n',m,count);
            count = count + 1;
            
        end
		V3 = zeros(1,10);
		for ii = 1:10
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
		sog_mat=sog_mat - ((trainy(indx,ii)-outj(ii))/V3(ii))*...
			((gamma{ii} - outj(ii)*gam_sum)/(1-outj(ii))).*conj(Ezj);
		end
		
		sog_zero = zeros(prod(N),1);
		tot_sog = [sog_zero;sog_zero;sog_mat(:)];
		tot_sog = tot_sog(order);
		
        
        Al = (omega0^2)*((1e-6)*1i.*((Ezj).*conj(Ezj))./(1e-20 + Ezj.*conj(Ezj)).^2).*masknl;
        Al = [sog_zero;sog_zero;Al(:)];
        Al = create_spdiag(Al);
        Al = Al(order,order);
        Al = Al + A;
        
        Bl = -(omega0^2)*((1e-6)*1i.*(conj(Ezj).*conj(Ezj))./(1e-20 + Ezj.*conj(Ezj)).^2).*masknl;
        Bl = [sog_zero;sog_zero;Bl(:)];
        Bl = create_spdiag(Bl);
        Bl = Bl(order,order);
        
        A_adj = [real(Al+Bl),imag(Al+Bl);imag(Bl-Al),real(Al-Bl)].';
        Src_adj = [-real(tot_sog);-imag(tot_sog)];


        lam_tot = A_adj\Src_adj;

        
        
        lam_mat = lam_tot(1:size(lam_tot,1)/2)+1i*lam_tot(1+size(lam_tot,1)/2:size(lam_tot,1));
        
		grad_tempor = (lam_mat.*ej_new) + grad_tempor;
		fprintf('%d\n',m);
    end
    cost = -cost/batch_size;
    fprintf('iteration: %04d , cost: %f \n',counter,cost);
    fprintf(fileID,'%.4f\n',cost);
    
    grad_tempor = (-(omega0^2)/batch_size)*real(grad_tempor)*gdl*gdl;
	
    grad_tempor = reshape(grad_tempor, Axis.count, prod(N));
    grad_tempor = grad_tempor(int(Axis.z), :);
    grad_tempor = reshape(grad_tempor,N);
    
    
    se = strel('diamond',1);
    temp = zeros(size(phi));
    temp(phi>0)=1;
    eroded = imerode(temp,se);
    edges = temp-eroded;
    
    lr = 80;
    grad = -lr*grad_tempor.*(edges).*mask;
    
    del_t = 1;
    counting = 0;
    iter_num = 1;
    
    %evolving with epsilon gradient
    
    [sgpx,sgpy] = gradient(phi);
    
    phi = phi + del_t*((0.2/del_t)*distReg_p2(phi)-grad);

    temp = eps_sio2*ones(size(phi));
    temp(phi>0) = 1;
    eps_cell{Axis.z}(floor(Nx/2)+(-410:410),floor(Ny/2)+(-100:100),:)=...
        temp(floor(Nx/2)+(-410:410),floor(Ny/2)+(-100:100),:);
end



fclose(fileID);

