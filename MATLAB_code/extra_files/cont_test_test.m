cost = 0;
index_array = randi([1,4000],1,batch_size);
grad_tempor = zeros(3*prod(N),1);
equ = MatrixEquation(eq,pml, omega0, eps_cell, mu_cell, s_factor_cell, {zeros(N),zeros(N),zeros(N)}, M_cell, grid3d);
[A,~] = equ.matrix_op();
[L,U,P,Q] = lu(A);
order = equ.r;
rcount_mat = zeros(1,1000);
parfor counter = 1:1000
    rcount = 0;
	V1 = zeros(Nx,1);
	V2 = zeros(N);
	for ii = 1:400
	V1(40+(ii-1)*1) = trainx(counter,ii)/(gdl*gdl);
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
    [~,idx] = max(outj);
    [~,idy] = max(trainy(counter,:));
    if idx ==idy
        rcount_mat(counter) = 1;
        rcount = 1;
    end
    %fprintf('%d',rcount);
end