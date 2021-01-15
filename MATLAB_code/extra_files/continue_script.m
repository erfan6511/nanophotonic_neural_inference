
fileID = fopen('cost.txt','w');


for counter = 287:1000

    
    index_array = randi([1,4000],1,batch_size);
    cost = 0;
    grad_tempor = zeros(3*prod(N),1);
    
    parfor m = 1:batch_size
        sog_mat = zeros(N);
		indx = index_array( m );
		V1 = zeros(Nx,1);
        V2 = zeros(N);
        for ii = 1:400
           V1(40+(ii-1)*2) = trainx(indx,ii)/(gdl*gdl);
        end
		V2(:,floor(Ny/2)-210,1) = V1;
        equ = MatrixEquation(eq,pml, omega0, eps_cell, mu_cell, s_factor_cell, {zeros(N),zeros(N),V2}, M_cell, grid3d);
        [A,srcj] = equ.matrix_op();
		ej_new = A\srcj;
		
		ej_tempor = reshape(ej_new, Axis.count, prod(N));
		Ezj = ej_tempor(int(Axis.z), :);
		Ezj = reshape(Ezj,N);
		V3 = zeros(1,10);
		for ii = 1:10
			outj_mat = (Ezj.*gamma{ii});
			V3(ii) = sum(outj_mat(:))*(gdl*gdl);
        end
        outj = V3;
		%cost function
		costj_mat = (outj - trainy(indx,:)).* conj(outj - trainy(indx,:));
		costj = sum(costj_mat(:));
		cost = cost + costj;
	
		
		% the first term in the gradient
		
        for ii = 1:10
            sog_mat = sog_mat + conj((outj(ii)-trainy(indx,ii)))*gamma{ii};
        end
		
        tot_sog = [sog_zero(:);sog_zero(:);sog_mat(:)];
        tot_sog = tot_sog(equ.r);
        lam_mat = ((A.')\tot_sog);
        grad_tempor = (lam_mat.*ej_new) + grad_tempor;
        fprintf('%d\n',m);
    end
    toc;
    fprintf('iteration: %04d , cost: %f \n',counter,cost);
    fprintf(fileID,'%.4f\n',cost);

    grad_tempor = ((omega0^2)/batch_size)*real(grad_tempor);
	
    grad_tempor = reshape(grad_tempor, Axis.count, prod(N));
    grad_tempor = grad_tempor(int(Axis.z), :);
    grad_tempor = reshape(grad_tempor,N);
    
    gradz = grad_tempor.*mask;
    
    eps_cell{Axis.z} = eps_cell{Axis.z} - (30-counter/50)*gradz;

    
end
toc
fclose(fileID);