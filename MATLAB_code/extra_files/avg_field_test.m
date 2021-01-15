moo = zeros(N);
counti = 0;
for icount =1100:1140

		V1 = zeros(N);
		V1(41:2:840,floor(Ny/2)-105,1) = X(icount,:)/(gdl*gdl);
		srcj = -1i*omega0*[zeros(prod(N),1);zeros(prod(N),1);V1(:)];
		srcj = srcj(order);
		eps_cellh = eps_cell;
		%the loop for the nonlinearity
        
        Ezj = Ezj_init;
        wow = 10*ones(N);
        count = 1;
        while 1
            sigmanl = -((5e-6)*1i./(1e-10 + Ezj.*conj(Ezj)));
        
            eps_cellh{Axis.z} = eps_cellh{Axis.z} + sigmanl.*masknl;
            equ2 = MatrixEquation(eq,pml, omega0, eps_cellh, mu_cell, s_factor_cell, J_cell, M_cell, grid3d);
            [A,~] = equ2.matrix_op();
        
            ej_new = A\srcj;
            ej_new2 = reshape(ej_new,3,prod(N));
            Ezj = ej_new2(3,:);
            Ezj = reshape(Ezj,N);

            eps_cellh{Axis.z} = eps_cellh{Axis.z} - sigmanl.*masknl;
			%figure();imagesc(abs(Ezj));
            if (abs(sum(sum(real(wow-Ezj)))/sum(sum(real(Ezj))))<0.05)||(count>=3)
            break;
            end
            wow = Ezj;    
            fprintf('batch_number:%d, iner_counter:%d\n',icount,count);
            count = count + 1;
            
        end
		V3 = zeros(1,10);
		for ii = 1:10
			%outj_mat = (Ezj.*gamma{ii});
			outj_mat = (Ezj.*conj(Ezj).*gamma{ii});
			V3(ii) = sum(outj_mat(:))*(gdl*gdl);
		end
		outj = V3/sum(V3);
        [~,icx] = max(outj);
    [~,icy] = max(Y(icount,:));
    if icx ==icy
        moo = moo+abs(Ezj);
        counti = counti + 1;
        fprintf('corr is %d\n',icount);
    end
end