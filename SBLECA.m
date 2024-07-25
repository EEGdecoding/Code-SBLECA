function [W, alpha,V, R_test] = SBLECA(X_train,Y,X_test, K, tau, e)
% ************************************************************************
% SBLEST    : Spatio-Temporal-filtering-based single-trial EEG classification
%
% --- Inputs ---
% Y         : Observed label vector
% X         : M EEG signals from train set
%             M cells: Each X{1,i} represents a trial with size of [C*T]
% Maxiters  : Maximum number of iterations (5000 is suggested in this paper)
% e         : Convergence threshold 
%
% --- Output ---
% W         : The estimated low-rank matrix
% alpha     : The classifier parameter
% V         : Each column of V represents a spatio-temporal filter

% Reference:

% Copyright:

% ************************************************************************

[~, ~, cov_train] = compute_covariance_train(X_train,K,tau);
 [R_test, cov_test] = compute_covariance_test_nowhiten(X_test, K, tau);

C_S = squeeze(mean(cov_train,3));
C_T = squeeze(mean(cov_test,3));

r_CS = rank(C_S);
r_CT = rank(C_T);
r = min(r_CS, r_CT);

% Compute the SVD
[U_S, S_S, V_S] = svd(C_S);
[U_T, S_T, V_T] = svd(C_T);


% Compute A* according to the theorem
Sig = sqrtm(pinv(S_S));
A_star = U_S*Sig * U_S' * U_T(:,1:r) * S_T(1:r,1:r)^(1/2) * U_T(:,1:r)';

cov_train_coral = cell(1,numel(X_train));
for num_train = 1: size(cov_train,3)
    cov_train_CORAL(num_train,:,:) = A_star'* cov_train(:,:,num_train)*A_star;
     R(num_train,:) = vec(squeeze(cov_train_CORAL(num_train,:,:)));
end

C_S_2 =  squeeze(mean(cov_train_CORAL,1));

% Cov = compute_covariance(X,K,tau);

% [R,Cov_mean_train] = compute_enhanced_cov_matrix(Cov);

%% Check properties of R
[M, D_R] = size(R); % M: # of samples;D_R: dimention of vec(R_m)
KC = round(sqrt(D_R));
epsilon =e;
Loss_old = 0;
if (D_R ~= KC^2)
    disp('ERROR: Columns of A do not align with square matrix');
    return;
end

% Check properties of R: symmetric ?
for c = 1:M
    row_cov = reshape(R(c,:), KC, KC);
    if ( norm(row_cov - row_cov','fro') > 1e-4 )
        disp('ERROR: Measurement row does not form symmetric matrix');
        return
    end
end

%% Initializations

U = zeros(KC,KC); % The estimated low-rank matrix W set to be Zeros

Psi = eye(KC); % the covariance matrix of Gaussian prior distribution is initialized to be Unit diagonal matrix

lambda = 1;% the variance of the additive noise set to 1 by default

%% Optimization loop
for i = 1:500
   %% Compute estimate of X 
    RPR = zeros(M, M); %  Predefined temporal variables RT*PSI*R
    B = zeros(KC^2, M); %  Predefined temporal variables
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        Temp = Psi*R(:,start:stop)'; 
        B(start:stop,:)= Temp; 
        RPR =  RPR + R(:,start:stop)*Temp;  
    end

    Sigma_y = RPR + lambda*eye(M); 
    u = B*( Sigma_y\Y ); % Maximum a posterior estimation of u
    U = reshape(u, KC, KC);
    U = (U + U')/2; % make sure U is symmetric
       
   %% Update the dual variables of PSI : PHi_i
    Phi = cell(1,KC);
    SR = Sigma_y\R;
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        Phi{1,c} = Psi -Psi * ( R(:,start:stop)'*SR(:,start:stop) ) * Psi;
    end
    
          %% Update covariance parameters Psi: Gx
    PHI = 0;    
    UU = 0;
    for c = 1:KC
        PHI = PHI +  Phi{1,c};
        UU = UU + U(:,c) * U(:,c)';
    end
    Psi = ( (UU + UU')/2 + (PHI + PHI')/2 )/KC;% make sure Psi is symmetric

   %% Update lambda
   theta = 0;
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        theta = theta +trace(Phi{1,c}* R(:,start:stop)'*R(:,start:stop)) ;
    end
    lambda = (sum((Y-R*u).^2) + theta)/M;  

   %% Output display and  convergence judgement
       logdet_Sigma_y =  calculate_log_det(Sigma_y);
       Loss = Y'*Sigma_y^(-1)*Y + logdet_Sigma_y;    
        delta_loss = norm(Loss - Loss_old,'fro')/norm( Loss_old,'fro');  
        if (delta_loss < epsilon)
            disp('EXIT: Change in Loss below threshold');
            break;
        end
        Loss_old = Loss;
         if (~rem(i,100))
            disp(['Iterations: ', num2str(i),  '  lambda: ', num2str(lambda),'  Loss: ', num2str(Loss), '  Delta_Loss: ', num2str(delta_loss)]);
         end   
end
     %% Eigendecomposition of W
     W = U;
     [~,D,V] = eig(W);% each column of V represents a spatio-temporal filter
     alpha = diag(D); % the classifier parameter
     
end


function [R,Cov_mean_train,cov_train] = compute_covariance_train(X,K, tau)
M = length(X);
[C,T] = size(X{1,1});
KC = K*C;
Cov = cell(1,M);
Sig_Cov = zeros(KC,KC);
for m = 1:M
    X_m = X{m};
    X_m_hat = [];
    for k = 1:K
        n_delay = (k-1)*tau;
        if n_delay ==0
            X_order_k = X_m;
        else
            X_order_k(:,1:n_delay) = 0;
            X_order_k(:,n_delay+1:T) = X_m(:, 1:T-n_delay);
        end
        X_m_hat = cat(1,X_m_hat,X_order_k);
    end
    Cov{1,m} = X_m_hat*X_m_hat';
    Cov{1,m} = Cov{1,m}/trace(Cov{1,m}); % trace normalizaiton
    Sig_Cov = Sig_Cov + Cov{1,m};
end

Cov_mean_train = Sig_Cov/M;
for m = 1:M
    R(m,:) = vec(logm(squeeze(Cov{1,m})));% logm
    cov_train(:,:,m) = logm(squeeze( Cov{1,m}));
end 
   R = real(R);
end

function [R_test, cov_test] = compute_covariance_test_nowhiten(X, K, tau)
M = length(X);
[C,T] = size(X{1,1});
KC = K*C;
Cov = cell(1,M);
Sig_Cov = zeros(KC,KC);
for m = 1:M
    X_m = X{m};
    X_m_hat = [];
    for k = 1:K
        n_delay = (k-1)*tau;
        if n_delay ==0
            X_order_k = X_m;
        else
            X_order_k(:,1:n_delay) = 0;
            X_order_k(:,n_delay+1:T) = X_m(:,1:T-n_delay);
        end
        X_m_hat = cat(1,X_m_hat,X_order_k);
    end
    Cov{1,m} = X_m_hat*X_m_hat';
    Cov{1,m}= Cov{1,m}/trace(Cov{1,m}); % trace normalizaiton
    Sig_Cov = Sig_Cov + Cov{1,m};
end

Cov_whiten = zeros(M,KC,KC);
for m = 1:M
    R_test(m,:) =  vec(logm(squeeze(Cov{1,m})));% logm
      cov_test(:,:,m) = logm(squeeze(Cov{1,m}));
end 
end

function log_det_X = calculate_log_det(X)
    % This function calculates the log determinant of a matrix X
    % by normalizing its diagonal elements to avoid infinite values.
    n = size(X,1); % Get the size of matrix X
    c = 10^floor(log10(X(1,1))); % Extract the scaling factor c as a power of 10
    A = X / c; % Normalize the matrix by the scaling factor
    L = chol(A, 'lower'); % Perform Cholesky decomposition on A
    log_det_A= 2 * sum(log(diag(L)));% Compute the log determinant of the normalized matrix via L
%      log_det_A = log(det(A)); 
    log_det_X = n*log(c) + log_det_A; % Combine the results to get the log determinant of the original matrix
end