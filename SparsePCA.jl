using LinearAlgebra
using ProgressMeter
using ElasticPDMats
using SparseArrays
using GLMNet


function SPCA(X, n_components, l1, l2 = 0.01, max_iter = 10000, tol = 0.0001)
    """
    SPCA algorithm for simultaneous sparse coding and principal component analysis.
    
    Parameters:
        X: Array of shape (n_samples, n_features)
            The input data matrix.
        n_components: Int
            Number of principal components to retain.
        l1: Float64
            Sparsity controlling parameter. Higher values lead to sparser components.
        l2: Float64
            Amount of ridge shrinkage to apply in order to improve conditioning when calling the transform method.
        
        max_iter: Int
            Maximum number of iterations for the optimization.
        tol: Float64
            Maximum allowed tolerance
    
    Returns:
        sparse_components : Array of shape (n_samples, n_components)
            The sparse codes for the input data.
        loadings: Array of shape (n_features, n_components)
            The projection matrix for the principal components.
    """

    cor_mat = X' * X
    eigen_decomp = eigen(cor_mat)
    V = eigen_decomp.vectors[:, sortperm(eigen_decomp.values, rev=true)]
    A = V[:, 1:n_components]
    B = zeros(size(V, 1), n_components)

    iter = 0
    pbar = Progress(max_iter+1)
    while iter < max_iter
        diff = 0
        B_temp = zeros(size(V, 1), n_components)

        for j in 1:size(A, 2)
            y_s = X * A[:, j]
            x_s = X
            alpha = l1[j] + 2 * l2
            l1_ratio = l1[j] / (l1[j] + 2 * l2)
        
            elas_net = glmnet(x_s, y_s, alpha=l1_ratio, lambda=[alpha])
            B_temp[:, j] = elas_net.betas
        end
        

        diff = norm(B_temp - B)
        B = B_temp

        cor_B = cor_mat * B
        U, D, V_ = svd(cor_B)
        A = U * V_'

        iter += 1
        if diff < tol
            break
        elseif iter == max_iter - 1
            println("Max Iterations reached")
        end

        next!(pbar)
    end

    loadings = B / norm(B)
    sparse_components = X * loadings
    return sparse_components, loadings
end
