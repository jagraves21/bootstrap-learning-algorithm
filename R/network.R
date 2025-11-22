# ========================================
# Explicit Bias Neural Network
# ========================================

# Representation Conventions and Shapes
# (Bias vectors are separate from weights)

# Input:
# - X: Input matrix of shape (batch_size × input_size), each row is one sample.

# Hidden layer:
# - W1: Weights of shape (hidden_size × input_size), each row is a hidden neuron
# - b1: Bias vector of length hidden_size, added to each row of X %*% t(W1)
# - Z1: Pre-activation matrix (batch_size × hidden_size)
#       Z1 <- X %*% t(W1) + b1
# - A1: Activation matrix (batch_size × hidden_size), e.g., A1 <- tanh(Z1)

# Output layer:
# - W2: Weights of shape (output_size × hidden_size), each row is an output neuron
# - b2: Bias vector of length output_size, added to each row of A1 %*% t(W2)
# - Z2: Pre-activation matrix (batch_size × output_size)
#       Z2 <- A1 %*% t(W2) + b2
# - Output: Activation matrix (batch_size × output_size), final outputs
#           (identity)

# Notes:
# - Each row of X, Z1, A1, Z2, and Output represents one sample
# - Each column corresponds to one neuron
# - Ensures consistent row-vector sample representation and column-wise
#   neuron alignment

# Initializes a two-layer neural network using explicit bias vectors.
# The network computes layer activations of the form: Z = X %*% t(W) + b.
init_network_with_bias_vectors <- function(
	input_size, hidden_size, output_size
) {
	W1 <- matrix(
		rnorm(hidden_size * input_size, sd = 0.5),
		nrow = hidden_size, ncol = input_size
	)
	b1 <- rnorm(hidden_size, sd = 0.5) # vector, not matrix

	W2 <- matrix(
		rnorm(output_size * hidden_size, sd = 0.5),
		nrow = output_size, ncol = hidden_size
	)
	b2 <- rnorm(output_size, sd = 0.5) # vector, not matrix

	network <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, type = "bias_vector")
	return(network)
}

# Computes the forward pass of the network when using explicit bias vectors.
forward_pass_with_bias_vectors <- function(network, X) {
	# if X is not a matrix, turn it into one with correct number of features
	if (!is.matrix(X)) {
		input_dim <- ncol(network$W1)
		X <- matrix(X, ncol = input_dim)
	}

	Z1 <- sweep(X %*% t(network$W1), 2, network$b1, "+")
	A1 <- tanh(Z1)

	Z2 <- sweep(A1 %*% t(network$W2), 2, network$b2, "+")
	A2 <- Z2 # identity activation

	results <- list(Z1 = Z1, A1 = A1, Z2 = Z2, output = A2)
	return(results)
}


# Trains a two-layer neural network with explicit bias vectors using mini-batch
# gradient descent.
train_network_with_bias_vectors_gd <- function(
	network, X, y, epochs = 2000, batch_size = 32, shuffle_batches = TRUE,
	lr = 0.01, seed = NULL
) {
	if (!is.null(seed)) set.seed(seed)
	if (!is.matrix(X)) X <- matrix(X, ncol = 1)
	if (!is.matrix(y)) y <- matrix(y, ncol = 1)
			
	fwd_full <- forward_pass_with_bias_vectors(network, X)
	mse <- mean((fwd_full$output - y)^2)
	cat("Epoch", 0, "MSE:", mse, "\n")

	num_samples <- nrow(X)

	for (epoch in 1:epochs) {
		# shuffle data
		if (shuffle_batches) {
			indices <- sample(1:num_samples)
			X_shuffled <- X[indices,, drop = FALSE]
			y_shuffled <- y[indices,, drop = FALSE]
		} else {
			X_shuffled <- X
			y_shuffled <- y
		}

		for (start_idx in seq(1, num_samples, by = batch_size)) {
			end_idx <- min(start_idx + batch_size - 1, num_samples)
			X_batch <- X_shuffled[start_idx:end_idx,, drop = FALSE]
			y_batch <- y_shuffled[start_idx:end_idx,, drop = FALSE]

			# forward pass
			fwd <- forward_pass_with_bias_vectors(network, X_batch)
			error <- fwd$output - y_batch

			# calculate gradients
			grad_output_pre <- error
			grad_W2 <- t(grad_output_pre) %*% fwd$A1 / nrow(X_batch)
			grad_b2 <- colMeans(grad_output_pre)

			grad_hidden_act <- grad_output_pre %*% network$W2
			grad_hidden_pre <- grad_hidden_act * (1 - fwd$A1^2)
			grad_W1 <- t(grad_hidden_pre) %*% X_batch / nrow(X_batch)
			grad_b1 <- colMeans(grad_hidden_pre)

			# update weights and biases
			network$W2 <- network$W2 - lr * grad_W2
			network$b2 <- network$b2 - lr * grad_b2
			network$W1 <- network$W1 - lr * grad_W1
			network$b1 <- network$b1 - lr * grad_b1
		}

		# print mse every 10 epochs
		if (epoch %% 10 == 0 || epoch == epochs) {
			# forward pass full dataset
			fwd_full <- forward_pass_with_bias_vectors(network, X)
			mse <- mean((fwd_full$output - y)^2)
			cat("Epoch", epoch, "MSE:", mse, "\n")
		}
	}

	return(network)
}



# ========================================
# Absorbed Bias Neural Network
# ========================================

# Representation Conventions and Shapes
# (Bias is absorbed into weight matrices as the last column)

# Input:
# - X: Input matrix of shape (batch_size × input_size), each row is one sample.

# Hidden layer:
# - W1: Weights of shape (hidden_size × (input_size + 1)), last column is bias
# - X_bias: Augmented input matrix (batch_size × (input_size + 1))
#           X_bias <- cbind(X, 1)
# - Z1: Pre-activation matrix (batch_size × hidden_size)
#       Z1 <- X_bias %*% t(W1)
# - A1: Activation matrix (batch_size × hidden_size), e.g., A1 <- tanh(Z1)

# Output layer:
# - W2: Weights of shape (output_size × (hidden_size + 1)), last column is bias
# - A1_bias: Augmented hidden activation matrix (batch_size × (hidden_size + 1))
#            A1_bias <- cbind(A1, 1)
# - Z2: Pre-activation matrix (batch_size × output_size)
#       Z2 <- A1_bias %*% t(W2)
# - Output: Activation matrix (batch_size × output_size), final outputs
#           (identity)

# Notes:
# - Each row of X, X_bias, Z1, A1, A1_bias, Z2, and Output represents one sample
# - Each column corresponds to one neuron
# - Bias is folded into weights as the last column; no separate bias vectors are used.
#   This convention prepends a -1 to the input/hidden activations, making the last
#   column of W1/W2 the bias. This is consistent across forward pass and gradient
#   updates.
# - Ensures consistent row-vector sample representation and column-wise neuron
#   alignment

# Computes the forward pass for networks where bias is absorbed into weights.
init_network_with_absorbed_bias <- function(
	input_size, hidden_size, output_size
) {
	W1 <- matrix(
		rnorm(hidden_size * (input_size+1), sd = 0.5),
		nrow = hidden_size, ncol = input_size+1
	)

	W2 <- matrix(
		rnorm(output_size * (hidden_size+1), sd = 0.5),
		nrow = output_size, ncol = hidden_size+1
	)

	network <- list(W1 = W1, W2 = W2, type = "absorbed_bias")
	return(network)
}

# Computes the forward pass for networks where bias is absorbed into weights.
forward_pass_with_absorbed_bias <- function(network, X) {
	# if X is not a matrix, turn it into one with correct number of features
	if (!is.matrix(X)) {
		input_dim <- ncol(network$W1) - 1
		X <- matrix(X, ncol = input_dim)
	}

	X_bias <- cbind(X, 1)

	Z1 <- X_bias %*% t(network$W1)
	A1 <- tanh(Z1)

	A1_bias <- cbind(A1, 1)

	Z2 <- A1_bias %*% t(network$W2)
	A2 <- Z2 # identity activation

	results <- list(Z1 = Z1, A1 = A1, Z2 = Z2, output = A2)
	return(results)
}

# Trains a two-layer neural network with absorbed bias (bias folded into
# weights using mini-batch gradient descent.
train_network_with_absorbed_bias_gd <- function(
	network, X, y, epochs = 2000, batch_size = 32, shuffle_batches = TRUE,
	lr = 0.01, seed = NULL
) {
	if (!is.null(seed)) set.seed(seed)
	if (!is.matrix(X)) X <- matrix(X, ncol = 1)
	if (!is.matrix(y)) y <- matrix(y, ncol = 1)
			
	fwd_full <- forward_pass_with_absorbed_bias(network, X)
	mse <- mean((fwd_full$output - y)^2)
	cat("Epoch", 0, "MSE:", mse, "\n")

	num_samples <- nrow(X)

	for (epoch in 1:epochs) {
		# shuffle data
		if (shuffle_batches) {
			indices <- sample(1:num_samples)
			X_shuffled <- X[indices,, drop = FALSE]
			y_shuffled <- y[indices,, drop = FALSE]
		} else {
			X_shuffled <- X
			y_shuffled <- y
		}

		for (start_idx in seq(1, num_samples, by = batch_size)) {
			end_idx <- min(start_idx + batch_size - 1, num_samples)
			X_batch <- X_shuffled[start_idx:end_idx,, drop = FALSE]
			y_batch <- y_shuffled[start_idx:end_idx,, drop = FALSE]

			# forward pass with absorbed bias
			fwd <- forward_pass_with_absorbed_bias(network, X_batch)
			batch_size_actual <- nrow(X_batch)

			# compute error
			error <- fwd$output - y_batch

			# add bias columns for gradient computation
			A1_bias <- cbind(fwd$A1, 1)
			X_bias <- cbind(X_batch, 1)

			# gradients
			grad_W2 <- t(error) %*% A1_bias / batch_size_actual

			# hidden only
			grad_hidden <- error %*% network$W2[, -ncol(network$W2), drop = FALSE]
			# derivative of tanh
			grad_hidden_pre <- grad_hidden * (1 - fwd$A1^2)
			grad_W1 <- t(grad_hidden_pre) %*% X_bias / batch_size_actual

			# update weights
			network$W2 <- network$W2 - lr * grad_W2
			network$W1 <- network$W1 - lr * grad_W1
		}

		# print mse every 10 epochs
		if (epoch %% 10 == 0 || epoch == epochs) {
			# forward pass full dataset
			fwd_full <- forward_pass_with_absorbed_bias(network, X)
			mse <- mean((fwd_full$output - y)^2)
			cat("Epoch", epoch, "MSE:", mse, "\n")
		}
	}

	return(network)
}



# =====================================================
# High-Level Wrapper Functions for Two-Layer Networks
# =====================================================

# High-level initializer that selects between explicit bias vectors or absorbed
# bias representation.
init_network <- function(
	input_size, hidden_size, output_size, absorbed_bias = FALSE
) {
	if (absorbed_bias) {
		network <- init_network_with_absorbed_bias(input_size, hidden_size,
			output_size)
	} else {
		network <- init_network_with_bias_vectors(input_size, hidden_size,
			output_size)
	}

	return(network)
}

# Wrapper forward-pass function that dispatches to the correct implementation
# based on the network type.
forward_pass <- function(network, X) {
	if (network$type == "absorbed_bias") {
		output <- forward_pass_with_absorbed_bias(network, X)
	} else {
		output <- forward_pass_with_bias_vectors(network, X)
	}

	return(output)
}

# Wrapper training function that dispatches to the correct gradient descent
# implementation based on network type.
train_network_gd <- function(
	network, X, y, epochs = 2000, batch_size = 32, shuffle_batches = TRUE,
	lr = 0.01, seed = NULL
) {
	if (network$type == "absorbed_bias") {
		network <- train_network_with_absorbed_bias_gd(
			network, X, y,
			epochs = epochs,
			batch_size = batch_size,
			shuffle_batches = shuffle_batches,
			lr = lr,
			seed = seed
		)
	} else {
		network <- train_network_with_bias_vectors_gd(
			network, X, y,
			epochs = epochs,
			batch_size = batch_size,
			shuffle_batches = shuffle_batches,
			lr = lr,
			seed = seed
		)
	}

	return(network)
}



# ========================================
# Bootstrap Learning Algorithm (BLA)
# ========================================

# Computes a softmin distribution over a numeric vector, optionally returning
# only the top delta elements.
softmin <- function(x, delta = NULL, temperature = 1) {
	scores <- exp(-x / temperature)
	scores[is.na(scores)] <- 0

	normalize <- function(s) {
		if (sum(s) == 0) {
			rep(1 / length(s), length(s))
		} else {
			s / sum(s)
		}
	}

	if (is.null(delta)) {
		return(normalize(scores))
	} else {
		top_indices <- order(scores, decreasing = TRUE)[1:delta]
		return(list(
			top_indices = top_indices,
			probs = normalize(scores[top_indices])
		))
	}
}

# Generates a bootstrap sample of row indices from the dataset using a softmin
# weighting based on distances between samples and predicted values.
bootstrap_sample <- function(
	X, y, y_hat, delta = NULL, temperature = 1, seed = NULL
) {
	if (!is.null(seed)) set.seed(seed)

	num_samples <- nrow(X)
	mbb <- sample(num_samples,2)
	l <- min(mbb)
	u <- max(mbb)

	chosen_indices <- integer(u)   # preallocate storage
	for (n in 1:u) {
		dists <- rowSums(
			(X[n,] - X)^2 + (y[n] - y_hat)^2
		)

		weights = softmin(dists, delta = delta, temperature = temperature)
		if (is.null(delta)) {
			chosen <- sample(length(weights), 1, prob = weights)
		}
		else {
			chosen <- sample(weights$top_indices, 1, prob = weights$probs)
		}

		chosen_indices[n] <- chosen
	}
			
	return(chosen_indices)
}

# Generates a bootstrap sample of row indices from the dataset using a softmin
# weighting based on distances between samples and predicted values.
bootstrap_sample2 <- function(
  X, y, y_hat, delta = NULL, temperature = 1, seed = NULL
) {
	if (!is.null(seed)) set.seed(seed)
	num_samples <- nrow(X)
	num_features <- ncol(X)
	
	# compute pairwise distances (rows)
	distances <- matrix(0, nrow = num_samples, ncol = num_samples)
	for (i in 1:num_samples) {
		# subtract row vector from each row
		x_diff <- rowSums((X[i, , drop = TRUE] - X)^2)
		# scalar difference
		y_diff <- (y[i] - y_hat)^2
		distances[i, ] <- sqrt(x_diff + y_diff)
	}
	
	bootstrap_indices <- integer(num_samples)
	for (i in 1:num_samples) {
		if (is.null(delta)) {
			probs <- softmin(distances[i, ], temperature = temperature)
			bootstrap_indices[i] <- sample(1:num_samples, 1, prob = probs)
		} else {
			top_idx <- order(distances[i, ])[1:delta]
			probs <- softmin(
				distances[i, top_idx], delta = delta, temperature = temperature
			)$probs
			bootstrap_indices[i] <- sample(top_idx, 1, prob = probs)
		}
	}
	
	return(bootstrap_indices)
}

# Performs iterative refinement to solve the linear system A_acc W^T = b_acc^T.
# This improves numerical stability compared to direct inversion, especially
# when A_acc is ill-conditioned.
iterative_refine <- function(
	W_init, A_acc, b_acc, mu = NULL, max_iter = 1000, tol = 1e-6
) {
	W <- W_init

	if (is.null(mu)) {
		eig_vals <- eigen(A_acc, symmetric = TRUE, only.values = TRUE)$values
		mu <- 1.95 / (max(eig_vals) + min(eig_vals))
	}

	for (iter in 1:max_iter) {
		W_old <- W

		delta_W <- mu * ( t(b_acc) - t(A_acc %*% t(W)) )

		W <- W + delta_W

		if (norm(W - W_old, type = "F") < tol) {
			break
		}
	}

	return(W)
}

# Recursive Least Squares (RLS) with Iterative Refinement (IR)
#
# 1. For each layer, form the batch normal-equation contributions:
#       Layer 1:   SumA1 = X^T X,       Sumb1 = X^T Z1_hat
#       Layer 2:   SumA2 = H_hat^T H_hat, Sumb2 = H_hat^T y
#
# 2. Update the recursive accumulators (RLS):
#       A1_acc = (r / (r+n)) * A1_prev + (1 / (r+n)) * SumA1
#       b1_acc = (r / (r+n)) * b1_prev + (1 / (r+n)) * Sumb1
#       A2_acc = (r / (r+n)) * A2_prev + (1 / (r+n)) * SumA2
#       b2_acc = (r / (r+n)) * b2_prev + (1 / (r+n)) * Sumb2
#     where n = batch size, r = prior weight / forgetting factor.
#
# 3. Solve the normal equations for each layer's weights:
#       Layer 1:   A1_acc W1^T = b1_acc
#       Layer 2:   A2_acc W2^T = b2_acc
#
#    Instead of directly computing W = (A_acc^{-1} b_acc)^T, use iterative
#    refinement:
#       W <- W_old + mu * (b_acc^T - (A_acc %*% W_old^T)^T)
#    This iteratively corrects W toward the least-squares solution in a
#    numerically stable way, even if A_acc is ill-conditioned.
#
# 4. After convergence, store the updated W1, W2, and accumulators back in the
#    network.
#
# In essence, this combines:
#    - Recursive Least Squares: incremental accumulation of normal equations
#    - Iterative Refinement: stable solution of the normal equations
update_weights <- function(network, r, X, Z1_hat, H_hat, y) {
	if (network$type == "absorbed_bias") {
		X <- cbind(X, 1)
		H_hat <- cbind(H_hat, 1)
	}

	num_samples <- nrow(X)

	# batch normal-equation contributions
	SumA1 <- t(X) %*% X                # (d_in × d_in)
	Sumb1 <- t(X) %*% Z1_hat           # (d_in × d_hid)

	SumA2 <- t(H_hat) %*% H_hat        # (d_hid × d_hid)
	Sumb2 <- t(H_hat) %*% y            # (d_hid × d_out)

	# recursive update of normal-equation accumulators
	A1_acc <- (r / (r + num_samples)) * network$A1_hat + 
		(1 / (r + num_samples)) * SumA1

	b1_acc <- (r / (r + num_samples)) * network$b1_hat + 
		(1 / (r + num_samples)) * Sumb1

	A2_acc <- (r / (r + num_samples)) * network$A2_hat + 
		(1 / (r + num_samples)) * SumA2

	b2_acc <- (r / (r + num_samples)) * network$b2_hat + 
		(1 / (r + num_samples)) * Sumb2

	# store accumulators back
	network$A1_hat <- A1_acc
	network$b1_hat <- b1_acc
	network$A2_hat <- A2_acc
	network$b2_hat <- b2_acc

	## -------------------------------------------------------------------------
	## Direct solve() calls are commented out because A1_acc / A2_acc
	## can be ill-conditioned or nearly singular. 
	## Iterative refinement is used instead to compute W1 and W2 stably.
	## -------------------------------------------------------------------------

	## Solve the normal equations for W1 and W2 (commented out)
	## W1 satisfies: (X^T X) W1 = X^T Z1_hat
	##   A1_acc is (d_in × d_in), b1_acc is (d_in × d_hid)
	##   W1 = (A1_acc^{-1} b1_acc)^T
	#W1 <- t(solve(A1_acc, b1_acc))

	## W2 satisfies: (H^T H) W2 = H^T y
	##   A2_acc is (d_hid × d_hid), b2_acc is (d_hid × d_out)
	##   W2 = (A2_acc^{-1} b2_acc)^T
	#W2 <- t(solve(A2_acc, b2_acc))

	# --------------------------------------------------------------------------
	# Initialize W1 and W2 for iterative refinement.
	# Can start from zeros (cold start) or use the current network weights
	# (warm start). Using the current weights improves convergence.
	# --------------------------------------------------------------------------
	# W1_init <- matrix(0, nrow = nrow(network$W1), ncol = ncol(network$W1))
	# W2_init <- matrix(0, nrow = nrow(network$W2), ncol = ncol(network$W2))
	W1_init <- network$W1
	W2_init <- network$W2

	# iterative refinement
	W1 <- iterative_refine(W1_init, A1_acc, b1_acc)
	W2 <- iterative_refine(W2_init, A2_acc, b2_acc)

	# store the results
	network$W1 <- W1
	network$W2 <- W2

	return(network)
}

train_network_bla <- function(
	network, X, y, num_epochs = 100, batch_size = NULL, temperature = 1,
	shuffle_batches = TRUE, delta = NULL, seed = NULL, extra = NULL
) {
	if (!is.null(seed)) set.seed(seed)
	if (!is.matrix(X)) X <- matrix(X, ncol = 1)
	if (!is.matrix(y)) y <- matrix(y, ncol = 1)
	num_samples <- nrow(X)
	if (is.null(batch_size)) batch_size <- num_samples

	if (!is.null(extra)) {
		init_frame_dir(extra$frame_dir)
	}

	# initialize accumulators
	network$A1_hat <- 0
	network$b1_hat <- 0
	network$A2_hat <- 0
	network$b2_hat <- 0
		
	if (!is.null(extra)) {
		y_pred <- forward_pass(
			network, matrix(extra$x_plot, ncol = 1)
		)$output
		save_training_frame(
			epoch = 0, frame_dir = extra$frame_dir,
			frame_width = extra$frame_width, frame_height = extra$frame_height,
			coefficients = extra$coefficients, x_plot = extra$x_plot,
			y_true = extra$y_true, y_pred = y_pred, xlim = extra$xlim,
			ylim = extra$ylim
		)
	}
	
	# full dataset mse
	fwd_full <- forward_pass(network, X)
	mse <- mean((fwd_full$output - y)^2)
	cat("Epoch", 0, "MSE:", mse, "\n")

	# for each epcoh
	for (epoch in 1:num_epochs) {
		# shuffle data
		if (shuffle_batches) {
			indices <- sample(1:num_samples)
			X_shuffled <- X[indices,, drop = FALSE]
			y_shuffled <- y[indices,, drop = FALSE]
		} else {
			X_shuffled <- X
			y_shuffled <- y
		}

		# for each mini batch
		for (start_idx in seq(1, num_samples, by = batch_size)) {
			# get the mini batch
			end_idx <- min(start_idx + batch_size - 1, num_samples)
			X_batch <- X_shuffled[start_idx:end_idx,, drop = FALSE]
			y_batch <- y_shuffled[start_idx:end_idx,, drop = FALSE]

			# forward step for predictions
			fwd_batch <- forward_pass(network, X_batch)
			
			# bootstrap sampling
			bootstrap_indices <- bootstrap_sample2(
				X = X_batch, y = y_batch, y_hat = fwd_batch$output, delta = delta, temperature = temperature
			)

			# update the weights
			network <- update_weights(
				network = network,
				r = ifelse(epoch > 1, length(X_batch), 0),
				X = X_batch[1:length(bootstrap_indices),, drop = FALSE],
				Z1_hat = fwd_batch$Z1[bootstrap_indices,, drop = FALSE],
				H_hat = fwd_batch$A1[bootstrap_indices,, drop = FALSE],
				y = y_batch[1:length(bootstrap_indices),, drop = FALSE]
			)
		}

		if (!is.null(extra)) {
			y_pred <- forward_pass(
				network, matrix(extra$x_plot, ncol = 1)
			)$output
			save_training_frame(
				epoch = epoch, frame_dir = extra$frame_dir,
				frame_width = extra$frame_width,
				frame_height = extra$frame_height,
				coefficients = extra$coefficients, x_plot = extra$x_plot,
				y_true = extra$y_true, y_pred = y_pred, xlim = extra$xlim,
				ylim = extra$ylim
			)
		}

		if (TRUE || epoch %% 10 == 0) {
			# full dataset mse
			fwd_full <- forward_pass(network, X)
			mse <- mean((fwd_full$output - y)^2)
			cat("Epoch", epoch, "MSE:", mse, "\n")
		}
	}

	if (!is.null(extra)) {
		create_gif_from_frames(
			frame_dir = extra$frame_dir,
			output_file = extra$output_file,
			fps = extra$fps
		)
	}

	return(network)
}

