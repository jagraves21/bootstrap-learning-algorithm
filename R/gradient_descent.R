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

