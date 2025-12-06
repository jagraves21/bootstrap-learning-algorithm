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

# Computes pairwise distances between (X,y) and (X,y_hat) in a loop.
compute_pairwise_distances <- function(X, y, y_hat) {
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

	return(distances)
}

# Computes pairwise distances between (X,y) and (X,y_hat) using vectorized
# operations for speed.
compute_pairwise_distances_vectorized <- function(X, y, y_hat) {
	# ensure y and y_hat are vectors (no copy if single-column matrices)
	y <- y[, 1, drop = TRUE]
	y_hat <- y_hat[, 1, drop = TRUE]

	# pairwise squared distances for X using optimized matrix ops
	sum_X <- rowSums(X^2)
	distances_X_sq <- outer(sum_X, sum_X, "+") - 2 * (X %*% t(X))
	distances_X_sq[distances_X_sq < 0] <- 0  # numerical stability

	# pairwise squared distances for y using BLAS-friendly matrix ops
	distances_y_sq <- matrix(y^2, nrow = length(y), ncol = length(y_hat)) +
		matrix(y_hat^2, nrow = length(y), ncol = length(y_hat), byrow = TRUE) -
		2 * (y %*% t(y_hat))
	distances_y_sq[distances_y_sq < 0] <- 0  # numerical stability

	# Combine and take square root
	sqrt(distances_X_sq + distances_y_sq)
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
	#distances <- compute_pairwise_distances(X, y, y_hat)
	distances <- compute_pairwise_distances_vectorized(X, y, y_hat)

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
update_weights <- function(network, r, X, Z1_hat, H_hat, y, lambda = 0) {
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
	
	if (lambda != 0) {
		# Add ridge regularization (lambda * I) to improve numerical stability.
		# This is especially useful if A1_acc or A2_acc is nearly singular or
		# ill-conditioned.
		A1_acc <- A1_acc + lambda * diag(ncol(X))
		A2_acc <- A2_acc + lambda * diag(ncol(H_hat))
	}

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

pretty_print_log <- function(log_entry) {
	cat("\n")
	cat(sprintf(
		"==============================  Epoch %4d  ==============================\n",
		log_entry$epoch
	))

	cat(sprintf(
		"MSE: %.6f\n\n",
		log_entry$mse
	))

	# Weights W1
	cat("Weights W1\n")
	cat(sprintf(
		"\t\u2016W1\u2016: %10.4f   Rank: %3d   Cond: %10.4e\n",
		log_entry$W1_norm,
		log_entry$W1_rank,
		log_entry$W1_cond
	))
	cat(sprintf(
		"\tChange  mean: %10.6f   min: %10.6f   max: %10.6f\n\n",
		log_entry$W1_mean_chnage,
		log_entry$W1_min_chnage,
		log_entry$W1_max_chnage
	))

	# Weights W2
	cat("Weights W2\n")
	cat(sprintf(
		"\t\u2016W2\u2016: %10.4f   Rank: %3d   Cond: %10.4e\n",
		log_entry$W2_norm,
		log_entry$W2_rank,
		log_entry$W2_cond
	))
	cat(sprintf(
		"\tChange  mean: %10.6f   min: %10.6f   max: %10.6f\n\n",
		log_entry$W2_mean_chnage,
		log_entry$W2_min_chnage,
		log_entry$W2_max_chnage
	))

	# Accumulator A1
	cat("Accumulator A1 (hat)\n")
	cat(sprintf(
		"\tNorm: %10.4f   Rank: %3d   Cond: %10.4e\n\n",
		log_entry$A1_norm,
		log_entry$A1_rank,
		log_entry$A1_cond
	))

	# Accumulator A2
	cat("Accumulator A2 (hat)\n")
	cat(sprintf(
		"\tNorm: %10.4f   Rank: %3d   Cond: %10.4e\n\n",
		log_entry$A2_norm,
		log_entry$A2_rank,
		log_entry$A2_cond
	))

	# Activation Regimes
	cat("Activation Regimes\n")
	cat(sprintf(
		"\tStrong: %7.4f   Soft: %7.4f   Linear: %7.4f\n",
		log_entry$strong_sat,
		log_entry$soft_sat,
		log_entry$linear_region
	))

	cat("============================================================================\n")
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

	# full dataset mse
	fwd_full <- forward_pass(network, X)
	mse <- mean((fwd_full$output - y)^2)
	strong_sat <- sum(abs(fwd_full$Z1) > 2.5) / length(fwd_full$Z1)
	soft_sat   <- sum(abs(fwd_full$Z1) > 2.0) / length(fwd_full$Z1)
	linear_region <- sum(abs(fwd_full$Z1) < 0.1) / length(fwd_full$Z1)

	# preallocate logging list
	training_log_list <- vector("list", num_epochs + 1)
	training_log_list[[1]] <- data.frame(
		epoch          = 0,
		mse            = mse,

		A1_norm        = NA,
		A1_rank        = NA,
		A1_cond        = NA,

		A2_norm        = NA,
		A2_rank        = NA,
		A2_cond        = NA,

		W1_norm        = norm(network$W1, type="F"),
		W1_rank        = qr(network$W1)$rank,
		W1_cond        = tryCatch(kappa(network$W1), error = function(e) NA),
		W1_mean_chnage = NA,
		W1_min_chnage  = NA,
		W1_max_chnage  = NA,
		
		W2_norm        = norm(network$W2, type="F"),
		W2_rank        = qr(network$W2)$rank,
		W2_cond        = tryCatch(kappa(network$W2), error = function(e) NA),
		W2_mean_chnage = NA,
		W2_min_chnage  = NA,
		W2_max_chnage  = NA,
		
		strong_sat     = strong_sat,
		soft_sat       = soft_sat, 
		linear_region  = linear_region
	)
	last_log <- training_log_list[[1]]
	pretty_print_log(last_log)

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

		# store per-epoch weight changes
		W1_changes <- c()
		W2_changes <- c()
			
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
				X = X_batch, y = y_batch, y_hat = fwd_batch$output,
				delta = delta, temperature = temperature
			)

			# save old weights
			old_W1 <- network$W1
			old_W2 <- network$W2

			# update the weights
			network <- update_weights(
				network = network,
				r = ifelse(epoch >= 1, length(X_batch), 0),
				X = X_batch[1:length(bootstrap_indices),, drop = FALSE],
				Z1_hat = fwd_batch$Z1[bootstrap_indices,, drop = FALSE],
				H_hat = fwd_batch$A1[bootstrap_indices,, drop = FALSE],
				y = y_batch[1:length(bootstrap_indices),, drop = FALSE],
				lambda = 0 #1e-4
			)

			# compute Frobenius norms
			dW1 <- norm(network$W1 - old_W1, type = "F")
			dW2 <- norm(network$W2 - old_W2, type = "F")

			W1_changes <- c(W1_changes, dW1)
			W2_changes <- c(W2_changes, dW2)
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
		
		# full dataset mse
		fwd_full <- forward_pass(network, X)
		mse <- mean((fwd_full$output - y)^2)
		strong_sat <- sum(abs(fwd_full$Z1) > 2.5) / length(fwd_full$Z1)
		soft_sat   <- sum(abs(fwd_full$Z1) > 2.0) / length(fwd_full$Z1)
		linear_region <- sum(abs(fwd_full$Z1) < 0.1) / length(fwd_full$Z1)

		training_log_list[[epoch + 1]] <- data.frame(
			epoch          = epoch,
			mse            = mse,

			A1_norm        = norm(network$A1_hat, type="F"),
			A1_rank        = qr(network$A1_hat)$rank,
			A1_cond        = tryCatch(kappa(network$A1_hat), error = function(e) NA),
			
			A2_norm        = norm(network$A2_hat, type="F"),
			A2_rank        = qr(network$A2_hat)$rank,
			A2_cond        = tryCatch(kappa(network$A2_hat), error = function(e) NA),

			W1_norm        = norm(network$W1, type="F"),
			W1_rank        = qr(network$W1)$rank,
			W1_cond        = tryCatch(kappa(network$W1), error = function(e) NA),
			W1_mean_chnage = mean(W1_changes),
			W1_min_chnage  = min(W1_changes),
			W1_max_chnage  = max(W1_changes),
			
			W2_norm        = norm(network$W2, type="F"),
			W2_rank        = qr(network$W2)$rank,
			W2_cond        = tryCatch(kappa(network$W2), error = function(e) NA),
			W2_mean_chnage = mean(W2_changes),
			W2_min_chnage  = min(W2_changes),
			W2_max_chnage  = max(W2_changes),
			
			strong_sat     = strong_sat,
			soft_sat       = soft_sat, 
			linear_region  = linear_region
		)
		last_log <- training_log_list[[epoch + 1]]
		pretty_print_log(last_log)
	}

	training_log <- do.call(rbind, training_log_list)

	if (!is.null(extra)) {
		plot_logs(training_log, dir = extra$plot_dir, vline_epoch = 21)

		output_file = file.path(extra$plot_dir, "predictions.gif")
		create_gif_from_frames(
			frame_dir = extra$frame_dir,
			output_file = output_file,
			fps = extra$fps
		)

	}

	return(network)
}

