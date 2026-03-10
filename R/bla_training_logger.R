# Computes summary statistics for a weight matrix, optionally using a
# vector of changes from the previous state. Returns a list containing
# Frobenius norm, rank, condition number, and change statistics if provided.
weight_stats <- function(W, changes = NULL) {
	list(
		norm = norm(W, type = "F"),
		rank = qr(W)$rank,
		cond = tryCatch(kappa(W), error = function(e) NA),
		mean_change = if (is.null(changes)) NA else mean(changes),
		min_change  = if (is.null(changes)) NA else min(changes),
		max_change  = if (is.null(changes)) NA else max(changes)
	)
}

# Computes saturation metrics for hidden-layer activations Z1.
# Returns the fraction of elements in strong saturation, soft saturation,
# and linear region.
saturation_stats <- function(Z1) {
	list(
		strong_sat = sum(abs(Z1) > 2.5) / length(Z1),
		soft_sat   = sum(abs(Z1) > 2.0) / length(Z1),
		linear_region = sum(abs(Z1) < 0.1) / length(Z1)
	)
}

# Creates a log row for a given epoch containing MSE, weight statistics,
# accumulator statistics, and activation regime metrics.
create_log_row <- function(
	epoch, mse, network, fwd_full, W1_changes = NULL, W2_changes = NULL
) {

	sat <- saturation_stats(fwd_full$Z1)
	W1 <- weight_stats(network$W1, W1_changes)
	W2 <- weight_stats(network$W2, W2_changes)

	# if epoch 0, set A1/A2 metrics to NA
	if (epoch == 0) {
		A1_norm <- NA
		A1_rank <- NA
		A1_cond <- NA
		A1_det  <- NA

		A2_norm <- NA
		A2_rank <- NA
		A2_cond <- NA
		A2_det  <- NA
	} else {
		A1_norm <- norm(network$A1_hat, type = "F")
		A1_rank <- qr(network$A1_hat)$rank
		A1_cond <- tryCatch(kappa(network$A1_hat), error = function(e) NA)
		A1_det  <- tryCatch(det(network$A1_hat), error = function(e) NA)

		A2_norm <- norm(network$A2_hat, type = "F")
		A2_rank <- qr(network$A2_hat)$rank
		A2_cond <- tryCatch(kappa(network$A2_hat), error = function(e) NA)
		A2_det  <- tryCatch(det(network$A2_hat), error = function(e) NA)
	}

	data.frame(
		epoch = epoch,
		mse = mse,

		A1_norm = A1_norm,
		A1_rank = A1_rank,
		A1_cond = A1_cond,
		A1_det  = A1_det,

		A2_norm = A2_norm,
		A2_rank = A2_rank,
		A2_cond = A2_cond,
		A2_det  = A2_det,

		W1_norm = W1$norm,
		W1_rank = W1$rank,
		W1_cond = W1$cond,
		W1_mean_chnage = W1$mean_change,
		W1_min_chnage  = W1$min_change,
		W1_max_chnage  = W1$max_change,

		W2_norm = W2$norm,
		W2_rank = W2$rank,
		W2_cond = W2$cond,
		W2_mean_chnage = W2$mean_change,
		W2_min_chnage  = W2$min_change,
		W2_max_chnage  = W2$max_change,

		strong_sat = sat$strong_sat,
		soft_sat = sat$soft_sat,
		linear_region = sat$linear_region
	)
}

# Initializes the logging for epoch 0 using the full dataset.
# Computes the forward pass and MSE, then creates a log row.
init_log_row <- function(network, X, y) {
	fwd_full <- forward_pass(network, X)
	mse <- mean((fwd_full$output - y)^2)
	create_log_row(epoch = 0, mse = mse, network = network, fwd_full = fwd_full)
}

# Pretty-prints a log row to the console in a structured, human-readable format.
# Includes MSE, weight norms, rank, condition numbers, weight changes,
# accumulator statistics, and activation regimes.
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
		"\tNorm: %10.4f   Rank: %3d   Cond: %10.4e   Det: %10.4f\n\n",
		log_entry$A1_norm,
		log_entry$A1_rank,
		log_entry$A1_cond,
		log_entry$A1_det
	))

	# Accumulator A2
	cat("Accumulator A2 (hat)\n")
	cat(sprintf(
		"\tNorm: %10.4f   Rank: %3d   Cond: %10.4e   Det: %10.4f\n\n",
		log_entry$A2_norm,
		log_entry$A2_rank,
		log_entry$A2_cond,
		log_entry$A2_det
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

