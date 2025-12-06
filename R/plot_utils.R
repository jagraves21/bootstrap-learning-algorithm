# ========================================
# Ploting Utilities
# ========================================

# Generate a human-readable string representing a polynomial from its
# coefficients.
poly_to_string <- function(coefficients) {
	deg <- length(coefficients) - 1
	terms <- character(0)
	for (i in seq_along(coefficients)) {
		coef <- coefficients[i]
		if (abs(coef) < 1e-12) next

		power <- deg - (i - 1)
		abs_coef <- abs(coef)

		if (abs_coef == 1 && power != 0) {
			coef_str <- ""
		} else {
			coef_str <- formatC(abs_coef, format = "f", digits = 3)
		}

		if (power == 0) {
			term_body <- coef_str
		} else if (power == 1) {
			term_body <- paste0(coef_str, "x")
		} else {
			term_body <- paste0(coef_str, "x^", power)
		}

		if (coef < 0) {
			sign_str <- " - "
		} else {
			sign_str <- if (length(terms) > 0) " + " else ""
		}

		terms <- c(terms, paste0(sign_str, term_body))
	}

	if (length(terms) == 0) {
		return("f(x) = 0")
	}

	return(paste0("f(x) = ", paste(terms, collapse = "")))
}

# Plots a true polynomial function and a neural network's predictions.
plot_results <- function(
	network, data, polynomial_fn, x_limits, coefficients = NULL
) {
	library(ggplot2)
	x_plot <- seq(
		from = x_limits[1],
		to   = x_limits[2],
		length.out = 200
	)
	
	y_true <- polynomial_fn(x_plot)
	X_plot <- matrix(x_plot, ncol = 1)
	y_pred <- forward_pass(network, X_plot)$output

	plot_title <- if (!is.null(coefficients)) {
		poly_to_string(coefficients)
	} else {
		"Polynomial Fit vs Network Prediction"
	}

	p <- ggplot() +
		geom_point(
			data = data, aes(x = x, y = y),
			color = "blue", size = 1.5, alpha = 0.25
		) +
		geom_line(
			data = data.frame(x = x_plot, y = y_true),
			aes(x = x, y = y), color = "blue", linewidth = 0.8
		) +
		geom_line(
			data = data.frame(x = x_plot, y = y_pred),
			aes(x = x, y = y), color = "red", linewidth = 0.8
		) +
		labs(x = "x", y = "y", title = plot_title) +
		theme_minimal() +
		theme(plot.title = element_text(hjust = 0.5, family = "mono"))
	
	print(p)
}



# ========================================
# Training Log Plotting Utilities
# ========================================

# Plot MSE over epochs
save_mse_plot <- function(
	training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	y_min <- min(training_log$mse, na.rm = TRUE)
	y_max <- max(training_log$mse, na.rm = TRUE)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$mse, ylim = ylim,
		type = "l", lwd = 2,
		xlab = "Epoch", ylab = "MSE",
		main = "Training MSE per Epoch",
		family = "mono"
	)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")
}

# Plot weight norms
save_weight_norm_plot <- function(
  training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	y_min <- min(training_log$W1_norm, training_log$W2_norm, na.rm = TRUE)
	y_max <- max(training_log$W1_norm, training_log$W2_norm, na.rm = TRUE)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$W1_norm, ylim = ylim,
		type = "l", col = "blue", lwd = 2,
		xlab = "Epoch", ylab = "Frobenius Norm",
		main = "Weight Norms per Epoch",
		family = "mono"
	)
	lines(training_log$epoch, training_log$W2_norm, col = "red", lwd = 2)
	legend("topright", legend = c("W1", "W2"), col = c("blue", "red"), lwd = 2)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")
}

# Plot weight min/max changes with ribbons around mean
save_weight_minmax_plot <- function(
	training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)
	
	y_min <- min(
		training_log$W1_min_chnage, training_log$W2_min_chnage, na.rm = TRUE
	)
	y_max <- max(
		training_log$W1_max_chnage, training_log$W2_max_chnage, na.rm = TRUE
	)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$W1_mean_chnage, ylim = ylim,
		type = "n", xlab = "Epoch", ylab = "Weight Change",
		main = "Weight Min/Max Changes per Epoch",
		family = "mono"
	)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")

	# W1 ribbon
	polygon(
		c(training_log$epoch, rev(training_log$epoch)),
		c(training_log$W1_min_chnage, rev(training_log$W1_max_chnage)),
		col = rgb(0, 0, 1, 0.2), border = NA
	)
	lines(training_log$epoch, training_log$W1_mean_chnage, col = "blue", lwd = 2)

	# W2 ribbon
	polygon(
		c(training_log$epoch, rev(training_log$epoch)),
		c(training_log$W2_min_chnage, rev(training_log$W2_max_chnage)),
		col = rgb(1, 0, 0, 0.2), border = NA
	)
	lines(training_log$epoch, training_log$W2_mean_chnage, col = "red", lwd = 2)

	legend(
		"topright",
		legend = c("W1 mean ± range", "W2 mean ± range"),
		col = c("blue", "red"), lwd = 2, bty = "n"
	)
}

# Plot weight ranks
save_weight_rank_plot <- function(
	training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	y_min <- min(training_log$W1_rank, training_log$W2_rank, na.rm = TRUE)
	y_max <- max(training_log$W1_rank, training_log$W2_rank, na.rm = TRUE)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$W1_rank, ylim = ylim,
		type = "l", col = "blue", lwd = 2,
		xlab = "Epoch", ylab = "Rank",
		main = "Weight Ranks per Epoch",
		family = "mono"
	)
	lines(training_log$epoch, training_log$W2_rank, col = "red", lwd = 2)
	legend("topright", legend = c("W1", "W2"), col = c("blue", "red"), lwd = 2)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")
}

# Plot weight condition numbers
save_weight_condition_plot <- function(
	training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	y_min <- min(training_log$W1_cond, training_log$W2_cond, na.rm = TRUE)
	y_max <- max(training_log$W1_cond, training_log$W2_cond, na.rm = TRUE)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$W1_cond, ylim = ylim,
		type = "l", col = "blue", lwd = 2,
		xlab = "Epoch", ylab = "Condition Number",
		main = "Weight Condition Numbers per Epoch",
		family = "mono"
	)
	lines(training_log$epoch, training_log$W2_cond, col = "red", lwd = 2)
	legend("topright", legend = c("W1", "W2"), col = c("blue", "red"), lwd = 2)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")
}

# Plot accumulator norms
save_acc_norm_plot <- function(
	training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	y_min <- min(training_log$A1_norm, training_log$A2_norm, na.rm = TRUE)
	y_max <- max(training_log$A1_norm, training_log$A2_norm, na.rm = TRUE)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$A1_norm, ylim = ylim,
		type = "l", col = "blue", lwd = 2,
		xlab = "Epoch", ylab = "Frobenius Norm",
		main = "Accumulator Norms per Epoch",
		family = "mono"
	)
	lines(training_log$epoch, training_log$A2_norm, col = "red", lwd = 2)
	legend(
		"topright", legend = c("A1_hat", "A2_hat"), col = c("blue", "red"),
		lwd = 2
	)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")
}

# Plot accumulator ranks
save_acc_rank_plot <- function(
	training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	y_min <- min(training_log$A1_rank, training_log$A2_rank, na.rm = TRUE)
	y_max <- max(training_log$A1_rank, training_log$A2_rank, na.rm = TRUE)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$A1_rank, ylim = ylim,
		type = "l", col = "blue", lwd = 2,
		xlab = "Epoch", ylab = "Rank",
		main = "Accumulator Ranks per Epoch",
		family = "mono"
	)
	lines(training_log$epoch, training_log$A2_rank, col = "red", lwd = 2)
	legend(
		"topright", legend = c("A1_hat", "A2_hat"), col = c("blue", "red"),
		lwd = 2
	)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")
}

# Plot accumulator condition numbers
save_acc_condition_plot <- function(
	training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	y_min <- min(training_log$A1_cond, training_log$A2_cond, na.rm = TRUE)
	y_max <- max(training_log$A1_cond, training_log$A2_cond, na.rm = TRUE)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$A1_cond, ylim = ylim,
		type = "l", col = "blue", lwd = 2,
		xlab = "Epoch", ylab = "Condition Number",
		main = "Accumulator Condition Numbers per Epoch",
		family = "mono"
	)
	lines(training_log$epoch, training_log$A2_cond, col = "red", lwd = 2)
	legend(
		"topright", legend = c("A1_hat", "A2_hat"), col = c("blue", "red"),
		lwd = 2
	)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")
}

# Plot activation regimes
save_activation_plot <- function(
	training_log, filename, width = 800, height = 600, vline_epoch = NULL
) {
	png(filename, width = width, height = height)
	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	y_min <- min(
		training_log$strong_sat, training_log$soft_sat,
		training_log$linear_region, na.rm = TRUE
	)
	y_max <- max(
		training_log$strong_sat, training_log$soft_sat,
		training_log$linear_region, na.rm = TRUE
	)
	y_pad <- (y_max - y_min) * 0.05
	ylim <- c(y_min - y_pad, y_max + y_pad)

	plot(
		training_log$epoch, training_log$strong_sat, ylim = ylim,
		type = "l", col = "red", lwd = 2,
		xlab = "Epoch", ylab = "Fraction",
		main = "Activation Regimes per Epoch",
		family = "mono"
	)
	lines(training_log$epoch, training_log$soft_sat, col = "orange", lwd = 2)
	lines(training_log$epoch, training_log$linear_region, col = "blue", lwd = 2)
	legend(
		"topright",
		legend = c("Strong Saturation", "Soft Saturation", "Linear Region"),
		col = c("red", "orange", "blue"), lwd = 2
	)
	grid(col = "gray80")
	if (!is.null(vline_epoch)) abline(v = vline_epoch, lty = 2, col = "black")
}

# Plot training logs
plot_logs <- function(
	training_log, dir, width = 800, height = 600, vline_epoch = NULL
) {
	if (!dir.exists(dir)) dir.create(dir, showWarnings = FALSE)

	save_mse_plot(
		training_log, file.path(dir, "mse_plot.png"),
		width, height, vline_epoch = vline_epoch
	)
	save_weight_norm_plot(
		training_log, file.path(dir, "weight_norm_plot.png"),
		width, height, vline_epoch = vline_epoch
	)
	save_weight_minmax_plot(
		training_log, file.path(dir, "weight_minmax_plot.png"),
		width, height, vline_epoch = vline_epoch
	)
	save_weight_rank_plot(
		training_log, file.path(dir, "weight_rank_plot.png"),
		width, height, vline_epoch = vline_epoch
	)
	save_weight_condition_plot(
		training_log, file.path(dir, "weight_condition_plot.png"),
		width, height, vline_epoch = vline_epoch
	)
	save_acc_norm_plot(
		training_log, file.path(dir, "acc_norm_plot.png"),
		width, height, vline_epoch = vline_epoch
	)
	save_acc_rank_plot(
		training_log, file.path(dir, "acc_rank_plot.png"),
		width, height, vline_epoch = vline_epoch
	)
	save_acc_condition_plot(
		training_log, file.path(dir, "acc_condition_plot.png"),
		width, height, vline_epoch = vline_epoch
	)
	save_activation_plot(
		training_log, file.path(dir, "activation_plot.png"),
		width, height, vline_epoch = vline_epoch
	)

	write.csv(
		training_log,
		file = file.path(dir, "training_log.csv"),
		row.names = FALSE
	)
}



# ========================================
# Animated GIF Utilities
# ========================================

# Initialize the frames directory safely.
init_frame_dir <- function(frame_dir = "frames") {
	if (frame_dir %in% c("/", "C:/", "C:\\", "", "~")) {
		stop("Refusing to initialize unsafe folder: ", frame_dir)
	}

	if (!dir.exists(frame_dir)) {
		dir.create(frame_dir, showWarnings = FALSE)
		message(sprintf("Directory '%s' created.", frame_dir))
	} else {
		files <- list.files(frame_dir, pattern = "\\.png$", full.names = TRUE)
		if (length(files) > 0) {
			file.remove(files)
		}
		message(
			sprintf("Directory '%s' cleared of existing PNG files.", frame_dir)
		)
	}
}

# Save a single training frame as a PNG image.
save_training_frame <- function(
	epoch, frame_dir, frame_width, frame_height, coefficients, x_plot, y_true,
	y_pred, xlim, ylim
) {
	frame_file <- file.path(frame_dir, sprintf("frame_%04d.png", epoch))
	png(frame_file, width = frame_width, height = frame_height)

	old_par <- par(no.readonly = TRUE)
	on.exit({ par(old_par); dev.off() }, add = TRUE)

	poly_str <- poly_to_string(coefficients)

	plot(
		x_plot, y_true,
		type = "l", lwd = 2,
		xlim = xlim, ylim = ylim,
		xlab = "x", ylab = "y",
		main = ""
	)

	lines(x_plot, y_pred, col = "red", lwd = 2)

	title(main = poly_str, family = "mono")
	
	mse <- mean((y_true - y_pred)^2)
	epoch_str <- sprintf("%4d", epoch)
	subtitle_text <- sprintf("Epoch %s   |   MSE = %12.6f", epoch_str, mse)
	mtext(subtitle_text, side = 3, line = 0.5, family = "mono", cex = 0.9)
}

# Convert saved PNG frames into an animated GIF.
create_gif_from_frames <- function(
	frame_dir, output_file, fps
) {
	library(magick)

	frame_files <- list.files(
		frame_dir, pattern = "frame_\\d+\\.png", full.names = TRUE
	)
	if (length(frame_files) == 0) {
		stop("No PNG frames found in the specified frame_dir")
	}

	frame_files <- sort(frame_files)
	frames <- image_read(frame_files)
	animation <- image_animate(frames, fps = fps)
	image_write(animation, path = output_file)

	message(sprintf("Animated GIF saved to '%s'.", output_file))
}

# Remove the entire frames directory and all its contents safely.
remove_frames_directory <- function(frame_dir = "frames") {
	if (!dir.exists(frame_dir)) {
		message(sprintf(
			"Directory '%s' does not exist. Nothing to remove.", frame_dir
		))
		return()
	}

	if (frame_dir %in% c("/", "C:/", "C:\\", "", "~")) {
		stop("Refusing to delete unsafe folder: ", frame_dir)
	}

	unlink(frame_dir, recursive = TRUE)
	message(sprintf(
		"Directory '%s' and all its contents have been removed.", frame_dir
	))
}

