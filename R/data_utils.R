# =====================================================
# Training Data Generation
# =====================================================

# Evaluates a polynomial function for given input values.
polynomial_function <- function(x, coeffs = c(1, 0, 0)) {
	degree <- length(coeffs) - 1
	y <- rep(0, length(x))
	for (i in 0:degree) y <- y + coeffs[i + 1] * x^(degree - i)
	return(y)
}

# Generates random data from a polynomial function with optional Gaussian noise.
generate_random_polynomial_data <- function(
	num_samples = 100, coeffs = c(1, 0, 0), x_range = c(-3, 3), noise_sd = 0,
	seed = NULL
) {
	if (!is.null(seed)) set.seed(seed)
	x <- runif(num_samples, min = x_range[1], max = x_range[2])
	y <- polynomial_function(x, coeffs)
	if (noise_sd > 0) y <- y + rnorm(num_samples, mean = 0, sd = noise_sd)
	result <- data.frame(x = x, y = y)
	return(result)
}

# Plots a true polynomial function and a neural network's predictions.
plot_results <- function(network, data, polynomial_fn, x_limits) {
	library(ggplot2)
	x_plot <- seq(
		from = x_limits[1],
		to   = x_limits[2],
		length.out = 200
	)
	
	y_true <- polynomial_fn(x_plot)
	X_plot <- matrix(x_plot, ncol = 1)
	y_pred <- forward_pass(network, X_plot)$output

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
		labs(x = "x", y = "y", title = "Polynomial Fit vs Network Prediction") +
		theme_minimal() +
		theme(plot.title = element_text(hjust = 0.5))
	
	print(p)
}

