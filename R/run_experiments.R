source("data_utils.R")
source("plot_utils.R")
source("network.R")

main <- function(
	num_points = 100,
	x_limits = c(-3, 3),
	coefficients = c(1, -2, 5, -1),
	noise_level = 2,
	hidden_size = 100,
	absorbed_bias = FALSE,
	epochs = 500,
	batch_size = 32,
	delta = NULL,
	shuffle_batches = TRUE,
	lr = 0.01,
	bla = TRUE,
	seed = 42
) {
	if (!is.null(seed)) set.seed(seed)
	cat("JEFF\n")

	# define the true polynomial
	polynomial <- function(x) polynomial_function(x, coefficients)

	# generate data
	data <- generate_random_polynomial_data(
		num_samples = num_points, x_range = x_limits, coeffs = coefficients,
		noise_sd = noise_level, seed = NULL
	)
	
	# initialize network
	network <- init_network(
		input_size = 1, hidden_size = hidden_size, output_size = 1,
		absorbed_bias = absorbed_bias
	)
	
	x_span <- x_limits[2] - x_limits[1]
	plot_xlim = c(
		x_limits[1] - 0.1 * x_span,
		x_limits[2] + 0.1 * x_span
	)
	x_plot <- seq(plot_xlim[1], plot_xlim[2], length.out = 500)
	y_true <- polynomial(x_plot)
	y_pad <- 0.1 * (max(y_true) - min(y_true))
	plot_ylim <- c(min(y_true) - y_pad, max(y_true) + y_pad)
	extra <- list(
		output_file = "result.gif",
		frame_dir = "frames",
		frame_width = 800,
		frame_height = 600,
		fps = 10,
		coefficients = coefficients,
		x_plot = x_plot,
		y_true = y_true,
		xlim = plot_xlim,
		ylim = plot_ylim
	)
	
	# train the network
	if (bla) {
		network <- train_network_bla(
			network, matrix(data$x, ncol = 1), matrix(data$y, ncol = 1),
			num_epochs = epochs, batch_size = batch_size, delta = delta,
			shuffle_batches = shuffle_batches, seed = NULL,
			extra = extra
		)
	} else {
		network <- train_network_gd(
			network, matrix(data$x, ncol = 1), matrix(data$y, ncol = 1),
			epochs = epochs, batch_size = batch_size,
			shuffle_batches = shuffle_batches, lr = lr, seed = NULL
		)
	}

	# plot results
	plot_results(
		network = network, data = data, polynomial_fn = polynomial,
		x_limits = plot_xlim
	)
}

# common parameters
num_points <- 6000
x_limits <- c(-3, 3)
noise_level <- 0
hidden_size <- 14
absorbed_bias <- TRUE
batch_size <- 256
delta <- 8
shuffle_batches <- TRUE
lr <- 0.01
bla <- TRUE
seed <- 42

#  experiments
experiments <- list(
	list(
		coefficients = c(1, -2, 5, -1),
		epochs       = 100,
		file         = "../images/cubic_1.gif"
	),
	list(
		coefficients = c(1, 0, -2, 0),
		epochs       = 35,
		file         = "../images/cubic_2.gif"
	),
	list(
		coefficients = c(2, 1, -2, 5, -1),
		epochs       = 35,
		file         = "../images/quartic_1.gif"
	),
	list(
		coefficients = c(4, 0, -20, 0, 0),
		epochs       = 35,
		file         = "../images/quartic_2.gif"
	)
)

# Run experiments
for (params in experiments) {
	main(
		num_points = num_points,
		x_limits = x_limits,
		coefficients = params$coefficients,
		noise_level = noise_level,
		hidden_size = hidden_size,
		absorbed_bias = absorbed_bias,
		epochs = params$epochs,
		batch_size = batch_size,
		delta = delta,
		shuffle_batches = shuffle_batches,
		lr = lr,
		bla = bla,
		seed = seed
	)

	file.rename("result.gif", params$file)
}

