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

