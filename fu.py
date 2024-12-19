import numpy as np

# Loss function for forget operation (L_forget)
def compute_loss_forget(W):
    """
    Compute the forget loss, which penalizes the model for not forgetting the data properly.
    A common choice for this could be a regularization on the model weights.
    We use L2 regularization (Ridge) here as an .
    
    Args:
    - W (np.ndarray): The model weights
    
    Returns:
    - float: The forget loss value
    """
    forget_loss = np.sum(np.square(W))  # Sum of squared weights
    return forget_loss

# Loss function for retain operation (L_retain)
def compute_loss_retain(W):
    """
    Compute the retain loss, which penalizes the model for forgetting important information.
    This can be defined as a function of the model weights' magnitude or other task-specific logic.
    
    Args:
    - W (np.ndarray): The model weights
    
    Returns:
    - float: The retain loss value
    """
    retain_loss = np.sum(np.abs(W))  # Sum of absolute values of weights
    return retain_loss

# Gradient of the forget loss with respect to model weights (∇W L_forget)
def compute_gradients_loss_forget(W):
    """
    Compute the gradient of the forget loss with respect to the model weights.
    This is the derivative of L2 regularization: ∇W L_forget = 2 * W
    
    Args:
    - W (np.ndarray): The model weights
    
    Returns:
    - np.ndarray: The gradient of the forget loss with respect to W
    """
    grad_forget = 2 * W  # Gradient of L2 regularization (L_forget)
    return grad_forget

# Gradient of the retain loss with respect to model weights (∇W L_retain)
def compute_gradients_loss_retain(W):
    """
    Compute the gradient of the retain loss with respect to the model weights.
    This is the derivative of the sum of absolute values of the weights.
    
    Args:
    - W (np.ndarray): The model weights
    
    Returns:
    - np.ndarray: The gradient of the retain loss with respect to W
    """
    grad_retain = np.sign(W)  # Gradient of the L1 regularization (retain) term
    return grad_retain

# Regularization term for the model weights (L2 regularization)
def compute_regularization(W):
    """
    Compute the regularization term for the model weights, which helps prevent overfitting.
    We use L2 regularization (Ridge) in this case.
    
    Args:
    - W (np.ndarray): The model weights
    
    Returns:
    - np.ndarray: The regularization term
    """
    regularization = 0.1 * W  # Regularization term, lambda * W (lambda = 0.1 for )
    return regularization

# Linear transformation (F(W) = A W + b)
def compute_linear_transformation(W, A, b):
    """
    Compute the linear transformation F(W) = A * W + b
    
    Args:
    - W (np.ndarray): The model weights
    - A (np.ndarray): The matrix A
    - b (np.ndarray): The vector b
    
    Returns:
    - np.ndarray: The result of the linear transformation F(W)
    """
    return np.dot(A, W) + b

# Federated Unlearning Methodology Implementation
def federated_unlearning(W_old, eta, W_aux, alpha, beta, gamma, delta, I_W, lambda_dynamic, A, b, N_i, N_total):
    """
    Update the model weights based on Federated Unlearning methodology.

    Args:
    - W_old (np.ndarray): The old model weights
    - eta (float): The learning rate
    - W_aux (np.ndarray): Auxiliary weights
    - alpha (float): Scaling factor for F(W)
    - beta (float): Scaling factor for forget gradients
    - gamma (float): Regularization parameter
    - delta (float): Damping factor for learning rate adjustment
    - I_W (np.ndarray): Indicator function (binary mask for pruning)
    - lambda_dynamic (float): Dynamic scaling factor for the retain loss
    - A (np.ndarray): Matrix for linear transformation
    - b (np.ndarray): Bias term for linear transformation
    - N_i (int): Local dataset size for each model
    - N_total (int): Total dataset size across all models
    
    Returns:
    - np.ndarray: The updated global model weights W_new
    """
    # Step 1: Compute the linear transformation F(W)
    F_W = compute_linear_transformation(W_old, A, b)
    
    # Step 2: Compute the total loss function
    L_forget = compute_loss_forget(W_old)
    L_retain = compute_loss_retain(W_old)
    L_total = L_forget + lambda_dynamic * L_retain  # Total loss function
    
    # Step 3: Compute the gradients
    grad_L_forget = compute_gradients_loss_forget(W_old)
    grad_L_retain = compute_gradients_loss_retain(W_old)
    
    # Step 4: Update the model weights based on the gradient
    W_new = W_old - eta * (grad_L_forget + alpha * F_W + beta * grad_L_forget)
    
    # Step 5: Gradient ascent with retention
    W_new = W_old - eta * (grad_L_forget + alpha * grad_L_retain)
    
    # Step 6: Adjust the learning rate with the damping factor
    eta_new = eta * (1 - delta * np.linalg.norm(grad_L_forget))
    
    # Step 7: Ensure convergence conditions
    if np.sum(eta_new * np.linalg.norm(grad_L_forget)) > 1:
        eta_new = eta / 2
    if alpha * np.linalg.norm(grad_L_retain) > eta:
        alpha = eta / np.linalg.norm(grad_L_retain)
    
    # Step 8: Model pruning update rule
    grad_regularize = compute_regularization(W_old)
    QW_new = W_old - eta * (grad_L_forget * I_W + gamma * grad_regularize)
    
    # Step 9: Compute the pruning loss
    L_prune = np.sum(I_W * np.square(W_old)) + lambda_dynamic * np.sum(np.linalg.norm(grad_L_forget))
    
    # Step 10: Federated unlearning update
    W_global_new = (N_i / N_total) * W_new
    
    # Return the updated global model weights
    return W_global_new



# Federated Unlearning Client class
class FederatedUnlearningClient:
    def __init__(self, model_weights, learning_rate, alpha, beta, gamma, delta):
        self.model_weights = model_weights
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def update_model(self, A, b, lambda_dynamic, delta_t):
        # Compute the linear transformation F(W) = A * W + b
        F_W = linear_transformation(self.model_weights, A, b)

        # Compute the total loss: L_total(W) = L_forget(W) + λ * L_retain(W)
        L_forget = compute_loss_forget(self.model_weights)
        L_retain = compute_loss_retain(self.model_weights)
        L_total = L_forget + lambda_dynamic * L_retain
        
        # Compute the gradients
        grad_L_total = compute_gradients_loss_forget(self.model_weights)
        grad_L_retain = compute_gradients_loss_retain(self.model_weights)

        # Update model weights based on the gradient
        new_weights = self.model_weights - self.learning_rate * (grad_L_total + self.alpha * F_W + self.beta * grad_L_forget)
        
        # Gradient ascent with retention
        new_weights = self.model_weights - self.learning_rate * (grad_L_forget + self.alpha * grad_L_retain)

        # Adjust learning rate with the damping factor
        learning_rate_new = self.learning_rate * (1 - self.delta * np.linalg.norm(grad_L_forget))

        # Pruning update rule
        I_W = np.ones_like(self.model_weights)  # Assuming indicator function I(W) is 1 for simplicity
         ggrad_regularize = compute_regularization(self.model_weights)
        new_weights = self.model_weights - self.learning_rate * (grad_L_forget * I_W + self.gamma * grad_regularize)

        return new_weights, learning_rate_new

# Federated Unlearning Server class
class FederatedUnlearningServer:
    def __init__(self, clients, global_weights, num_rounds):
        self.clients = clients
        self.global_weights = global_weights
        self.num_rounds = num_rounds

    def aggregate_weights(self, client_weights):
        # Federated unlearning update: W_global_new = Σ Ni/N_total * Wi_new
        N_total = sum([client.model_weights.shape[0] for client in self.clients])
        aggregated_weights = np.zeros_like(self.global_weights)
        for i, client in enumerate(self.clients):
            Ni = client.model_weights.shape[0]
            aggregated_weights += (Ni / N_total) * client.model_weights
        return aggregated_weights

    def federated_unlearning(self, A, b, lambda_dynamic, delta_t):
        # Perform federated unlearning for multiple rounds
        for round_num in range(self.num_rounds):
            client_weights = []
            for client in self.clients:
                new_weights, _ = client.update_model(A, b, lambda_dynamic, delta_t)
                client_weights.append(new_weights)
            
            # Aggregate the weights from all clients
            self.global_weights = self.aggregate_weights(client_weights)
            {print(f"Round {round_num+1}: Global weights updated.")
        return self.global_weights

# Main Federated Unlearning framework
class FederatedUnlearning:
    def __init__(self, clients, initial_weights, learning_rate, alpha, beta, gamma, delta, num_rounds):
        self.clients = clients
        self.server = FederatedUnlearningServer(clients, initial_weights, num_rounds)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.num_rounds = num_rounds

    def run(self, A, b, lambda_dynamic, delta_t):
        # Run federated unlearning across the server and clients
        final_weights = self.server.federated_unlearning(A, b, lambda_dynamic, delta_t)
        return final_weights

#  usage
if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.01
    alpha = 0.5
    beta = 0.5
    gamma = 0.1
    delta = 0.05
    num_rounds = 10
    initial_weights = np.random.randn(10)  #  initial model weights
    A = np.random.randn(10, 10)  #  matrix A for linear transformation
    b = np.random.randn(10)  #  bias vector b
    lambda_dynamic = 1  #  dynamic lambda value
    delta_t = 0.1  #  time delta for lambda dynamic adjustment

#===============================FOR CODE TESTING===========================================================================
    # Initialize clients with random weights
    clients = [FederatedUnlearningClient(np.random.randn(10), learning_rate, alpha, beta, gamma, delta) for _ in range(5)]

    # Initialize and run federated unlearning
    federated_unlearning = FederatedUnlearning(clients, initial_weights, learning_rate, alpha, beta, gamma, delta, num_rounds)
    final_weights = federated_unlearning.run(A, b, lambda_dynamic, delta_t)

    print(f"Final global model weights after federated unlearning: {final_weights}")
