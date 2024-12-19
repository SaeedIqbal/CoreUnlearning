import numpy as np

class GraphUnlearning:
    def __init__(self, G, H, A, forget_set, subgraph, masks, alpha=0.5, beta=0.5, gamma=0.5, delta=0.5, lambda_subgraph=0.1):
        """
        Initialize the Graph Unlearning process.
        
        Parameters:
            G: tuple, (V, E), the graph with nodes (V) and edges (E)
            H: ndarray, node features, H_i^(l)
            A: ndarray, adjacency matrix
            forget_set: list, edges to forget, E_forget
            subgraph: tuple, subgraph G_k = (V_k, E_k)
            masks: tuple, (M_i, M_ij), node masks and edge masks
            alpha, beta, gamma, delta: float, weights for loss components
            lambda_subgraph: float, regularization parameter for subgraph loss
        """
        self.V, self.E = G  # Graph nodes and edges
        self.H = H  # Node features
        self.A = A  # Adjacency matrix
        self.forget_set = forget_set  # Forget set of edges
        self.subgraph = subgraph  # Subgraph G_k = (V_k, E_k)
        self.masks = masks  # Masks (M_i, M_ij)
        
        # Hyperparameters for loss function components
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_subgraph = lambda_subgraph

    def delete_edges(self):
        """
        Deletes edges from graph G based on the forget set E_forget.
        Creates the modified adjacency matrix A'.
        """
        A_prime = self.A.copy()  # Make a copy of the adjacency matrix
        for edge in self.forget_set:
            i, j = edge
            A_prime[i, j] = 0  # Remove edge (i, j)
            A_prime[j, i] = 0  # Because adjacency matrix is undirected
        return A_prime

    def update_node_features(self, A_prime):
        """
        Updates node features H_i^(l+1) using the provided adjacency matrix A_prime.
        
        Parameters:
            A_prime: ndarray, modified adjacency matrix after forgetting edges
        """
        H_new = np.zeros_like(self.H)
        for i in range(self.H.shape[0]):
            neighbors = np.nonzero(A_prime[i])[0]  # Get the neighbors of node i
            H_new[i] = np.dot(A_prime[i, neighbors], self.H[neighbors])  # Update node feature
        return H_new

    def compute_subgraph_loss(self):
        """
        Computes the subgraph loss.
        
        Returns:
            subgraph_loss: float, the computed subgraph loss
        """
        V_k, E_k = self.subgraph  # Subgraph nodes and edges
        L_subgraph = np.sum(np.linalg.norm(self.H[V_k] - self.H[V_k], axis=1)**2)  # , should calculate the correct loss
        for i, j in E_k:
            L_subgraph += np.linalg.norm(self.H[i] - self.H[j])**2
        return L_subgraph

    def compute_cross_partition_loss(self, H_old):
        """
        Computes the cross-partition loss to ensure uniformity across partitions.
        
        Parameters:
            H_old: ndarray, old node features before unlearning
        """
        return np.sum(np.linalg.norm(self.H - H_old, axis=1)**2)

    def compute_mask_loss(self):
        """
        Computes the mask-based loss function to selectively remove information.
        
        Returns:
            mask_loss: float, the computed mask-based loss
        """
        M_i, M_ij = self.masks
        L_mask = np.sum(np.linalg.norm(M_i * self.H, axis=1)**2)  # Apply mask on node features
        for i, j in self.E:
            L_mask += np.linalg.norm(M_ij[i, j] * self.A[i, j])**2  # Apply mask on edge features
        return L_mask

    def compute_latent_loss(self, H_latent, A_latent, H_original, A_original):
        """
        Computes the latent space optimization loss function.
        
        Parameters:
            H_latent: ndarray, latent node features
            A_latent: ndarray, latent adjacency matrix
            H_original: ndarray, original node features
            A_original: ndarray, original adjacency matrix
        """
        L_latent = np.sum(np.linalg.norm(H_latent - H_original, axis=1)**2)
        L_latent += self.gamma * np.sum(np.linalg.norm(A_latent - A_original, axis=1)**2)
        return L_latent

    def compute_total_loss(self, L_core, L_GNNDelete, L_subgraph, L_mask, L_latent):
        """
        Computes the total loss function by combining all the loss components.
        
        Parameters:
            L_core, L_GNNDelete, L_subgraph, L_mask, L_latent: float, individual loss components
        """
        L_total = L_core + self.alpha * L_GNNDelete + self.beta * L_subgraph + self.gamma * L_mask + self.delta * L_latent
        return L_total

    def graph_unlearning(self, H_old):
        """
        Perform the Graph Unlearning process and return the updated features, modified adjacency matrix, and total loss.
        
        Parameters:
            H_old: ndarray, original node features before the unlearning process
        """
        # Step 1: Delete edges and create modified adjacency matrix A'
        A_prime = self.delete_edges()
        
        # Step 2: Update node features using the new adjacency matrix
        H_new = self.update_node_features(A_prime)
        
        # Step 3: Compute subgraph loss
        L_subgraph = self.compute_subgraph_loss()
        
        # Step 4: Compute cross-partition loss
        L_cross_partition = self.compute_cross_partition_loss(H_old)
        
        # Step 5: Compute mask-based loss
        L_mask = self.compute_mask_loss()
        
        # Step 6: Compute latent loss (assuming H_latent, A_latent, H_original, A_original are available)
        L_latent = self.compute_latent_loss(H_new, A_prime, H_old, self.A)
        
        # Step 7: Combine all losses to compute total loss
        L_total = self.compute_total_loss(L_subgraph, L_cross_partition, L_mask, L_latent)
        
        # Return updated node features, modified adjacency matrix, and total loss
        return H_new, A_prime, L_total

#  usage:
# Assuming we have the following data:
# G = (V, E), H = node features, A = adjacency matrix, forget_set, subgraph, masks

# Initialize Graph Unlearning
graph_unlearning_obj = GraphUnlearning(G, H, A, forget_set, subgraph, masks)

# Perform Graph Unlearning
H_updated, A_prime, total_loss = graph_unlearning_obj.graph_unlearning(H)

# Output results
print("Updated Node Features:", H_updated)
print("Modified Adjacency Matrix:", A_prime)
print("Total Loss:", total_loss)
