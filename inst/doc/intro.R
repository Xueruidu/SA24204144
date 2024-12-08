## -----------------------------------------------------------------------------
library(SA24204144)
data(data)
matrix <- cor(data, use = "pairwise.complete.obs")
matrix[is.na(matrix)] <- 0

## ----eval=FALSE---------------------------------------------------------------
# function(matrix){
#   if (nrow(matrix) != ncol(matrix)) {
#     stop("Error: The matrix is not square!")
#   }
#   if (!all(matrix == t(matrix), na.rm = TRUE)) {
#     stop("Error: The matrix is not symmetric!")
#   }
#   diag(matrix) <- 0
#   graph <- graph_from_adjacency_matrix(matrix, mode = "undirected", weighted = TRUE, diag = FALSE)
#   edge_weights <- E(graph)$weight
#   E(graph)$weight
#   edge_colors <- ifelse(edge_weights > 0, "firebrick1", "gray20")
#   suppressWarnings({
#     plot(graph,
#          vertex.size = 30,
#          vertex.label.cex = 1.0,
#          vertex.color = "skyblue",
#          vertex.frame.color = "blue",
#          edge.width = sqrt(abs(edge_weights) * 7),
#          edge.color = edge_colors,
#          main = "Network Graph without Self-Loops")
#   })
# }

## -----------------------------------------------------------------------------
graph_plot(matrix)

## ----eval=FALSE---------------------------------------------------------------
# function(matrix){
#   n <- nrow(matrix)
#   rho <- matrix[upper.tri(matrix)]
#   rho_positive <- sort(rho[which(rho > 0)],decreasing = TRUE)
#   rho_negative <- sort(rho[which(rho < 0)],decreasing = FALSE)
#   Phi_positive <- pnorm(rho_positive * sqrt(n),0,1)
#   Phi_negative <- pnorm(rho_negative * sqrt(n),0,1)
#   delta_positive <- c(diff(Phi_positive),0)
#   delta_negative <- c(diff(Phi_negative),0)
#   n_positive <- length(delta_positive)
#   min_error_positive <- Inf
#   break_point_positive <- 1
#   for (k in 1:(n_positive-1)) {
#     mean1 <- mean(delta_positive[1:k])
#     mean2 <- mean(delta_positive[(k+1):n_positive])
#     error1 <- sum((delta_positive[1:k] - mean1)^2)
#     error2 <- sum((delta_positive[(k+1):n_positive] - mean2)^2)
#     total_error <- error1 + error2
#     if (total_error < min_error_positive) {
#       min_error_positive <- total_error
#       break_point_positive <- k
#     }
#   }
# 
#   n_negative <- length(delta_negative)
#   min_error_negative <- Inf
#   break_point_negative <- 1
#   for (k in 1:(n_negative-1)) {
#     mean1 <- mean(delta_negative[1:k])
#     mean2 <- mean(delta_negative[(k+1):n_negative])
#     error1 <- sum((delta_negative[1:k] - mean1)^2)
#     error2 <- sum((delta_negative[(k+1):n_negative] - mean2)^2)
#     total_error <- error1 + error2
#     if (total_error < min_error_negative) {
#       min_error_negative <- total_error
#       break_point_negative <- k
#     }
#   }
#   result <- c(rho_positive[break_point_positive],rho_negative[break_point_negative])
#   names(result) <- c("break_point_positive","break_point_negative")
#   matrix_modified <- ifelse(matrix > result[1], 1, ifelse(matrix < result[2], -1, 0))
#   m1 <- break_point_positive
#   num1 <- 0
#   for (j in 2:(m1-1)) {
#     part1_1 <- delta_positive[j] + delta_positive[j-1] - mean( sort(delta_positive,decreasing = TRUE)[2:(m1-1)] + sort(delta_positive,decreasing = TRUE)[1:(m1-2)]  )
#     num1 <- num1 + part1_1^2
#   }
#   denom1 <- 0
#   for (k in 1:(m1-1)) {
#     part1_2 <- delta_positive[k] - mean( sort(delta_positive,decreasing = TRUE)[2:m1] )
#     denom1 <- denom1 + part1_2^2
#   }
#   SVR_positive <- ( (m1-1) / (2 * (m1 - 3))) * (num1 / denom1) - 1
# 
#   m2 <- break_point_negative
#   num2 <- 0
#   for (j in 2:(m2-1)) {
#     part2_1 <- delta_negative[j] + delta_negative[j-1] - mean( sort(delta_negative)[2:(m2-1)] + sort(delta_negative)[1:(m2-2)]  )
#     num2 <- num2 + part2_1^2
#   }
#   denom2 <- 0
#   for (k in 1:(m2-1)) {
#     part2_2 <- delta_negative[k] - mean( sort(delta_negative)[2:m2] )
#     denom2 <- denom2 + part2_2^2
#   }
#   SVR_negative <- ( (m2-1) / (2 * (m2 - 3))) * (num2 / denom2) - 1
# 
#   T <- m1 * SVR_positive^2 + m2 * SVR_negative^2
#   df <- 2
#   p_value <- 1 - pchisq(T, df)
#   SVR_test_result <- list(
#     statistic = T,
#     p_value = p_value,
#     df = df
#   )
#   return(list(break_points = result, simple_matrix = matrix_modified,SVR_test_result=SVR_test_result))
# }

## -----------------------------------------------------------------------------
SVR_result <- SVR_test(matrix)
SVR_result$break_points

## -----------------------------------------------------------------------------
SVR_result$simple_matrix
graph_plot(SVR_result$simple_matrix)

## -----------------------------------------------------------------------------
SVR_result$SVR_test_result

## ----eval=FALSE---------------------------------------------------------------
# function(matrix){
#   if (nrow(matrix) != ncol(matrix)) {
#     stop("Error: The matrix is not square!")
#   }
#   if (!all(matrix == t(matrix), na.rm = TRUE)) {
#     stop("Error: The matrix is not symmetric!")
#   }
#   n_nodes <- nrow(matrix)
#   matrix_upper <- matrix[upper.tri(matrix)]
#   positive_edges <- sum(matrix_upper > 0)
#   negative_edges <- sum(matrix_upper < 0)
#   type_1 <- 0
#   type_2 <- 0
#   type_3 <- 0
#   type_4 <- 0
#   for (i in 1:(n_nodes-2)) {
#     for (j in (i+1):(n_nodes-1)) {
#       for (k in (j+1):n_nodes) {
#         edge_ij <- matrix[i, j]
#         edge_jk <- matrix[j, k]
#         edge_ki <- matrix[k, i]
#         edges <- c(edge_ij, edge_jk, edge_ki)
#         if (edge_ij == 0 || edge_jk == 0 || edge_ki == 0) next
#         if (all(edges > 0)) {
#           type_1 <- type_1 + 1  # 所有边为正
#         } else if (sum(edges > 0) == 2) {
#           type_2 <- type_2 + 1  # 两边为正，一边为负
#         } else if (sum(edges > 0) == 1) {
#           type_3 <- type_3 + 1  # 一边为正，两边为负
#         } else {
#           type_4 <- type_4 + 1  # 所有边为负
#         }
#       }
#     }
#   }
#   total_triangles <- type_1 + type_2 + type_3 + type_4
#   w_1 <- type_1 / total_triangles
#   w_2 <- type_2 / total_triangles
#   w_3 <- type_3 / total_triangles
#   w_4 <- type_4 / total_triangles
#   w_balanced <- (w_1 + w_3) / (w_1 + w_2 + w_3 + w_4)
#   network_info <- list(
#     n_nodes = n_nodes,
#     positive_edges = positive_edges,
#     negative_edges = negative_edges
#   )
#   triangle_ratios <- list(
#     w_1 = w_1,
#     w_2 = w_2,
#     w_3 = w_3,
#     w_4 = w_4,
#     w_balanced = w_balanced
#   )
#   return(list(network_info = network_info, triangle_ratios = triangle_ratios))
# }

## -----------------------------------------------------------------------------
network_balance(matrix)
network_balance(SVR_result$simple_matrix)

## -----------------------------------------------------------------------------
print(network_balance(matrix)$triangle_ratios$w_balanced)
print(network_balance(SVR_result$simple_matrix)$triangle_ratios$w_balanced)

## ----eval=FALSE---------------------------------------------------------------
# NumericVector degree_centrality_cpp(NumericMatrix A) {
#    int n = A.nrow(); // Get the number of nodes (rows) in the adjacency matrix
#    NumericVector centrality(n); // Initialize a vector to store degree centrality of each node
# 
#    // Iterate over each node (row of the adjacency matrix)
#    for (int i = 0; i < n; i++) {
#      int degree = 0; // Initialize the degree of node i
# 
#      // Iterate over all other nodes to check for connections (edges)
#      for (int j = 0; j < n; j++) {
#        if (A(i, j) != 0) { // If there is an edge between node i and node j
#          degree += 1; // Increment the degree of node i
#        }
#      }
# 
#      centrality[i] = degree/2; // Store the degree centrality of node i
#    }
# 
#    return centrality; // Return the degree centrality vector
#  }

## ----eval=FALSE---------------------------------------------------------------
# NumericVector betweenness_centrality_cpp(NumericMatrix A) {
#   int n = A.nrow();
#   NumericVector centrality(n, 0.0); // Initialize centrality vector
# 
#   for (int s = 0; s < n; ++s) {
#     // Shortest path lengths and counts
#     std::vector<int> shortest_paths(n, -1); // -1 indicates infinite distance
#     std::vector<int> num_paths(n, 0);
#     std::vector<std::vector<int>> predecessors(n); // Predecessors for each node
#     std::queue<int> queue; // BFS queue
#     std::stack<int> stack; // Process order stack
# 
#     // Initialize for source node s
#     shortest_paths[s] = 0;
#     num_paths[s] = 1;
#     queue.push(s);
# 
#     // BFS to compute shortest paths and counts
#     while (!queue.empty()) {
#       int v = queue.front();
#       queue.pop();
#       stack.push(v);
# 
#       for (int w = 0; w < n; ++w) {
#         if (A(v, w) > 0) { // Check if there is an edge
#           if (shortest_paths[w] == -1) { // First visit to w
#             shortest_paths[w] = shortest_paths[v] + 1;
#             queue.push(w);
#           }
#           if (shortest_paths[w] == shortest_paths[v] + 1) { // Shortest path via v
#             num_paths[w] += num_paths[v];
#             predecessors[w].push_back(v);
#           }
#         }
#       }
#     }
# 
#     // Dependency calculation
#     std::vector<double> dependency(n, 0.0);
#     while (!stack.empty()) {
#       int w = stack.top();
#       stack.pop();
#       for (int v : predecessors[w]) {
#         dependency[v] += (static_cast<double>(num_paths[v]) / num_paths[w]) * (1.0 + dependency[w]);
#       }
#       if (w != s) {
#         centrality[w] += dependency[w];
#       }
#     }
#   }
# 
#   // Normalize centrality for undirected graph
#   for (int i = 0; i < n; ++i) {
#     centrality[i] /= 2.0;
#   }
# 
#   return centrality;
# }

## -----------------------------------------------------------------------------
degree_centrality <- degree_centrality_cpp(matrix)
names(degree_centrality) <- colnames(matrix)
sort(degree_centrality,decreasing = TRUE)

betweenness_centrality <- betweenness_centrality_cpp(matrix)
names(betweenness_centrality) <- colnames(matrix)
sort(betweenness_centrality,decreasing = TRUE)

## -----------------------------------------------------------------------------
degree_centrality <- degree_centrality_cpp(SVR_result$simple_matrix)
names(degree_centrality) <- colnames(SVR_result$simple_matrix)
sort(degree_centrality,decreasing = TRUE)

betweenness_centrality <- betweenness_centrality_cpp(SVR_result$simple_matrix)
names(betweenness_centrality) <- colnames(SVR_result$simple_matrix)
sort(betweenness_centrality,decreasing = TRUE)


