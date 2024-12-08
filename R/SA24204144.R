#' @import microbenchmark
#' @import igraph
#' @importFrom Rcpp evalCpp
#' @importFrom stats rnorm rgamma pchisq pnorm
#' @useDynLib SA24204144
NULL

#' @title Dataset
#' @name data
#' @description A dataset containing various example variables for analysis.
#' \describe{
#'   \item{id}{Unique identifier for each observation.}
#'   \item{months}{Time in months since the start of the study.}
#'   \item{age}{Age of the individual at baseline (years).}
#'   \item{cd4}{CD4 cell count (cells/mm³).}
#'   \item{cd8}{CD8 cell count (cells/mm³).}
#'   \item{vload0}{Baseline viral load (copies/mL). \code{NA} indicates missing values.}
#'   \item{aidscase}{Indicator variable for AIDS diagnosis. \code{NA} indicates missing values.}
#'   \item{vtime}{Time to viral suppression (months).}
#'   \item{sctime}{Time to seroconversion (months).}
#'   \item{atime}{Time to AIDS diagnosis (months). \code{NA} indicates missing values.}
#'   \item{dtime}{Time to death (months).}
#'   \item{ideath}{Indicator for death event (0 = no, 1 = yes).}
#' }
#' @examples
#' \dontrun{
#' data(data)
#' summary(data)
#' }
NULL


#' @title Plot Network Graph without Self-Loops
#' @description This function plots a network graph from an adjacency matrix, 
#'              ensuring that the matrix is square, symmetric, and free of self-loops.
#'              It uses edge weights to color the edges and adjusts their thickness.
#' @param matrix A symmetric weighted adjacency matrix with no self-loops.
#' @return A network plot showing the relationships between nodes.
#' @examples
#' \dontrun{
#'     # Example matrix
#'     adj_matrix <- matrix(c(0, 1, -1, 1, 0, 0, -1, 0, 0), nrow = 3)
#'     graph_plot(adj_matrix)
#' }
#' @export
graph_plot <- function(matrix){
  if (nrow(matrix) != ncol(matrix)) {
    stop("Error: The matrix is not square!")
  }
  if (!all(matrix == t(matrix), na.rm = TRUE)) {
    stop("Error: The matrix is not symmetric!")
  }
  diag(matrix) <- 0
  graph <- graph_from_adjacency_matrix(matrix, mode = "undirected", weighted = TRUE, diag = FALSE)
  edge_weights <- E(graph)$weight
  E(graph)$weight
  edge_colors <- ifelse(edge_weights > 0, "firebrick1", "gray20")
  suppressWarnings({
    plot(graph, 
         vertex.size = 30, 
         vertex.label.cex = 1.0,
         vertex.color = "skyblue", 
         vertex.frame.color = "blue", 
         edge.width = sqrt(abs(edge_weights) * 7), 
         edge.color = edge_colors, 
         main = "Network Graph without Self-Loops")
  })
}
#' @title Measure the Balance of a Network
#' @description This function calculates the balance of a network based on the number of 
#'              positive and negative edges in the adjacency matrix. It also computes 
#'              the distribution of triangle types in the network.
#' @param matrix A symmetric weighted adjacency matrix.
#' @return A list containing:
#'         - `network_info`: The number of nodes, positive edges, and negative edges.
#'         - `triangle_ratios`: The proportion of each triangle type and overall network balance.
#' @examples
#' \dontrun{
#'     adj_matrix <- matrix(c(0, 1, -1, 1, 0, 0, -1, 0, 0), nrow = 3)
#'     result <- network_balance(adj_matrix)
#'     print(result)
#' }
#' @export
network_balance <- function(matrix){
  if (nrow(matrix) != ncol(matrix)) {
    stop("Error: The matrix is not square!")
  }
  if (!all(matrix == t(matrix), na.rm = TRUE)) {
    stop("Error: The matrix is not symmetric!")
  }
  n_nodes <- nrow(matrix)
  matrix_upper <- matrix[upper.tri(matrix)]
  positive_edges <- sum(matrix_upper > 0)
  negative_edges <- sum(matrix_upper < 0)
  type_1 <- 0
  type_2 <- 0
  type_3 <- 0
  type_4 <- 0
  for (i in 1:(n_nodes-2)) {
    for (j in (i+1):(n_nodes-1)) {
      for (k in (j+1):n_nodes) {
        edge_ij <- matrix[i, j]
        edge_jk <- matrix[j, k]
        edge_ki <- matrix[k, i]
        edges <- c(edge_ij, edge_jk, edge_ki)
        if (edge_ij == 0 || edge_jk == 0 || edge_ki == 0) next
        if (all(edges > 0)) {
          type_1 <- type_1 + 1  # 所有边为正
        } else if (sum(edges > 0) == 2) {
          type_2 <- type_2 + 1  # 两边为正，一边为负
        } else if (sum(edges > 0) == 1) {
          type_3 <- type_3 + 1  # 一边为正，两边为负
        } else {
          type_4 <- type_4 + 1  # 所有边为负
        }
      }
    }
  }
  total_triangles <- type_1 + type_2 + type_3 + type_4
  w_1 <- type_1 / total_triangles
  w_2 <- type_2 / total_triangles
  w_3 <- type_3 / total_triangles
  w_4 <- type_4 / total_triangles
  w_balanced <- (w_1 + w_3) / (w_1 + w_2 + w_3 + w_4)
  network_info <- list(
    n_nodes = n_nodes,
    positive_edges = positive_edges,
    negative_edges = negative_edges
  )
  triangle_ratios <- list(
    w_1 = w_1,
    w_2 = w_2,
    w_3 = w_3,
    w_4 = w_4,
    w_balanced = w_balanced
  )
  return(list(network_info = network_info, triangle_ratios = triangle_ratios))
}
#' @title SVR Test for Network Correlation
#' @description This function performs an SVR (Squared Variance Ratio) test to analyze
#'              the breakpoints of positive and negative correlations in a network's
#'              weighted adjacency matrix. The test computes the SVR value and p-value
#'              based on a chi-square distribution, which allows us to test the null 
#'              hypothesis that the m1 smallest positive and m2 largest negative 
#'              correlations are zero.
#'
#'              The SVR test helps to identify significant positive and negative 
#'              correlations in the network by breaking them into smaller and larger 
#'              segments and calculating the squared variance ratio for each segment.
#'              The breakpoints, along with the SVR statistic and p-value, are returned
#'              to provide insights into the strength and significance of the correlations.
#'
#' @param matrix A symmetric weighted adjacency matrix representing the network. 
#'               The matrix elements should be numeric values where each element 
#'               represents the strength of the correlation between two nodes in the network.
#'               The matrix must be square and symmetric (i.e., the correlation between node i and j 
#'               is the same as between node j and i).
#' 
#' @return A list containing:
#'         - `break_points`: A named vector with the breakpoints for positive and negative correlations.
#'         - `simple_matrix`: A transformed matrix with values set to 1, 0, or -1 based on thresholds.
#'         - `SVR_test_result`: A list with the following elements:
#'             - `statistic`: The calculated SVR test statistic.
#'             - `p_value`: The p-value for the chi-square test.
#'             - `df`: The degrees of freedom for the chi-square test.
#'
#' @examples
#' \dontrun{
#'     adj_matrix <- matrix(c(0, 1, -1, 1, 0, 0, -1, 0, 0), nrow = 3)
#'     result <- SVR_test(adj_matrix)
#'     print(result)
#' }
#' @export

SVR_test <- function(matrix){
  n <- nrow(matrix)
  rho <- matrix[upper.tri(matrix)]
  rho_positive <- sort(rho[which(rho > 0)],decreasing = TRUE)
  rho_negative <- sort(rho[which(rho < 0)],decreasing = FALSE)
  Phi_positive <- pnorm(rho_positive * sqrt(n),0,1)
  Phi_negative <- pnorm(rho_negative * sqrt(n),0,1)
  delta_positive <- c(diff(Phi_positive),0)
  delta_negative <- c(diff(Phi_negative),0)
  n_positive <- length(delta_positive)
  min_error_positive <- Inf
  break_point_positive <- 1
  for (k in 1:(n_positive-1)) {
    mean1 <- mean(delta_positive[1:k])
    mean2 <- mean(delta_positive[(k+1):n_positive])
    error1 <- sum((delta_positive[1:k] - mean1)^2)
    error2 <- sum((delta_positive[(k+1):n_positive] - mean2)^2)
    total_error <- error1 + error2
    if (total_error < min_error_positive) {
      min_error_positive <- total_error
      break_point_positive <- k
    }
  }
  
  n_negative <- length(delta_negative)
  min_error_negative <- Inf
  break_point_negative <- 1
  for (k in 1:(n_negative-1)) {
    mean1 <- mean(delta_negative[1:k])
    mean2 <- mean(delta_negative[(k+1):n_negative])
    error1 <- sum((delta_negative[1:k] - mean1)^2)
    error2 <- sum((delta_negative[(k+1):n_negative] - mean2)^2)
    total_error <- error1 + error2
    if (total_error < min_error_negative) {
      min_error_negative <- total_error
      break_point_negative <- k
    }
  }
  result <- c(rho_positive[break_point_positive],rho_negative[break_point_negative])
  names(result) <- c("break_point_positive","break_point_negative")
  matrix_modified <- ifelse(matrix > result[1], 1, ifelse(matrix < result[2], -1, 0))
  m1 <- break_point_positive
  num1 <- 0
  for (j in 2:(m1-1)) {
    part1_1 <- delta_positive[j] + delta_positive[j-1] - mean( sort(delta_positive,decreasing = TRUE)[2:(m1-1)] + sort(delta_positive,decreasing = TRUE)[1:(m1-2)]  )
    num1 <- num1 + part1_1^2
  }
  denom1 <- 0
  for (k in 1:(m1-1)) {
    part1_2 <- delta_positive[k] - mean( sort(delta_positive,decreasing = TRUE)[2:m1] )
    denom1 <- denom1 + part1_2^2
  }
  SVR_positive <- ( (m1-1) / (2 * (m1 - 3))) * (num1 / denom1) - 1
  
  m2 <- break_point_negative
  num2 <- 0
  for (j in 2:(m2-1)) {
    part2_1 <- delta_negative[j] + delta_negative[j-1] - mean( sort(delta_negative)[2:(m2-1)] + sort(delta_negative)[1:(m2-2)]  )
    num2 <- num2 + part2_1^2
  }
  denom2 <- 0
  for (k in 1:(m2-1)) {
    part2_2 <- delta_negative[k] - mean( sort(delta_negative)[2:m2] )
    denom2 <- denom2 + part2_2^2
  }
  SVR_negative <- ( (m2-1) / (2 * (m2 - 3))) * (num2 / denom2) - 1
  
  T <- m1 * SVR_positive^2 + m2 * SVR_negative^2
  df <- 2
  p_value <- 1 - pchisq(T, df) 
  SVR_test_result <- list(
    statistic = T,
    p_value = p_value,
    df = df
  )
  return(list(break_points = result, simple_matrix = matrix_modified,SVR_test_result=SVR_test_result))
}