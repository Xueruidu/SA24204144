% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{degree_centrality_cpp}
\alias{degree_centrality_cpp}
\title{Compute Degree Centrality of Network}
\usage{
degree_centrality_cpp(A)
}
\arguments{
\item{A}{A numeric matrix representing the adjacency matrix of the graph.
Each element is either 0 (no edge) or 1 (edge exists between nodes).}
}
\value{
A numeric vector containing the degree centrality scores for each node. 
        The degree centrality for node `i` is the number of edges connected to node `i`.
}
\description{
This function calculates the degree centrality of each node in a network 
             based on the adjacency matrix. The degree centrality is simply the number of 
             connections (edges) each node has in the network.
}
\examples{
\dontrun{
    A <- matrix(c(0, 1, 1, 0, 0,
                           1, 0, 1, 1, 0,
                           1, 1, 0, 1, 1,
                           0, 1, 1, 0, 1,
                           0, 0, 1, 1, 0), nrow = 5, byrow = TRUE)
    centrality_scores <- degree_centrality_cpp(A)
    print(centrality_scores)
}
}
