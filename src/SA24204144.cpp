#include <Rcpp.h>
#include <vector>
#include <queue>
#include <stack>
using namespace Rcpp;

//' @title Compute Betweenness Centrality of Network
//' @description This function calculates the betweenness centrality of each node in a network 
//'              based on the adjacency matrix. It uses the Floyd-Warshall algorithm 
//'              implemented in C++ to calculate shortest paths between nodes.
//' @param A A numeric matrix representing the adjacency matrix.
//' @return A numeric vector containing the betweenness centrality scores for each node.
//' @examples
//' \dontrun{
//'     A <- matrix(c(0, 1, -1, 1, 0, 0, -1, 0, 0), nrow = 3)
//'     centrality_scores <- betweenness_centrality(A)
//'     print(centrality_scores)
//' }
//' @export
// [[Rcpp::export]]
NumericVector betweenness_centrality_cpp(NumericMatrix A) {
  int n = A.nrow();
  NumericVector centrality(n, 0.0); // Initialize centrality vector
  
  for (int s = 0; s < n; ++s) {
    // Shortest path lengths and counts
    std::vector<int> shortest_paths(n, -1); // -1 indicates infinite distance
    std::vector<int> num_paths(n, 0);
    std::vector<std::vector<int>> predecessors(n); // Predecessors for each node
    std::queue<int> queue; // BFS queue
    std::stack<int> stack; // Process order stack
    
    // Initialize for source node s
    shortest_paths[s] = 0;
    num_paths[s] = 1;
    queue.push(s);
    
    // BFS to compute shortest paths and counts
    while (!queue.empty()) {
      int v = queue.front();
      queue.pop();
      stack.push(v);
      
      for (int w = 0; w < n; ++w) {
        if (A(v, w) > 0) { // Check if there is an edge
          if (shortest_paths[w] == -1) { // First visit to w
            shortest_paths[w] = shortest_paths[v] + 1;
            queue.push(w);
          }
          if (shortest_paths[w] == shortest_paths[v] + 1) { // Shortest path via v
            num_paths[w] += num_paths[v];
            predecessors[w].push_back(v);
          }
        }
      }
    }
    
    // Dependency calculation
    std::vector<double> dependency(n, 0.0);
    while (!stack.empty()) {
      int w = stack.top();
      stack.pop();
      for (int v : predecessors[w]) {
        dependency[v] += (static_cast<double>(num_paths[v]) / num_paths[w]) * (1.0 + dependency[w]);
      }
      if (w != s) {
        centrality[w] += dependency[w];
      }
    }
  }
  
  // Normalize centrality for undirected graph
  for (int i = 0; i < n; ++i) {
    centrality[i] /= 2.0;
  }
  
  return centrality;
}




//' @title Compute Degree Centrality of Network
//' @description This function calculates the degree centrality of each node in a network 
//'              based on the adjacency matrix. The degree centrality is simply the number of 
//'              connections (edges) each node has in the network.
//' @param A A numeric matrix representing the adjacency matrix of the graph.
//'                   Each element is either 0 (no edge) or 1 (edge exists between nodes).
//' @return A numeric vector containing the degree centrality scores for each node. 
//'         The degree centrality for node `i` is the number of edges connected to node `i`.
//' @examples
//' \dontrun{
//'     A <- matrix(c(0, 1, 1, 0, 0,
//'                            1, 0, 1, 1, 0,
//'                            1, 1, 0, 1, 1,
//'                            0, 1, 1, 0, 1,
//'                            0, 0, 1, 1, 0), nrow = 5, byrow = TRUE)
//'     centrality_scores <- degree_centrality(A)
//'     print(centrality_scores)
//' }
//' @export
// [[Rcpp::export]]
 NumericVector degree_centrality_cpp(NumericMatrix A) {
   int n = A.nrow(); // Get the number of nodes (rows) in the adjacency matrix
   NumericVector centrality(n); // Initialize a vector to store degree centrality of each node
   
   // Iterate over each node (row of the adjacency matrix)
   for (int i = 0; i < n; i++) {
     int degree = 0; // Initialize the degree of node i
     
     // Iterate over all other nodes to check for connections (edges)
     for (int j = 0; j < n; j++) {
       if (A(i, j) != 0) { // If there is an edge between node i and node j
         degree += 1; // Increment the degree of node i
       }
     }
     
     centrality[i] = degree/2; // Store the degree centrality of node i
   }
   
   return centrality; // Return the degree centrality vector
 }
