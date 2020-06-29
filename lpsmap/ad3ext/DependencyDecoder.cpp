// Copyright (c) 2012-2015 Andre Martins
// All Rights Reserved.
//
// This file is part of TurboParser 2.3.
//
// TurboParser 2.3 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TurboParser 2.3 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with TurboParser 2.3.  If not, see <http://www.gnu.org/licenses/>.


#include <vector>
#include <limits>

#include "DependencyDecoder.h"

using namespace std;


// Decoder for the basic model; it finds a maximum weighted arborescence
// using Edmonds' algorithm (which runs in O(n^2)).
void DependencyDecoder::RunChuLiuEdmondsIteration(
  vector<bool> *disabled,
  vector<vector<int> > *candidate_heads,
  vector<vector<double> > *candidate_scores,
  vector<int> *heads,
  double *value) {
  // Original number of nodes (including the root).
  int length = disabled->size();

  // Pick the best incoming arc for each node.
  heads->resize(length);
  vector<double> best_scores(length);
  for (int m = 1; m < length; ++m) {
    if ((*disabled)[m]) continue;
    int best = -1;
    for (int k = 0; k < (*candidate_heads)[m].size(); ++k) {
      if (best < 0 ||
          (*candidate_scores)[m][k] >(*candidate_scores)[m][best]) {
        best = k;
      }
    }
    if (best < 0) {
      // No spanning tree exists. Assign the parent of this node
      // to the root, and give it a minus infinity score.
      (*heads)[m] = 0;
      best_scores[m] = -std::numeric_limits<double>::infinity();
    } else {
      (*heads)[m] = (*candidate_heads)[m][best]; //best;
      best_scores[m] = (*candidate_scores)[m][best]; //best;
    }
  }

  // Look for cycles. Return after the first cycle is found.
  vector<int> cycle;
  vector<int> visited(length, 0);
  for (int m = 1; m < length; ++m) {
    if ((*disabled)[m]) continue;
    // Examine all the ancestors of m until the root or a cycle is found.
    int h = m;
    while (h != 0) {
      // If already visited, break and check if it is part of a cycle.
      // If visited[h] < m, the node was visited earlier and seen not
      // to be part of a cycle.
      if (visited[h]) break;
      visited[h] = m;
      h = (*heads)[h];
    }

    // Found a cycle to which h belongs.
    // Obtain the full cycle.
    if (visited[h] == m) {
      m = h;
      do {
        cycle.push_back(m);
        m = (*heads)[m];
      } while (m != h);
      break;
    }
  }

  // If there are no cycles, then this is a well formed tree.
  if (cycle.empty()) {
    *value = 0.0;
    for (int m = 1; m < length; ++m) {
      *value += best_scores[m];
    }
    return;
  }

  // Build a cycle membership vector for constant-time querying and compute the
  // score of the cycle.
  // Nominate a representative node for the cycle and disable all the others.
  double cycle_score = 0.0;
  vector<bool> in_cycle(length, false);
  int representative = cycle[0];
  for (int k = 0; k < cycle.size(); ++k) {
    int m = cycle[k];
    in_cycle[m] = true;
    cycle_score += best_scores[m];
    if (m != representative) (*disabled)[m] = true;
  }

  // Contract the cycle.
  // 1) Update the score of each child to the maximum score achieved by a parent
  // node in the cycle.
  vector<int> best_heads_cycle(length);
  for (int m = 1; m < length; ++m) {
    if ((*disabled)[m] || m == representative) continue;
    double best_score;
    // If the list of candidate parents of m is shorter than the length of
    // the cycle, use that. Otherwise, loop through the cycle.
    int best = -1;
    for (int k = 0; k < (*candidate_heads)[m].size(); ++k) {
      if (!in_cycle[(*candidate_heads)[m][k]]) continue;
      if (best < 0 || (*candidate_scores)[m][k] > best_score) {
        best = k;
        best_score = (*candidate_scores)[m][best];
      }
    }
    if (best < 0) continue;
    best_heads_cycle[m] = (*candidate_heads)[m][best];

    // Reconstruct the list of candidate heads for this m.
    int l = 0;
    for (int k = 0; k < (*candidate_heads)[m].size(); ++k) {
      int h = (*candidate_heads)[m][k];
      double score = (*candidate_scores)[m][k];
      if (!in_cycle[h]) {
        (*candidate_heads)[m][l] = h;
        (*candidate_scores)[m][l] = score;
        ++l;
      }
    }
    // If h is in the cycle and is not the representative node,
    // it will be dropped from the list of candidate heads.
    (*candidate_heads)[m][l] = representative;
    (*candidate_scores)[m][l] = best_score;
    (*candidate_heads)[m].resize(l + 1);
    (*candidate_scores)[m].resize(l + 1);
  }

  // 2) Update the score of each candidate parent of the cycle supernode.
  vector<int> best_modifiers_cycle(length, -1);
  vector<int> candidate_heads_representative;
  vector<double> candidate_scores_representative;

  vector<double> best_scores_cycle(length);
  // Loop through the cycle.
  for (int k = 0; k < cycle.size(); ++k) {
    int m = cycle[k];
    for (int l = 0; l < (*candidate_heads)[m].size(); ++l) {
      // Get heads out of the cycle.
      int h = (*candidate_heads)[m][l];
      if (in_cycle[h]) continue;

      double score = (*candidate_scores)[m][l] - best_scores[m];
      if (best_modifiers_cycle[h] < 0 || score > best_scores_cycle[h]) {
        best_modifiers_cycle[h] = m;
        best_scores_cycle[h] = score;
      }
    }
  }
  for (int h = 0; h < length; ++h) {
    if (best_modifiers_cycle[h] < 0) continue;
    double best_score = best_scores_cycle[h] + cycle_score;
    candidate_heads_representative.push_back(h);
    candidate_scores_representative.push_back(best_score);
  }

  // Reconstruct the list of candidate heads for the representative node.
  (*candidate_heads)[representative] = candidate_heads_representative;
  (*candidate_scores)[representative] = candidate_scores_representative;

  // Save the current head of the representative node (it will be overwritten).
  int head_representative = (*heads)[representative];

  // Call itself recursively.
  RunChuLiuEdmondsIteration(disabled,
                            candidate_heads,
                            candidate_scores,
                            heads,
                            value);

  // Uncontract the cycle.
  int h = (*heads)[representative];
  (*heads)[representative] = head_representative;
  (*heads)[best_modifiers_cycle[h]] = h;

  for (int m = 1; m < length; ++m) {
    if ((*disabled)[m]) continue;
    if ((*heads)[m] == representative) {
      // Get the right parent from within the cycle.
      (*heads)[m] = best_heads_cycle[m];
    }
  }
  for (int k = 0; k < cycle.size(); ++k) {
    int m = cycle[k];
    (*disabled)[m] = false;
  }
}

// Run the Chu-Liu-Edmonds algorithm for finding a maximal weighted spanning
// tree.
void DependencyDecoder::RunChuLiuEdmonds(int sentence_length,
                                         const vector<vector<int> > &index_arcs,
                                         const vector<double> &scores,
                                         vector<int> *heads,
                                         double *value) {
  vector<vector<int> > candidate_heads(sentence_length);
  vector<vector<double> > candidate_scores(sentence_length);
  vector<bool> disabled(sentence_length, false);

  for (int m = 1; m < sentence_length; ++m) {
    for (int h = 0; h < sentence_length; ++h) {
      int r = index_arcs[h][m];
      if (r < 0) continue;
      candidate_heads[m].push_back(h);
      candidate_scores[m].push_back(scores[r]);
    }
  }

  heads->assign(sentence_length, -1);
  RunChuLiuEdmondsIteration(&disabled, &candidate_heads,
                            &candidate_scores, heads, value);
}

// Run Eisner's algorithm for finding a maximal weighted projective dependency
// tree.
void DependencyDecoder::RunEisner(int sentence_length,
                                  int num_arcs,
                                  const vector<vector<int> > &index_arcs,
                                  const vector<double> &scores,
                                  vector<int> *heads,
                                  double *value) {
  heads->assign(sentence_length, -1);

  // Initialize CKY table.
  vector<vector<double> > complete_spans(sentence_length, vector<double>(
    sentence_length, 0.0));
  vector<vector<int> > complete_backtrack(sentence_length,
                                          vector<int>(sentence_length, -1));
  vector<double> incomplete_spans(num_arcs);
  vector<int> incomplete_backtrack(num_arcs, -1);

  // Loop from smaller items to larger items.
  for (int k = 1; k < sentence_length; ++k) {
    for (int s = 1; s < sentence_length - k; ++s) {
      int t = s + k;

      // First, create incomplete items.
      int left_arc_index = index_arcs[t][s];
      int right_arc_index = index_arcs[s][t];
      if (left_arc_index >= 0 || right_arc_index >= 0) {
        double best_value = -std::numeric_limits<double>::infinity();
        int best = -1;
        for (int u = s; u < t; ++u) {
          double val = complete_spans[s][u] + complete_spans[t][u + 1];
          if (best < 0 || val > best_value) {
            best = u;
            best_value = val;
          }
        }
        if (left_arc_index >= 0) {
          incomplete_spans[left_arc_index] =
            best_value + scores[left_arc_index];
          incomplete_backtrack[left_arc_index] = best;
        }
        if (right_arc_index >= 0) {
          incomplete_spans[right_arc_index] =
            best_value + scores[right_arc_index];
          incomplete_backtrack[right_arc_index] = best;
        }
      }

      // Second, create complete items.
      // 1) Left complete item.
      double best_value = -std::numeric_limits<double>::infinity();
      int best = -1;
      for (int u = s; u < t; ++u) {
        int left_arc_index = index_arcs[t][u];
        if (left_arc_index >= 0) {
          double val = complete_spans[u][s] + incomplete_spans[left_arc_index];
          if (best < 0 || val > best_value) {
            best = u;
            best_value = val;
          }
        }
      }
      complete_spans[t][s] = best_value;
      complete_backtrack[t][s] = best;

      // 2) Right complete item.
      best_value = -std::numeric_limits<double>::infinity();
      best = -1;
      for (int u = s + 1; u <= t; ++u) {
        int right_arc_index = index_arcs[s][u];
        if (right_arc_index >= 0) {
          double val = complete_spans[u][t] + incomplete_spans[right_arc_index];
          if (best < 0 || val > best_value) {
            best = u;
            best_value = val;
          }
        }
      }
      complete_spans[s][t] = best_value;
      complete_backtrack[s][t] = best;
    }
  }

  // Get the optimal (single) root.
  double best_value = -std::numeric_limits<double>::infinity();
  int best = -1;
  for (int s = 1; s < sentence_length; ++s) {
    int arc_index = index_arcs[0][s];
    if (arc_index >= 0) {
      double val = complete_spans[s][1] + complete_spans[s][sentence_length - 1] +
        scores[arc_index];
      if (best < 0 || val > best_value) {
        best = s;
        best_value = val;
      }
    }
  }

  *value = best_value;
  (*heads)[best] = 0;

  // Backtrack.
  RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs, best,
                     1, true, heads);
  RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs, best,
                     sentence_length - 1, true, heads);

  //*value = complete_spans[0][sentence_length-1];
  //RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs, 0,
  //                   sentence_length-1, true, heads);
}

void DependencyDecoder::RunEisnerBacktrack(
  const vector<int> &incomplete_backtrack,
  const vector<vector<int> > &complete_backtrack,
  const vector<vector<int> > &index_arcs,
  int h, int m, bool complete, vector<int> *heads) {
  if (h == m) return;
  if (complete) {
    //CHECK_GE(h, 0);
    //CHECK_LT(h, complete_backtrack.size());
    //CHECK_GE(m, 0);
    //CHECK_LT(m, complete_backtrack.size());
    int u = complete_backtrack[h][m];
    //CHECK_GE(u, 0) << h << " " << m;
    RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs,
                       h, u, false, heads);
    RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs,
                       u, m, true, heads);
  } else {
    int r = index_arcs[h][m];
    //CHECK_GE(r, 0);
    //CHECK_LT(r, incomplete_backtrack.size());
    //CHECK_GE(h, 0);
    //CHECK_LT(h, heads->size());
    //CHECK_GE(m, 0);
    //CHECK_LT(m, heads->size());
    (*heads)[m] = h;
    int u = incomplete_backtrack[r];
    if (h < m) {
      RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs,
                         h, u, true, heads);
      RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs,
                         m, u + 1, true, heads);
    } else {
      RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs,
                         m, u, true, heads);
      RunEisnerBacktrack(incomplete_backtrack, complete_backtrack, index_arcs,
                         h, u + 1, true, heads);
    }
  }
}

