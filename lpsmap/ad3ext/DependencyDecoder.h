#pragma once

/// Copyright (c) 2012-2015 Andre Martins
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

#include<vector>

using std::vector;

class DependencyDecoder {
public:
  DependencyDecoder() {};
  virtual ~DependencyDecoder() {};

  void RunChuLiuEdmonds(int sentence_length,
                        const vector<vector<int> > &index_arcs,
                        const vector<double> &scores,
                        vector<int> *heads,
                        double *value);

  void RunEisner(int sentence_length,
                 int num_arcs,
                 const vector<vector<int> > &index_arcs,
                 const vector<double> &scores,
                 vector<int> *heads,
                 double *value);

  void RunChuLiuEdmondsIteration(vector<bool> *disabled,
                                 vector<vector<int> > *candidate_heads,
                                 vector<vector<double> > *candidate_scores,
                                 vector<int> *heads,
                                 double *value);

  void RunEisnerBacktrack(const vector<int> &incomplete_backtrack,
                          const vector<vector<int> > &complete_backtrack,
                          const vector<vector<int> > &index_arcs,
                          int h, int m, bool complete, vector<int> *heads);

};
