// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.1.
//
// AD3 2.1 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.1 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.1.  If not, see <http://www.gnu.org/licenses/>.

#ifndef UTILS_H
#define UTILS_H

#if defined(_WIN32) && !defined(__MINGW32__)
#include <time.h>
#else
#include <sys/time.h>
#endif
#include <algorithm>
#include <string>
#include <vector>

#if defined(_WIN32) && !defined(__MINGW32__)
#include <windows.h> //I've ommited this line.
#ifndef _WINSOCKAPI_
struct timeval
{
    long tv_sec;  /* seconds */
    long tv_usec; /* and microseconds */
};
#endif
extern int
gettimeofday(struct timeval* tv, struct timezone* tz);
#endif

using namespace std;

namespace AD3 {

#define NEARLY_EQ_TOL(a, b, tol) (((a) - (b)) * ((a) - (b)) <= (tol))
#define NEARLY_BINARY(a, tol)                                                  \
    (NEARLY_EQ_TOL((a), 1.0, (tol)) || NEARLY_EQ_TOL((a), 0.0, (tol)))
#define NEARLY_ZERO_TOL(a, tol) (((a) <= (tol)) && ((a) >= (-(tol))))

#define MAX(a, b) (((a) < (b)) ? (b) : (a))

#define LOG_ZERO -1000
#define LOG_STABLE(a) (a == 0 ? LOG_ZERO : log(a))

extern int
diff_ms(timeval t1, timeval t2);

extern int
diff_us(timeval t1, timeval t2);

extern void
InsertionSort(pair<double, size_t> arr[], size_t length);

extern int
project_onto_simplex_cached(double* x,
                            size_t d,
                            double r,
                            vector<pair<double, size_t>>& y);

extern int
project_onto_simplex(double* x, size_t d, double r);

extern int
project_onto_cone_cached(double* x,
                         size_t d,
                         vector<pair<double, size_t>>& y,
                         vector<bool>& tight_ineq);

extern int
project_onto_weighted_cone(vector<double>& x,
                           const vector<double>& degrees,
                           const vector<double>& sqrt_degrees,
                           vector<bool>& tight_ineq);

extern int
project_onto_budget_constraint(double* x, size_t d, double budget);

extern int
project_onto_budget_constraint_cached(double* x,
                                      size_t d,
                                      double budget,
                                      vector<pair<double, size_t>>& y);

extern int
project_onto_knapsack_constraint(double* x,
                                 double* costs,
                                 size_t d,
                                 double budget);

extern int
project_onto_weighted_simplex(vector<double>& x,
                              const vector<double>& degrees,
                              const vector<double>& sqrt_degrees);

extern int
project_onto_weighted_knapsack(vector<double>& x,
                               const vector<double>& costs,
                               double budget,
                               const vector<double>& degrees,
                               const vector<double>& sqrt_degrees);

extern int
solve_canonical_qp_knapsack(const vector<double>& lower_bounds,
                            const vector<double>& upper_bounds,
                            const vector<double>& weights,
                            double total_weight,
                            vector<double>* solution);

extern void
StringSplit(const string& str, const string& delim, vector<string>* results);

extern void
TrimComments(const string& delim, string* line);

extern void
TrimLeft(const string& delim, string* line);

extern void
TrimRight(const string& delim, string* line);

extern void
Trim(const string& delim, string* line);

} // namespace AD3

#endif
