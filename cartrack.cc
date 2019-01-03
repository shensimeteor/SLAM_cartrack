// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

#include "ceres/ceres.h"
#include "glog/logging.h"
#include <stdio.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
// (x-5)^2 + (8-x)^2
struct CostFunctorB0 {
  double B0sqrtinv[2][2] = { {0.6787, 0.4195}, {-1.0982, 1.7769}};
  double xb0[2] = {0,0};
  template <typename T> bool operator()(const T* const x, T* residual) const {
//    residual[0] = T(25.0) - T(10.0) * x[0]+ x[0] * x[0];
    residual[0] = B0sqrtinv[0][0] * (x[0] - xb0[0]) + B0sqrtinv[0][1] * (x[1] - xb0[1]);
    residual[1] = B0sqrtinv[1][0] * (x[0] - xb0[0]) + B0sqrtinv[1][1] * (x[1] - xb0[1]);
    return true;
  }
};

struct CostFunctorR {
  double Rsqrtinv[2][2][2] = { { {-1.7769, -1.0982}, {0.4195, -0.6787} },  {{-1.7769, 1.0982}, {-0.4195, -0.6787} } };  //0: Ra, 1: Rb
  double speed[2] = {0.085, -0.085};
  double t;
  int sensor_id; // 0: Ra, 1: Rb
  double obs_position[2];
  
  CostFunctorR(double t, double* obs, int sensor_id){
      this->t = t;
      obs_position[0] = obs[0];
      obs_position[1] = obs[1];
      this->sensor_id = sensor_id;
  }

  template <typename T> bool operator()(const T* const x, T* residual) const {
    int i = this->sensor_id;
    residual[0] = Rsqrtinv[i][0][0] * (x[0] + (T) (speed[0]*t) - (T) obs_position[0]) \
                + Rsqrtinv[i][0][1] * (x[1] + (T) (speed[1]*t) - (T) obs_position[1]);
    residual[1] = Rsqrtinv[i][1][0] * (x[0] + (T) (speed[0]*t) - (T) obs_position[0]) \
                + Rsqrtinv[i][1][1] * (x[1] + (T) (speed[1]*t) - (T) obs_position[1]);
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x[2] = {0,0};
  double initial_x[2];
  for(int i=0;i<2; i++) { initial_x[i] = x[i]; }

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction* cost_functionB0 =
      new AutoDiffCostFunction<CostFunctorB0, 2, 2>(new CostFunctorB0);
  problem.AddResidualBlock(cost_functionB0, NULL, x);

  double obsA[7][2] = {{1,0}, {1,-0.1}, {1,-0.2}, {1,-0.3}, {1,-0.4}, {1,-0.5}, {1,-0.5} };
  double obsB[7][2] = {{4,0}, {4,-0.1}, {4,-0.2}, {4,-0.3}, {4,-0.4}, {4,-0.5}, {4,-0.5} };
  for(int i=0;i <=6 ;i++){
      CostFunction* cost_functionRA = 
         new AutoDiffCostFunction<CostFunctorR, 2, 2> (new CostFunctorR(i, obsA[i], 0));
      problem.AddResidualBlock(cost_functionRA, NULL, x);
      CostFunction* cost_functionRB = 
         new AutoDiffCostFunction<CostFunctorR, 2, 2> (new CostFunctorR(i, obsB[i], 1));
      problem.AddResidualBlock(cost_functionRB, NULL, x);
  }


  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x[0] << "," << initial_x[1]
            << " -> " << x[0] << "," << x[1] << "\n";
  // get the 0->6 position
  double speed[2] = {0.085, -0.085};
  double xs[2];
  xs[0] = x[0];
  xs[1] = x[1];
  for(int i=0; i<=6; i++){
      printf("t=%d, position = %f, %f \n", i, xs[0], xs[1]);
      xs[0] = speed[0] + xs[0];
      xs[1] = speed[1] + xs[1];
  }
  return 0;
}
