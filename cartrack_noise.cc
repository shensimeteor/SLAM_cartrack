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
  double B0sqrtinv[2][2] = { {0.9129, 0.}, {-0.9129, 1.8257}};
  double xb0[2] = {0,0};
  template <typename T> bool operator()(const T* const x, T* residual) const {
//    residual[0] = T(25.0) - T(10.0) * x[0]+ x[0] * x[0];
    residual[0] = B0sqrtinv[0][0] * (x[0] - xb0[0]) + B0sqrtinv[0][1] * (x[1] - xb0[1]);
    residual[1] = B0sqrtinv[1][0] * (x[0] - xb0[0]) + B0sqrtinv[1][1] * (x[1] - xb0[1]);
    return true;
  }
};


//process noise
struct CostFunctorBt {
//  double Btsqrtinv[2][2] = { {1.5811, 0.}, {-1.5811, 3.1623}};  //for Process-Noise, Bt=[0.4,0.2],[0.2,0.2]
  double Btsqrtinv[2][2] = { {1.1180, 0.}, {0., 1.5811} };
  double speed[2] = {0.085, -0.085};
  template <typename T> bool operator()(const T* const x, const T* const xprev, T* residual) const {
    residual[0] = Btsqrtinv[0][0] * (x[0] - xprev[0] - speed[0]) + Btsqrtinv[0][1] * (x[1] - xprev[1] - speed[1]);
    residual[1] = Btsqrtinv[1][0] * (x[0] - xprev[0] - speed[0]) + Btsqrtinv[1][1] * (x[1] - xprev[1] - speed[1]);
    return true;
  }
};

//just obs projection, no propagation
struct CostFunctorRt {
  double Rsqrtinv[2][2][2] = { { {1.2910, 0.}, {1.2910, 1.2910} },  {{1.2910, 0.}, {-1.2910, 1.2910} } };  //0: Ra, 1: Rb
  int sensor_id; // 0: Ra, 1: Rb
  double obs_position[2];
  
  CostFunctorRt(double* obs, int sensor_id){
      obs_position[0] = obs[0];
      obs_position[1] = obs[1];
      this->sensor_id = sensor_id;
  }

  template <typename T> bool operator()(const T* const xt, T* residual) const {
    int i = this->sensor_id;
    residual[0] = Rsqrtinv[i][0][0] * (xt[0] -  obs_position[0]) \
                + Rsqrtinv[i][0][1] * (xt[1] -  obs_position[1]);
    residual[1] = Rsqrtinv[i][1][0] * (xt[0] -  obs_position[0]) \
                + Rsqrtinv[i][1][1] * (xt[1] -  obs_position[1]);
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.

  int nt=7; //0..6
  double** xs = new double* [nt];  //xs: the indepedent variable (7*2 size) to optimize
  for(int i=0; i<nt; i++){
      xs[i] = new double[2];
  }  //xs -> [xs[0], xs[1], ...],  xs[i] -> double[2]
  xs[0][0] = 0.;
  xs[0][1] = 0.;
  double initial_x0[2];
  for(int i=0;i<2; i++) { initial_x0[i] = xs[0][i]; }

  // Build the problem.
  Problem problem;

  //priori item
  CostFunction* cost_functionB0 =
      new AutoDiffCostFunction<CostFunctorB0, 2, 2>(new CostFunctorB0);
  problem.AddResidualBlock(cost_functionB0, NULL, xs[0]);

  //observation item
  double obsA[7][2] = {{1,0}, {1,-0.1}, {1,-0.2}, {1,-0.3}, {1,-0.4}, {1,-0.5}, {1,-0.5} };
  double obsB[7][2] = {{4,0}, {4,-0.1}, {4,-0.2}, {4,-0.3}, {4,-0.4}, {4,-0.5}, {4,-0.5} };
  for(int i=0;i <=6 ;i++){
      CostFunction* cost_functionRA = 
         new AutoDiffCostFunction<CostFunctorRt, 2, 2> (new CostFunctorRt(obsA[i], 0));
      problem.AddResidualBlock(cost_functionRA, NULL, xs[i]);
      CostFunction* cost_functionRB = 
         new AutoDiffCostFunction<CostFunctorRt, 2, 2> (new CostFunctorRt(obsB[i], 1));
      problem.AddResidualBlock(cost_functionRB, NULL, xs[i]);
  }

  //process noise item
  for(int i=1; i<=6; i++){
      CostFunction* cost_functionBt = 
         new AutoDiffCostFunction<CostFunctorBt, 2, 2, 2> (new CostFunctorBt);
      problem.AddResidualBlock(cost_functionBt, NULL, xs[i], xs[i-1]);
  }
  

  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x0[0] << "," << initial_x0[1]
            << " -> " << xs[0][0] << "," << xs[0][1] << "\n";
  // get the 0->6 position
  for(int i=0; i<=6; i++){
      printf("t=%d, position = %f, %f \n", i, xs[i][0], xs[i][1]);
  }
    
  return 0;
}
