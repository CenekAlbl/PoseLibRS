// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_RANSAC_IMPL_H_
#define POSELIB_RANSAC_IMPL_H_

#include "PoseLib/types.h"

#include <vector>
#include <chrono>
#include <iostream>

namespace poselib {


// Templated LO-RANSAC implementation (inspired by RansacLib from Torsten Sattler)
template <typename Solver, typename Model = CameraPose>
RansacStats ransac(Solver &estimator, const RansacOptions &opt, Model *best_model) {
    RansacStats stats;

    if (estimator.num_data < estimator.sample_sz) {
        return stats;
    }

    // Score/Inliers for best model found so far
    stats.num_inliers = 0;
    stats.model_score = std::numeric_limits<double>::max();
    // best inl/score for minimal model, used to decide when to LO
    size_t best_minimal_inlier_count = 0;
    double best_minimal_msac_score = std::numeric_limits<double>::max();

    const double log_prob_missing_model = std::log(1.0 - opt.success_prob);
    size_t inlier_count = 0;
    std::vector<Model> models;
    size_t dynamic_max_iter = opt.max_iterations;
    stats.iteration_times_us = std::vector<int>(opt.max_iterations);
    stats.inliers_per_iteration = std::vector<int>(opt.max_iterations);
    auto start = std::chrono::high_resolution_clock::now();
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; stats.iterations++) {

        if (stats.iterations > opt.min_iterations && stats.iterations > dynamic_max_iter) {
            break;
        }
        models.clear();
        estimator.generate_models(&models);

        // Find best model among candidates
        int best_model_ind = -1;
        for (size_t i = 0; i < models.size(); ++i) {
            double score_msac = estimator.score_model(models[i], &inlier_count);
            bool more_inliers = inlier_count > best_minimal_inlier_count;
            bool better_score = score_msac < best_minimal_msac_score;

            if (more_inliers || better_score) {
                if (more_inliers) {
                    best_minimal_inlier_count = inlier_count;
                }
                if (better_score) {
                    best_minimal_msac_score = score_msac;
                }
                best_model_ind = i;

                // check if we should update best model already
                if (score_msac < stats.model_score) {
                    stats.model_score = score_msac;
                    *best_model = models[i];
                    stats.num_inliers = inlier_count;
                }
            }
        }

        if (best_model_ind == -1){
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            stats.iteration_times_us[stats.iterations] = duration.count();
            stats.inliers_per_iteration[stats.iterations] = stats.num_inliers;
            continue;
        }
        // Refinement
        Model refined_model = models[best_model_ind];
        
        estimator.refine_model(&refined_model);
        stats.refinements++;
        double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
        if (refined_msac_score < stats.model_score) {
            stats.model_score = refined_msac_score;
            stats.num_inliers = inlier_count;
            *best_model = refined_model;
        }

        // update number of iterations
        stats.inlier_ratio = static_cast<double>(stats.num_inliers) / static_cast<double>(estimator.num_data);
        if (stats.inlier_ratio >= 0.9999) {
            // this is to avoid log(prob_outlier) = -inf below
            dynamic_max_iter = opt.min_iterations;
        } else if (stats.inlier_ratio <= 0.0001) {
            // this is to avoid log(prob_outlier) = 0 below
            dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(stats.inlier_ratio, estimator.sample_sz);
            dynamic_max_iter = std::ceil(log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        stats.iteration_times_us[stats.iterations] = duration.count();
        stats.inliers_per_iteration[stats.iterations] = stats.num_inliers;
    }

    // Final refinement
    Model refined_model = *best_model;
    estimator.refine_model(&refined_model);
    stats.refinements++;
    double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
    if (refined_msac_score < stats.model_score) {
        *best_model = refined_model;
        stats.num_inliers = inlier_count;
    }

    return stats;
}

// Multi model LO-RANSAC with two models - Solver uses simpler model and LO uses more complex model
// modelUpgrader initializes Model2 from Model1
template <typename Solver, typename Refiner, typename Model1, typename Model2>
RansacStats different_lo_model_ransac(Solver &estimator, Refiner &refiner, const RansacOptions &opt, Model2 *best_model) {
    RansacStats stats;

    if (estimator.num_data < estimator.sample_sz) {
        return stats;
    }

    // Score/Inliers for best model found so far
    stats.num_inliers = 0;
    stats.model_score = std::numeric_limits<double>::max();
    // best inl/score for minimal model, used to decide when to LO
    size_t best_minimal_inlier_count = 0;
    double best_minimal_msac_score = std::numeric_limits<double>::max();

    const double log_prob_missing_model = std::log(1.0 - opt.success_prob);
    size_t inlier_count = 0;
    std::vector<Model1> models;
    size_t dynamic_max_iter = opt.max_iterations;
    stats.iteration_times_us = std::vector<int>(opt.max_iterations);
    stats.inliers_per_iteration = std::vector<int>(opt.max_iterations);
    auto start = std::chrono::high_resolution_clock::now();
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; stats.iterations++) {

        if (stats.iterations > opt.min_iterations && stats.iterations > dynamic_max_iter) {
            break;
        }
        models.clear();
        estimator.generate_models(&models);

        // Find best model among candidates
        int best_model_ind = -1;
        for (size_t i = 0; i < models.size(); ++i) {
            double score_msac = estimator.score_model(models[i], &inlier_count);
            bool more_inliers = inlier_count > best_minimal_inlier_count;
            bool better_score = score_msac < best_minimal_msac_score;

            if (more_inliers || better_score) {
                if (more_inliers) {
                    best_minimal_inlier_count = inlier_count;
                }
                if (better_score) {
                    best_minimal_msac_score = score_msac;
                }
                best_model_ind = i;

                // check if we should update best model already
                if (score_msac < stats.model_score) {
                    stats.model_score = score_msac;
                    *best_model = model_upgrader(models[i]);
                    stats.num_inliers = inlier_count;
                }
            }
        }

        if (best_model_ind == -1){
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            stats.iteration_times_us[stats.iterations] = duration.count();
            stats.inliers_per_iteration[stats.iterations] = stats.num_inliers;
            continue;
        }

        // Refinement
        Model2 refined_model = model_upgrader(models[best_model_ind]);
        refiner.refine_model(&refined_model);
        stats.refinements++;   
        double refined_msac_score = refiner.score_model(refined_model, &inlier_count);

        if (refined_msac_score < stats.model_score) {
            stats.model_score = refined_msac_score;
            stats.num_inliers = inlier_count;
            *best_model = refined_model;
        }

        // update number of iterations
        stats.inlier_ratio = static_cast<double>(stats.num_inliers) / static_cast<double>(estimator.num_data);
        if (stats.inlier_ratio >= 0.9999) {
            // this is to avoid log(prob_outlier) = -inf below
            dynamic_max_iter = opt.min_iterations;
        } else if (stats.inlier_ratio <= 0.0001) {
            // this is to avoid log(prob_outlier) = 0 below
            dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(stats.inlier_ratio, estimator.sample_sz);
            dynamic_max_iter = std::ceil(log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        stats.iteration_times_us[stats.iterations] = duration.count();
        stats.inliers_per_iteration[stats.iterations] = stats.num_inliers;
    }

    // Final refinement
    Model2 refined_model = *best_model;
    refiner.refine_model(&refined_model);
    stats.refinements++;
    double refined_msac_score = refiner.score_model(refined_model, &inlier_count);
    if (refined_msac_score < stats.model_score) {
        *best_model = refined_model;
        stats.num_inliers = inlier_count;
    }

    return stats;
}

// Multi model LO-RANSAC with two models - Solver uses simpler model and LO uses more complex model
// modelUpgrader initializes Model2 from Model1
template <typename Solver1, typename Solver2, typename Model1, typename Model2>
RansacStats double_model_ransac(Solver1 &estimator1, Solver2 &estimator2, const RansacOptions &opt, Model2 *best_model) {
    RansacStats stats;

    if (estimator1.num_data < estimator1.sample_sz) {
        return stats;
    }

    // Score/Inliers for best model found so far
    stats.num_inliers = 0;
    stats.model_score = std::numeric_limits<double>::max();
    // best inl/score for minimal model, used to decide when to LO
    size_t best_minimal_inlier_count1 = 0;
    size_t best_minimal_inlier_count2 = 0;
    double best_minimal_msac_score1 = std::numeric_limits<double>::max();
    double best_minimal_msac_score2 = std::numeric_limits<double>::max();

    const double log_prob_missing_model = std::log(1.0 - opt.success_prob);
    size_t inlier_count = 0;
    std::vector<Model1> models1;
    std::vector<Model2> models2;
    size_t dynamic_max_iter = opt.max_iterations;
    stats.iteration_times_us = std::vector<int>(opt.max_iterations);
    stats.inliers_per_iteration = std::vector<int>(opt.max_iterations);
    auto start = std::chrono::high_resolution_clock::now();
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; stats.iterations++) {

        if (stats.iterations > opt.min_iterations && stats.iterations > dynamic_max_iter) {
            break;
        }

        //Start with etimator1
        models1.clear();
        estimator1.generate_models(&models1);

        // Find best model among candidates
        int best_model1_ind = -1;
        for (size_t i = 0; i < models1.size(); ++i) {
            double score_msac = estimator1.score_model(models1[i], &inlier_count);
            bool more_inliers = inlier_count > best_minimal_inlier_count1;
            bool better_score = score_msac < best_minimal_msac_score1;
            
            if (more_inliers || better_score) {
                if (more_inliers) {
                    best_minimal_inlier_count1 = inlier_count;
                }
                if (better_score) {
                    best_minimal_msac_score1 = score_msac;
                }
                best_model1_ind = i;

                std::cout << "found p3p model with " << inlier_count << " inliers and msac score " << score_msac << "\n";
                std::cout << "q: " << models1[i].q << " t: " << models1[i].t << " \n";

                // check if we should update best model already
                if (score_msac < stats.model_score) {
                    stats.model_score = score_msac;
                    *best_model = model_upgrader(models1[i]);
                }
            }
        }


        if (best_model1_ind == -1){
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            stats.iteration_times_us[stats.iterations] = duration.count();
            stats.inliers_per_iteration[stats.iterations] = stats.num_inliers;
            continue;
        }

        

        // If better model was found, use estimator2 initialized with the result of estimator1
        estimator2.init(models1[best_model1_ind]);

        models2.clear();
        estimator2.generate_models(&models2);

        // Find best model among candidates
        int best_model2_ind = -1;
        for (size_t i = 0; i < models2.size(); ++i) {
            double score_msac = estimator2.score_model(models2[i], &inlier_count);
            bool more_inliers = inlier_count > best_minimal_inlier_count2;
            bool better_score = score_msac < best_minimal_msac_score2;
            std::cout << "found r6p iter model with " << inlier_count << " inliers and msac score " << score_msac << "\n";
            std::cout << "q: " << models2[i].q << " t: " << models2[i].t << " w: " << models2[i].w << " v: " << models2[i].v << "\n";
            if (more_inliers || better_score) {
                if (more_inliers) {
                    best_minimal_inlier_count2 = inlier_count;
                }
                if (better_score) {
                    best_minimal_msac_score2 = score_msac;
                }
                best_model2_ind = i;

                // check if we should update best model already
                if (score_msac < stats.model_score) {
                    stats.model_score = score_msac;
                    *best_model = models2[i];
                    stats.num_inliers = inlier_count;
                }
            }
        }

        

        // Refinement
        Model2 refined_model;
        if(best_model2_ind == -1){ 
            std::cout << "estimator 2 did not find a better model\n";
            refined_model = model_upgrader(models1[best_model1_ind]);
        }else{
            std::cout << "using the model from estimator 2\n";
            refined_model = models2[best_model2_ind];
        }
        estimator2.refine_model(&refined_model);
        stats.refinements++;   
        double refined_msac_score = estimator2.score_model(refined_model, &inlier_count);

        if (refined_msac_score < stats.model_score) {
            stats.model_score = refined_msac_score;
            stats.num_inliers = inlier_count;
            *best_model = refined_model;
        }

        std::cout << "refined model has " << inlier_count << " inliers and msac score " << refined_msac_score << "\n";
        
        std::cout << "q: " << refined_model.q << " t: " << refined_model.t << " w: " << refined_model.w << " v: " << refined_model.v << "\n";
            

        // update number of iterations
        stats.inlier_ratio = static_cast<double>(stats.num_inliers) / static_cast<double>(estimator1.num_data);
        if (stats.inlier_ratio >= 0.9999) {
            // this is to avoid log(prob_outlier) = -inf below
            dynamic_max_iter = opt.min_iterations;
        } else if (stats.inlier_ratio <= 0.0001) {
            // this is to avoid log(prob_outlier) = 0 below
            dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(stats.inlier_ratio, estimator1.sample_sz);
            dynamic_max_iter = std::ceil(log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        stats.iteration_times_us[stats.iterations] = duration.count();
        stats.inliers_per_iteration[stats.iterations] = stats.num_inliers;
    }

    // Final refinement
    Model2 refined_model = *best_model;
    estimator2.refine_model(&refined_model);
    stats.refinements++;
    double refined_msac_score = estimator2.score_model(refined_model, &inlier_count);
    if (refined_msac_score < stats.model_score) {
        *best_model = refined_model;
        stats.num_inliers = inlier_count;
    }

    return stats;
}

} // namespace poselib

#endif