#ifndef EXPERIMENT_RUNNER_HPP
#define EXPERIMENT_RUNNER_HPP

#include "../imputation_lib/i_imputer.hpp"
#include "logger.hpp"
#include "../imputation_lib/gpu/knowledge_loader.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "json.hpp"

using json = nlohmann::json;

struct DatasetMeta {
  int rows;
  int cols;
  std::string x_file;
  std::string m_file;
  std::string holdout_idx_path;
  std::string holdout_y_path;
};

class ExperimentRunner {
public:
  ExperimentRunner(const std::string &meta_json_path) {
    std::ifstream f(meta_json_path);
    if (!f.is_open()) throw std::runtime_error("Could not open meta.json");
    json data = json::parse(f);
    meta_.rows = data["rows"];
    meta_.cols = data["cols"];
    meta_.x_file = data["x_file"];
    meta_.m_file = data["m_file"];
    meta_.holdout_idx_path = data.value("holdout_idx_path", "");
    meta_.holdout_y_path = data.value("holdout_y_path", "");

    N = meta_.rows;
    D = meta_.cols;

    std::string base_dir = meta_json_path.substr(0, meta_json_path.find_last_of("/\\") + 1);
    load_binary(base_dir + meta_.x_file, X_orig, (size_t)N * D);
    load_binary(base_dir + meta_.m_file, Mask, (size_t)N * D);

    // SAFETY: Clear any values in X_orig where Mask is 0 to prevent data leakage
    int cleared = 0;
    for (size_t i = 0; i < (size_t)N * D; ++i) {
        if (Mask[i] == 0) {
            X_orig[i] = 0.0f;
            cleared++;
        }
    }
    std::cout << "Safety: Cleared " << cleared << " missing values in X_orig." << std::endl;

    if (!meta_.holdout_idx_path.empty()) {
      load_binary(base_dir + meta_.holdout_idx_path, holdout_idx);
      load_binary(base_dir + meta_.holdout_y_path, holdout_y);
      has_eval_data = true;
      std::cout << "Loaded " << holdout_idx.size() << " holdout points." << std::endl;
      
      // FORCE MASKING of holdout points (critical for D1 which has mask=1 in pre-existing binaries)
      int forced = 0;
      for (int idx : holdout_idx) {
          if (idx >= 0 && (size_t)idx < Mask.size()) {
              if (Mask[idx] == 1) {
                  Mask[idx] = 0;
                  X_orig[idx] = 0.0f; 
                  forced++;
              }
          }
      }
      if (forced > 0) {
          std::cout << "Safety: Forced " << forced << " holdout points to be masked (imputation target)." << std::endl;
      }

      if (holdout_idx.size() > 0) {
          std::cout << "First holdout point: idx=" << holdout_idx[0] << " y=" << holdout_y[0] << " mask=" << (int)Mask[holdout_idx[0]] << std::endl;
      }
    }
  }

  int get_rows() const { return N; }
  int get_cols() const { return D; }

  void add_algorithm(std::shared_ptr<impute::IImputer> algo) { algorithms.push_back(algo); }

  void run(const std::string &output_csv) {
    std::ofstream csv(output_csv, std::ios::app);
    std::ifstream check_file(output_csv);
    if (check_file.peek() == std::ifstream::traits_type::eof()) {
        csv << "Algorithm,Class,NRMSE,MAE,RMSE,TimeMs\n";
    }
    check_file.close();

    for (auto &algo : algorithms) {
      std::cout << "Benchmarking " << algo->name() << "..." << std::endl;
      std::vector<float> X_work = X_orig;
      
      auto start = std::chrono::high_resolution_clock::now();
      algo->impute(X_work.data(), Mask.data(), N, D);
      auto end = std::chrono::high_resolution_clock::now();
      double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

      float mae = 0, rmse = 0, nrmse = 0;
      if (has_eval_data) {
        evaluate(X_work, mae, rmse, nrmse);
      }
      
      std::string cat = get_category(algo->name());
      csv << "\"" << algo->name() << "\"," << cat << "," << nrmse << "," << mae << "," << rmse << "," << elapsed_ms << std::endl;
      std::cout << "Result: " << algo->name() << " NRMSE=" << nrmse << " (" << elapsed_ms << " ms)" << std::endl;
    }
  }

  void run_streaming(const std::string &output_csv, int chunk_size = 4096) {
    run(output_csv);
  }

  std::string get_category(const std::string& name) {
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    if (n.find("svd") != std::string::npos || n.find("bpca") != std::string::npos) return "Global";
    if (n.find("pocs") != std::string::npos || n.find("goimpute") != std::string::npos ||
        n.find("hal") != std::string::npos || n.find("wenni") != std::string::npos ||
        n.find("imiss") != std::string::npos || n.find("metamiss") != std::string::npos) return "Knowledge";
    if (n.find("lincmb") != std::string::npos) return "Hybrid";
    return "Local";
  }

private:
  int N, D;
  std::vector<float> X_orig;
  std::vector<uint8_t> Mask;
  bool has_eval_data = false;
  std::vector<int> holdout_idx;
  std::vector<float> holdout_y;
  std::vector<std::shared_ptr<impute::IImputer>> algorithms;
  DatasetMeta meta_;

  template <typename T>
  void load_binary(const std::string &path, std::vector<T> &vec, size_t size = 0) {
    std::ifstream it(path, std::ios::binary);
    if (!it.is_open()) return;
    if (size == 0) {
      it.seekg(0, std::ios::end);
      size = it.tellg() / sizeof(T);
      it.seekg(0, std::ios::beg);
    }
    vec.assign(size, 0);
    it.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
  }

  void evaluate(const std::vector<float> &X_pred, float &mae, float &rmse, float &nrmse) {
    double sum_abs = 0, sum_sq = 0, sum_sq_truth = 0;
    double sum_pred = 0, sum_truth = 0;
    size_t count = 0;
    size_t non_zero_preds = 0;

    for (size_t k = 0; k < holdout_idx.size(); ++k) {
      int idx = holdout_idx[k];
      if (idx < 0 || (size_t)idx >= X_pred.size()) continue;
      
      float p = X_pred[idx];
      float t = holdout_y[k];
      float diff = p - t;
      
      sum_abs += std::abs(diff); 
      sum_sq += (double)diff * diff;
      sum_sq_truth += (double)t * t;
      sum_pred += p;
      sum_truth += t;
      if (std::abs(p) > 1e-7) non_zero_preds++;
      count++;
    }

    if (count > 0) {
      mae = (float)(sum_abs / count); 
      rmse = (float)sqrt(sum_sq / count);
      nrmse = (sum_sq_truth > 0) ? (float)sqrt(sum_sq / sum_sq_truth) : 0.0f;
      
      std::cout << "  [Eval Dev] Count: " << count 
                << " | Non-zero Preds: " << non_zero_preds 
                << " | Pred Mean: " << (sum_pred / count)
                << " | Truth Mean: " << (sum_truth / count) << std::endl;
    }
    
    if (nrmse > 0.999 && nrmse < 1.001 && count > 0 && non_zero_preds == 0) {
        std::cout << "  WARNING: Algorithm produced ZERO imputations for all holdout points." << std::endl;
    }
  }
};
#endif
