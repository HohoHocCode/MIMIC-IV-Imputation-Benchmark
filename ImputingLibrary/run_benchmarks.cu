#include "ImputingLibrary/cli_parser.hpp"
#include "ImputingLibrary/experiment_runner.hpp"
#include "ImputingLibrary/logger.hpp"
#include "imputation_lib/gpu/gpu_wrappers.hpp"
#include <iostream>
#include <memory>
#include <string>

using namespace impute;

int main(int argc, char **argv) {
  CLIParser parser(argc, argv);

  if (parser.has("help") || argc < 2) {
    parser.print_help();
    return 0;
  }

  // Expect meta.json inside the input directory
  std::string input_dir = parser.get("input", "./data");
  std::string meta_path = input_dir + "/meta.json";

  Logger::info("Initializing Imputing Library Runner...");
  // My new ExperimentRunner handles everything from meta.json
  ExperimentRunner runner(meta_path);

  // Configure Algorithm
  std::string algo_arg = parser.get("algo", "rlsp");
  int k = parser.get_int("k", 5);
  int pc = parser.get_int("pc", 2);
  float ridge = parser.get_float("ridge", 0.1f);
  int genes = parser.get_int("genes", 10);
  int rank = parser.get_int("rank", 10);
  int iter = parser.get_int("iter", 5);
  int chunk_size = parser.get_int("chunk", 4096);
  float tau = parser.get_float("tau", 1.0f);
  bool use_tensor = parser.has("tensor_cores");
  bool streaming = parser.has("streaming");

  // PRE-LOAD Knowledge base for full dataset
  KnowledgeBase::instance().load(runner.get_rows(), runner.get_cols());

  if (algo_arg == "rlsp") {
    runner.add_algorithm(std::make_shared<impute::RlspImputerGpu>(k, pc, use_tensor));
  } else if (algo_arg == "bgs") {
    runner.add_algorithm(std::make_shared<impute::BgsImputerGpu>(genes, ridge, use_tensor));
  } else if (algo_arg == "svd") {
    runner.add_algorithm(std::make_shared<impute::SvdImputerGpu>(rank, iter, use_tensor));
  } else if (algo_arg == "bpca") {
    runner.add_algorithm(std::make_shared<impute::BpcaImputerGpu>(k, iter, use_tensor));
  } else if (algo_arg == "lls") {
    runner.add_algorithm(std::make_shared<impute::LlsImputerGpu>(k, use_tensor));
  } else if (algo_arg == "ills") {
    runner.add_algorithm(std::make_shared<impute::IllsImputerGpu>(k, iter, use_tensor));
  } else if (algo_arg == "knn") {
    runner.add_algorithm(std::make_shared<impute::KnnImputerGpu>(k, use_tensor, tau));
  } else if (algo_arg == "sknn") {
    runner.add_algorithm(std::make_shared<impute::SknnImputerGpu>(k, use_tensor));
  } else if (algo_arg == "iknn") {
    runner.add_algorithm(std::make_shared<impute::IknnImputerGpu>(k, iter, use_tensor));
  } else if (algo_arg == "ls") {
    runner.add_algorithm(std::make_shared<impute::LsImputerGpu>(k));
  } else if (algo_arg == "slls") {
    runner.add_algorithm(std::make_shared<impute::SllsImputerGpu>(k, use_tensor));
  } else if (algo_arg == "gmc") {
    runner.add_algorithm(std::make_shared<impute::GmcImputerGpu>(k, iter, use_tensor));
  } else if (algo_arg == "cmve") {
    runner.add_algorithm(std::make_shared<impute::CmveImputerGpu>(k, use_tensor));
  } else if (algo_arg == "amvi") {
    runner.add_algorithm(std::make_shared<impute::AmviImputerGpu>(k, use_tensor));
  } else if (algo_arg == "arls") {
    runner.add_algorithm(std::make_shared<impute::ArlsImputerGpu>(k, ridge, use_tensor));
  } else if (algo_arg == "lincmb") {
    runner.add_algorithm(std::make_shared<impute::LinCmbImputerGpu>(k, rank, use_tensor));
  } else if (algo_arg == "pocs") {
    runner.add_algorithm(std::make_shared<impute::PocsImputerGpu>(rank, iter));
  } else if (algo_arg == "goimpute") {
    runner.add_algorithm(std::make_shared<impute::GoImputerGpu>(k, use_tensor));
  } else if (algo_arg == "imiss") {
    runner.add_algorithm(std::make_shared<impute::ImissImputerGpu>(k, use_tensor));
  } else if (algo_arg == "wenni") {
    runner.add_algorithm(std::make_shared<impute::WenniImputerGpu>(k, use_tensor));
  } else if (algo_arg == "wenni_bc") {
    runner.add_algorithm(std::make_shared<impute::WenniBcImputerGpu>(k, use_tensor));
  } else if (algo_arg == "metamiss") {
    runner.add_algorithm(std::make_shared<impute::MetaMissImputerGpu>(k, use_tensor));
  } else if (algo_arg == "halimpute") {
    runner.add_algorithm(std::make_shared<impute::HaImputerGpu>(k, use_tensor));
  } else {
    Logger::error("Unknown algorithm: " + algo_arg);
    return 1;
  }

  if (streaming) {
    runner.run_streaming("benchmark_results.csv", chunk_size);
  } else {
    runner.run("benchmark_results.csv");
  }

  Logger::info("Benchmark Complete. Results saved to benchmark_results.csv");
  return 0;
}
