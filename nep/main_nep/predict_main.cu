#include "parameters.cuh"
#include "structure.cuh"
#include "dataset.cuh"
#include "nep.cuh"
#include "nep_charge.cuh"
#include "tnep.cuh"
#include "utilities/error.cuh"
#include <memory>
#include <vector>
#include <sstream>
#include <cstdio>
#include <chrono>
#include <stdio.h>
void  output(
  bool is_stress,
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int n = 0; n < num_components; ++n) {
      int offset = n * dataset.N + dataset.Na_sum_cpu[nc];
      float data_nc = 0.0f;
      for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
        data_nc += prediction[offset + m];
      }
      if (!is_stress) {
        fprintf(fid, "%g ", data_nc / dataset.Na_cpu[nc]);
      } else {
        fprintf(fid, "%g ", data_nc / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION);
      }
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.Nc + nc];
      if (is_stress) {
        ref_value *= dataset.Na_cpu[nc] / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
      }
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}

void  update_energy_force_virial(
  FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset)
{
  dataset.energy.copy_to_host(dataset.energy_cpu.data());
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  dataset.force.copy_to_host(dataset.force_cpu.data());

  std::vector<float> force_ref = dataset.force_ref_cpu;
  std::vector<float> energy_ref = dataset.energy_ref_cpu;
  std::vector<float> virial_ref = dataset.virial_ref_cpu;

  for (int nc = 0; nc < dataset.Nc; ++nc) {
    int offset = dataset.Na_sum_cpu[nc];
    if (!dataset.structures[nc].has_force) {
      for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
        int n = offset + m;
        force_ref[n] = dataset.force_cpu[n];
        force_ref[n + dataset.N] = dataset.force_cpu[n + dataset.N];
        force_ref[n + dataset.N * 2] = dataset.force_cpu[n + dataset.N * 2];
      }
    }

    if (!dataset.structures[nc].has_energy) {
      float sum = 0.0f;
      for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
        sum += dataset.energy_cpu[offset + m];
      }
      energy_ref[nc] = sum / dataset.Na_cpu[nc];
    }

    if (!dataset.structures[nc].has_virial) {
      for (int comp = 0; comp < 6; ++comp) {
        float sum = 0.0f;
        for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
          sum += dataset.virial_cpu[comp * dataset.N + offset + m];
        }
        virial_ref[comp * dataset.Nc + nc] = sum / dataset.Na_cpu[nc];
      }
    }
  }

  for (int nc = 0; nc < dataset.Nc; ++nc) {
    int offset = dataset.Na_sum_cpu[nc];
    for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
      int n = offset + m;
      fprintf(
        fid_force,
        "%g %g %g %g %g %g\n",
        dataset.force_cpu[n],
        dataset.force_cpu[n + dataset.N],
        dataset.force_cpu[n + dataset.N * 2],
        force_ref[n],
        force_ref[n + dataset.N],
        force_ref[n + dataset.N * 2]);
    }
  }

  output(false, 1, fid_energy, dataset.energy_cpu.data(), energy_ref.data(), dataset);

  output(false, 6, fid_virial, dataset.virial_cpu.data(), virial_ref.data(), dataset);
  output(true, 6, fid_stress, dataset.virial_cpu.data(), virial_ref.data(), dataset);
}




int main(int argc, char* argv[])
{
  if (argc < 3) {
    printf("Usage: %s nep.txt structure.xyz [batch_size]\n", argv[0]);
    return 1;
  }

  const char* nep_file = argv[1];
  const char* xyz_file = argv[2];

  Parameters para(true);
  std::vector<float> elite;
  para.load_from_nep_txt(nep_file, elite);
   para.prediction=1;
   para.output_descriptor=1;
  int batch_size = para.batch_size;
  if (argc > 3) {
    batch_size = atoi(argv[3]);
    if (batch_size < 1) {
      printf("Invalid batch_size %d.\n", batch_size);
      return 1;
    }
  }
  const auto time_begin1 = std::chrono::high_resolution_clock::now();

  std::vector<Structure> structures;
  if (!read_structures_from_file(xyz_file, para, structures)) {
    return 1;
  }
  const auto time_finish1 = std::chrono::high_resolution_clock::now();

  const std::chrono::duration<double> time_used1 = time_finish1 - time_begin1;
  print_line_1();
  printf("read_structures_from_file initialization = %f s.\n", time_used1.count());
  std::vector<Dataset> dataset_vec(1);
  int total_configs = structures.size();
  int printed_index = 0;

  std::string base_name(xyz_file);
  size_t slash = base_name.find_last_of("/");
  if (slash != std::string::npos) {
    base_name = base_name.substr(slash + 1);
  }
  size_t dot = base_name.find_last_of('.');
  std::string name_no_ext = (dot == std::string::npos) ? base_name : base_name.substr(0, dot);

  std::string force_file = "force_" + name_no_ext + ".out";
  std::string energy_file = "energy_" + name_no_ext + ".out";
  std::string virial_file = "virial_" + name_no_ext + ".out";
  std::string stress_file = "stress_" + name_no_ext + ".out";

  if (name_no_ext == "train") {
    para.descriptor_filename = "descriptor.out";
  } else {
    para.descriptor_filename = std::string("descriptor_") + name_no_ext + ".out";
  }

  const auto time_begin2 = std::chrono::high_resolution_clock::now();
    FILE* fid_force = my_fopen(force_file.c_str(), "w");
    FILE* fid_energy = my_fopen(energy_file.c_str(), "w");
    FILE* fid_virial = my_fopen(virial_file.c_str(), "w");
    FILE* fid_stress = my_fopen(stress_file.c_str(), "w");
  for (int start = 0; start < total_configs; start += batch_size) {
    int end = start + batch_size;
    if (end > total_configs) {
      end = total_configs;
    }

    dataset_vec[0].construct(para, structures, start, end, 0);

    std::unique_ptr<Potential> potential;
    if (para.train_mode == 1 || para.train_mode == 2) {
      potential.reset(new TNEP(para,
                               dataset_vec[0].N,
                               dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                               dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                               para.version,
                               1));
    } else {
      if (para.charge_mode) {
        potential.reset(new NEP_Charge(para,
                                       dataset_vec[0].N,
                                       dataset_vec[0].Nc,
                                       dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                                       dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                                       para.version,
                                       1));
      } else {
        potential.reset(new NEP(para,
                                dataset_vec[0].N,
                                dataset_vec[0].N * dataset_vec[0].max_NN_radial,
                                dataset_vec[0].N * dataset_vec[0].max_NN_angular,
                                para.version,
                                1));
      }
    }

    potential->find_force(para, elite.data(), dataset_vec, false, true, 1);
        update_energy_force_virial(
        fid_energy, fid_force, fid_virial, fid_stress, dataset_vec[0] );
//     dataset_vec[0].energy.copy_to_host(dataset_vec[0].energy_cpu.data());
//     for (int nc = 0; nc < dataset_vec[0].Nc; ++nc) {
//       int offset = dataset_vec[0].Na_sum_cpu[nc];
//       float energy_sum = 0.0f;
//       for (int m = 0; m < dataset_vec[0].Na_cpu[nc]; ++m) {
//         energy_sum += dataset_vec[0].energy_cpu[offset + m];
//       }
//       printf("Energy[%d] = %g\n", printed_index + nc,
//              energy_sum / dataset_vec[0].Na_cpu[nc]);
//     }

    printed_index += dataset_vec[0].Nc;
  }
    fclose(fid_energy);
    fclose(fid_force);
    fclose(fid_virial);
    fclose(fid_stress);
   const auto time_finish2 = std::chrono::high_resolution_clock::now();

  const std::chrono::duration<double> time_used2 = time_finish2 - time_begin2;

    printf("Time used for predicting = %f s.\n", time_used2.count());

  return 0;
}
