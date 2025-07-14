#include "parameters.cuh"
#include "structure.cuh"
#include "nep.cuh"
#include "nep_charge.cuh"
#include "tnep.cuh"
#include "utilities/error.cuh"
#include <memory>
#include <vector>
#include <cstdio>

int main(int argc, char* argv[])
{
  if (argc < 3) {
    printf("Usage: %s nep.txt structure.xyz\n", argv[0]);
    return 1;
  }

  const char* nep_file = argv[1];
  const char* xyz_file = argv[2];

  Parameters para;
  std::vector<float> elite;
  para.load_from_nep_txt(nep_file, elite);

  std::vector<Structure> structures;
  if (!read_structures_from_file(xyz_file, para, structures)) {
    return 1;
  }

  std::vector<Dataset> dataset_vec(1);
  dataset_vec[0].construct(para, structures, 0, structures.size(), 0);

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

  dataset_vec[0].energy.copy_to_host(dataset_vec[0].energy_cpu.data());
  for (int nc = 0; nc < dataset_vec[0].Nc; ++nc) {
    printf("Energy[%d] = %g\n", nc, dataset_vec[0].energy_cpu[nc]);
  }

  return 0;
}
