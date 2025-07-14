# NEP Prediction Call Flow

This document summarizes the key function calls executed when `prediction 1` is
set in `nep.in`.

1. **`main_backup.cu.bak`** – original entry point
   - Constructs a `Parameters` instance which reads `nep.in`.
   - Creates a `Fitness` object with the parameters.
   - Instantiates `SNES`, which handles both training and prediction.

2. **`SNES::SNES` constructor**
   - Initializes internal arrays and RNG state.
   - Calls `compute(para, fitness)` immediately.

3. **`SNES::compute`**
   - Prints whether training or predicting has started.
   - When `para.prediction == 1`:
     - Opens `nep.txt` and reads the stored parameters into `population`.
     - Loads descriptor scaling factors into `para.q_scaler_cpu`.
     - Calls `fitness->predict(para, population.data())`.

4. **`Fitness::predict`**
   - Creates the appropriate `Potential` implementation (`NEP`, `NEP_Charge` or
     `TNEP`) during its constructor.
   - For each batch of structures, invokes
     `Potential::find_force` to compute energies/forces.
   - Results are written to output files such as `energy_train.out` and
     `force_train.out`.

5. **`Potential::find_force`** (implemented in `nep.cu`, `nep_charge.cu` or
   `tnep.cu`)
   - Calculates neighbor lists and descriptors on the GPU.
   - Evaluates the neural network to obtain energies and forces.

The new `predict_main.cu` follows the same steps but skips the SNES layer. It
creates a `Parameters` object with `Parameters(true)` so no `nep.in` is read,
loads a `nep.txt` file using `Parameters::load_from_nep_txt`, reads structures
from a provided XYZ file, constructs a single `Dataset`, builds the proper
`Potential`, and directly calls `find_force` to obtain predictions.  Energies
are then averaged over all atoms in each configuration before printing.
