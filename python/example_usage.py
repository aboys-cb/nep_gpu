import numpy as np
import nep_bindings

# Example structure: two atoms in a cubic box
calculator = nep_bindings.NepCalculator("nep.txt")

types = [0, 0]
box = [3.0, 0.0, 0.0,
       0.0, 3.0, 0.0,
       0.0, 0.0, 3.0]
positions = [0.0, 0.0, 0.0,
             1.5, 0.0, 0.0]

result = calculator.compute(types, box, positions)
print("Potential", result["potential"])
print("Forces", result["force"])
print("Virial", result["virial"])

descriptor = calculator.find_descriptor(types, box, positions)
print("Descriptor shape", descriptor.shape)
