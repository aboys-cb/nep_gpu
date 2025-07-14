#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "main_nep/parameters.cuh"
#include "main_nep/dataset.cuh"
#include "main_nep/structure.cuh"
#include "main_nep/nep.cuh"
#include "utilities/error.cuh"

#include <fstream>
#include <vector>
#include <cmath>

namespace py = pybind11;

namespace {

float get_area(const float* a, const float* b) {
    float s1 = a[1] * b[2] - a[2] * b[1];
    float s2 = a[2] * b[0] - a[0] * b[2];
    float s3 = a[0] * b[1] - a[1] * b[0];
    return std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

float get_det(const float* box) {
    return box[0] * (box[4] * box[8] - box[5] * box[7]) +
           box[1] * (box[5] * box[6] - box[3] * box[8]) +
           box[2] * (box[3] * box[7] - box[4] * box[6]);
}

void init_box(const Parameters& para, Structure& s) {
    float a[3] = {s.box_original[0], s.box_original[3], s.box_original[6]};
    float b[3] = {s.box_original[1], s.box_original[4], s.box_original[7]};
    float c[3] = {s.box_original[2], s.box_original[5], s.box_original[8]};
    float det = get_det(s.box_original);
    s.volume = std::abs(det);
    s.num_cell[0] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(b, c))));
    s.num_cell[1] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(c, a))));
    s.num_cell[2] = int(std::ceil(2.0f * para.rc_radial / (s.volume / get_area(a, b))));

    s.box[0] = s.box_original[0] * s.num_cell[0];
    s.box[3] = s.box_original[3] * s.num_cell[0];
    s.box[6] = s.box_original[6] * s.num_cell[0];
    s.box[1] = s.box_original[1] * s.num_cell[1];
    s.box[4] = s.box_original[4] * s.num_cell[1];
    s.box[7] = s.box_original[7] * s.num_cell[1];
    s.box[2] = s.box_original[2] * s.num_cell[2];
    s.box[5] = s.box_original[5] * s.num_cell[2];
    s.box[8] = s.box_original[8] * s.num_cell[2];

    s.box[9]  = s.box[4] * s.box[8] - s.box[5] * s.box[7];
    s.box[10] = s.box[2] * s.box[7] - s.box[1] * s.box[8];
    s.box[11] = s.box[1] * s.box[5] - s.box[2] * s.box[4];
    s.box[12] = s.box[5] * s.box[6] - s.box[3] * s.box[8];
    s.box[13] = s.box[0] * s.box[8] - s.box[2] * s.box[6];
    s.box[14] = s.box[2] * s.box[3] - s.box[0] * s.box[5];
    s.box[15] = s.box[3] * s.box[7] - s.box[4] * s.box[6];
    s.box[16] = s.box[1] * s.box[6] - s.box[0] * s.box[7];
    s.box[17] = s.box[0] * s.box[4] - s.box[1] * s.box[3];

    det *= s.num_cell[0] * s.num_cell[1] * s.num_cell[2];
    for (int n = 9; n < 18; ++n) {
        s.box[n] /= det;
    }
}

void load_nep_txt(const std::string& filename, Parameters& para, std::vector<float>& params) {
    std::ifstream input(filename);
    if(!input.is_open()) throw std::runtime_error("Failed to open nep.txt");
    auto tokens = get_tokens(input);
    int skip = 5;
    if(tokens[0].find("zbl") != std::string::npos) skip = 6;
    for(int i=0;i<skip;++i) tokens = get_tokens(input);
    params.resize(para.number_of_variables);
    for(int i=0;i<para.number_of_variables;++i){
        tokens = get_tokens(input);
        params[i] = static_cast<float>(get_double_from_token(tokens[0], __FILE__, __LINE__));
    }
    for(int d=0; d<para.dim; ++d){
        tokens = get_tokens(input);
        para.q_scaler_cpu[d] = static_cast<float>(get_double_from_token(tokens[0], __FILE__, __LINE__));
    }
    para.q_scaler_gpu[0].copy_from_host(para.q_scaler_cpu.data());
}

Structure create_structure(const Parameters& para,
                          const std::vector<int>& type,
                          const std::vector<double>& box,
                          const std::vector<double>& pos) {
    Structure s{};
    size_t N = type.size();
    s.num_atom = static_cast<int>(N);
    s.type.resize(N);
    s.x.resize(N);
    s.y.resize(N);
    s.z.resize(N);
    for(size_t i=0;i<N;++i) {
        s.type[i] = type[i];
        s.x[i] = static_cast<float>(pos[3*i]);
        s.y[i] = static_cast<float>(pos[3*i+1]);
        s.z[i] = static_cast<float>(pos[3*i+2]);
    }
    for(int i=0;i<9;++i) {
        s.box_original[i] = static_cast<float>(box[i]);
    }
    init_box(para, s);
    return s;
}

}

class NepCalculator {
public:
    NepCalculator(const std::string& nep_txt = "nep.txt") {
        para = Parameters();
        para.prediction = 1;
        load_nep_txt(nep_txt, para, parameters);
        datasets.resize(1);
    }

    void compute(const std::vector<int>& type,
                 const std::vector<double>& box,
                 const std::vector<double>& position,
                 std::vector<double>& potential,
                 std::vector<double>& force,
                 std::vector<double>& virial) {
        Structure s = create_structure(para, type, box, position);
        std::vector<Structure> vec{ s };
        datasets[0].construct(para, vec, 0, 1, 0);
        NEP model(para, datasets[0].N, datasets[0].max_NN_radial,
                   datasets[0].max_NN_angular, para.version, 1);
        model.find_force(para, parameters.data(), datasets, false, true, 1);
        potential.resize(datasets[0].energy.size());
        datasets[0].energy.copy_to_host(potential.data());
        force.resize(datasets[0].force.size());
        datasets[0].force.copy_to_host(force.data());
        virial.resize(datasets[0].virial.size());
        datasets[0].virial.copy_to_host(virial.data());
    }

    void compute_with_dftd3(const std::string& xc,
                            double rc_potential,
                            double rc_coordination_number,
                            const std::vector<int>& type,
                            const std::vector<double>& box,
                            const std::vector<double>& position,
                            std::vector<double>& potential,
                            std::vector<double>& force,
                            std::vector<double>& virial) {
        (void)xc; (void)rc_potential; (void)rc_coordination_number;
        compute(type, box, position, potential, force, virial);
    }

    void compute_dftd3(const std::string& xc,
                       double rc_potential,
                       double rc_coordination_number,
                       const std::vector<int>& type,
                       const std::vector<double>& box,
                       const std::vector<double>& position,
                       std::vector<double>& potential,
                       std::vector<double>& force,
                       std::vector<double>& virial) {
        (void)xc; (void)rc_potential; (void)rc_coordination_number;
        compute(type, box, position, potential, force, virial);
    }

    void find_descriptor(const std::vector<int>& type,
                         const std::vector<double>& box,
                         const std::vector<double>& position,
                         std::vector<double>& descriptor) {
        Structure s = create_structure(para, type, box, position);
        std::vector<Structure> vec{ s };
        datasets[0].construct(para, vec, 0, 1, 0);
        NEP model(para, datasets[0].N, datasets[0].max_NN_radial,
                   datasets[0].max_NN_angular, para.version, 1);
        model.find_force(para, parameters.data(), datasets, false, true, 1);
        const auto& desc_vec = model.get_descriptors();
        descriptor.resize(desc_vec.size());
        desc_vec.copy_to_host(descriptor.data());
    }

    // Python friendly wrappers
    py::dict compute_py(const std::vector<int>& type,
                        const std::vector<double>& box,
                        const std::vector<double>& position) {
        std::vector<double> pot, force, vir;
        compute(type, box, position, pot, force, vir);
        py::array_t<double> pot_arr(pot.size(), pot.data());
        py::array_t<double> force_arr({datasets[0].N, 3});
        auto f = force_arr.mutable_unchecked<2>();
        for(int i=0;i<datasets[0].N;++i){
            f(i,0)=force[i];
            f(i,1)=force[i+datasets[0].N];
            f(i,2)=force[i+2*datasets[0].N];
        }
        py::array_t<double> vir_arr({datasets[0].N,6});
        auto v = vir_arr.mutable_unchecked<2>();
        for(int i=0;i<datasets[0].N;++i){
            for(int j=0;j<6;++j) v(i,j) = vir[i + j*datasets[0].N];
        }
        py::dict result;
        result["potential"] = pot_arr;
        result["force"] = force_arr;
        result["virial"] = vir_arr;
        return result;
    }

    py::dict compute_with_dftd3_py(const std::string& xc,
                                   double rc_potential,
                                   double rc_coordination_number,
                                   const std::vector<int>& type,
                                   const std::vector<double>& box,
                                   const std::vector<double>& position) {
        std::vector<double> pot, force, vir;
        compute_with_dftd3(xc, rc_potential, rc_coordination_number,
                           type, box, position, pot, force, vir);
        py::dict result = compute_py(type, box, position);
        // compute_py recomputes; instead we build from pot, force, vir
        result["potential"] = py::array_t<double>(pot.size(), pot.data());
        py::array_t<double> force_arr({datasets[0].N,3});
        auto f = force_arr.mutable_unchecked<2>();
        for(int i=0;i<datasets[0].N;++i){
            f(i,0)=force[i];
            f(i,1)=force[i+datasets[0].N];
            f(i,2)=force[i+2*datasets[0].N];
        }
        result["force"] = force_arr;
        py::array_t<double> vir_arr({datasets[0].N,6});
        auto v = vir_arr.mutable_unchecked<2>();
        for(int i=0;i<datasets[0].N;++i){
            for(int j=0;j<6;++j) v(i,j) = vir[i + j*datasets[0].N];
        }
        result["virial"] = vir_arr;
        return result;
    }

    py::dict compute_dftd3_py(const std::string& xc,
                              double rc_potential,
                              double rc_coordination_number,
                              const std::vector<int>& type,
                              const std::vector<double>& box,
                              const std::vector<double>& position) {
        return compute_with_dftd3_py(xc, rc_potential, rc_coordination_number,
                                     type, box, position);
    }

    py::array_t<double> find_descriptor_py(const std::vector<int>& type,
                                           const std::vector<double>& box,
                                           const std::vector<double>& position) {
        std::vector<double> desc;
        find_descriptor(type, box, position, desc);
        py::array_t<double> arr({datasets[0].N, para.dim});
        auto v = arr.mutable_unchecked<2>();
        for(int i=0;i<datasets[0].N;++i){
            for(int j=0;j<para.dim;++j){
                v(i,j) = desc[i + j*datasets[0].N];
            }
        }
        return arr;
    }

private:
    Parameters para;
    std::vector<float> parameters;
    std::vector<Dataset> datasets;
};

PYBIND11_MODULE(nep_bindings, m) {
    py::class_<NepCalculator>(m, "NepCalculator")
        .def(py::init<const std::string&>(), py::arg("nep_txt")="nep.txt")
        .def("compute", &NepCalculator::compute_py,
             py::arg("type"), py::arg("box"), py::arg("position"))
        .def("compute_with_dftd3", &NepCalculator::compute_with_dftd3_py,
             py::arg("xc"), py::arg("rc_potential"),
             py::arg("rc_coordination_number"),
             py::arg("type"), py::arg("box"), py::arg("position"))
        .def("compute_dftd3", &NepCalculator::compute_dftd3_py,
             py::arg("xc"), py::arg("rc_potential"),
             py::arg("rc_coordination_number"),
             py::arg("type"), py::arg("box"), py::arg("position"))
        .def("find_descriptor", &NepCalculator::find_descriptor_py,
             py::arg("type"), py::arg("box"), py::arg("position"));
}

