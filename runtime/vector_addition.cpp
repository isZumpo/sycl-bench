#include "common.h"

#include <vector>

using namespace cl;

class VectorAddition
{
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;
  cl::sycl::buffer<float, 1> a_sycl;
  cl::sycl::buffer<float, 1> b_sycl;
  cl::sycl::buffer<float, 1> c_sycl;
  BenchmarkArgs args;

public:
  VectorAddition(const BenchmarkArgs &_args) 
  : args(_args), 
  a(_args.problem_size, 5.0f),
  b(_args.problem_size, 4.0f),
  c(_args.problem_size, 0.0f),
  a_sycl{cl::sycl::buffer<float, 1>{a.data(), cl::sycl::range<1>{_args.problem_size}}},
  b_sycl{cl::sycl::buffer<float, 1>{b.data(), cl::sycl::range<1>{_args.problem_size}}},
  c_sycl{cl::sycl::buffer<float, 1>{c.data(), cl::sycl::range<1>{_args.problem_size}}}
  {}

  void setup() {

  }
  
  void run()
  {
    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
        auto a_acc = a_sycl.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b_sycl.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c_sycl.get_access<cl::sycl::access::mode::discard_write>(cgh);
        
        cgh.parallel_for<class vector_addition>(cl::sycl::range<1>{args.problem_size}, [=] (cl::sycl::id<1> idx) {
            auto index = idx[0];
            c_acc[index] = a_acc[index] + b_acc[index];
            });
    });
  }

  bool verify(VerificationSetting &ver) { 
    auto host_acc_c = c_sycl.get_access<sycl::access::mode::read>();
    for(int i = 0; i < args.problem_size; i++) {
        if(host_acc_c[i] != 9.0f) {
            std::cout << "\n** Invalid entry at C[" << i << "], got: " << host_acc_c[i] << ", but expected 9.0\n";
            return false;
        }
    }
    return true;
  }

   static std::string getBenchmarkName() {
    return "Runtime_VectorAddition";
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);

  app.run<VectorAddition>();
  
  return 0;
}
