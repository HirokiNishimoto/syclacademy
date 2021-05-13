
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <CL/sycl.hpp>

#include <numeric>

// name kernel as add
template <typename T>
class add;

// Allow vectors of various types
template <typename T>
void parallel_add(std::vector<T>& inputA, std::vector<T>& inputB,
  std::vector<T>& output) {
  using namespace cl::sycl;

  auto size = inputA.size();

  queue defaultQueue;

  // Preparing buffer
  // buffer provides an abstract view of memory that can be accessed on either the host or device
  buffer<T, 1> inputABuf(inputA.data(), range<1>(size));
  buffer<T, 1> inputBBuf(inputB.data(), range<1>(size));
  buffer<T, 1> outputBuf(output.data(), range<1>(size));

  // buffer is accessed through " accessor" objects.
  // buffer is accessed through " accessor" objects.
  // `inputA` and `inputB` need not to be written anything. So, access::mode is read
  // `output` need to be written in the result of the calculation of vector addition. So, access::mode is write
  defaultQueue.submit([&](handler& cgh) {
    auto inputAAcc = inputABuf.template get_access<access::mode::read>(cgh);
    auto inputBAcc = inputBBuf.template get_access<access::mode::read>(cgh);
    auto outputAcc = outputBuf.template get_access<access::mode::write>(cgh);

    // kernel is named add<t>
    // <add<T>> is nameing kernel as `add`.
    // So, without this, code works fine.
    cgh.parallel_for<add<T>>(range<1>(size), [=](id<1> i) {
      outputAcc[i] = inputAAcc[i] + inputBAcc[i];
      });
    });
}

TEST_CASE("add_floats", "sycl_04_vector_add") {
  const int size = 1024;

  std::vector<float> inputA(size);
  std::vector<float> inputB(size);
  std::vector<float> output(size);

  // A = (0., 1., 2., ..., 1023.) <--- size: 1023
  // B = (0., 1., 2., ..., 1023.) <--- size: 1023
  // output = (0., 0., 0., ..., 0.) <- size: 1023
  std::iota(begin(inputA), end(inputA), 0.0f);
  std::iota(begin(inputB), end(inputB), 0.0f);
  std::fill(begin(output), end(output), 0.0f);

  // output = inputA + inputB
  parallel_add(inputA, inputB, output);


  for (int i = 0; i < size; i++) {
    REQUIRE(output[i] == static_cast<float>(i * 2.0f));
  }
}
