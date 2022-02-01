// test for newest version
// TODO make sure I am actually using the release and not the head
// TODO mirror the use in naoth-soccer here
// TODO compare to older frugally version
#include <fdeep/fdeep.hpp>
int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, 4), {1, 2, 3, 4})});
    std::cout << fdeep::show_tensor5s(result) << std::endl;
}