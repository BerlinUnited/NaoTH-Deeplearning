#include "dummy_model2.h"

#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

static float w_5  [3][3][1][4] = { 1.7989498e-01f,-2.6024538e-01f,-1.6197309e-01f,2.3455852e-01f,-8.8465005e-02f,3.1639665e-01f,-2.1744877e-02f,-4.161206e-02f,-1.5569532e-01f,-2.6148674e-01f,-1.7537029e-01f,-1.2105514e-01f,1.4158511e-01f,-1.3164273e-01f,-3.6495683e-01f,1.8553287e-01f,-2.7371728e-01f,3.118012e-01f,-2.9049397e-02f,-3.1273824e-01f,-1.2876275e-01f,2.4791771e-01f,1.2439287e-01f,3.1311607e-01f,-1.0389939e-01f,-2.839204e-01f,-3.3317095e-01f,1.7308885e-01f,4.7162265e-02f,-8.973381e-02f,2.935205e-01f,3.045975e-01f,-5.1541924e-02f,2.858634e-01f,-3.2106638e-01f,-2.402705e-01f };
static float b_5  [4] = { 0.e+00f,0.e+00f,0.e+00f,0.e+00f };
static float w_4  [3][3][4][4] = { 9.4813764e-02f,1.022076e-01f,-2.4721712e-02f,-3.9095506e-02f,1.1901328e-01f,-3.5262704e-03f,-1.2707888e-01f,-9.7670764e-02f,2.409522e-01f,-1.7808616e-02f,1.2244418e-01f,1.3857576e-01f,2.780246e-01f,5.4126382e-02f,2.7717692e-01f,2.159248e-01f,4.756105e-02f,1.916871e-01f,-3.984797e-02f,-8.7658584e-02f,-2.7706453e-01f,-1.7992854e-02f,1.0803354e-01f,-1.4399791e-01f,2.0149818e-01f,9.5697254e-02f,-3.9363578e-02f,-4.8055142e-02f,-1.11248255e-01f,-2.3233649e-01f,1.025852e-01f,-3.4688115e-03f,1.4379486e-01f,8.056426e-02f,4.0612012e-02f,-1.2195331e-01f,-1.836008e-01f,-7.7904776e-02f,-1.7946817e-01f,-1.9874834e-01f,-6.844814e-02f,5.144307e-02f,-1.2675354e-01f,-2.289948e-01f,-2.0275348e-01f,-1.6462561e-01f,1.5317759e-01f,1.7683601e-01f,-6.4738035e-02f,2.8207284e-01f,-2.2201955e-01f,-2.64505e-01f,1.3593307e-01f,2.7101094e-01f,-1.860971e-01f,1.2526491e-01f,-4.225135e-03f,-2.6850876e-01f,-2.004934e-01f,2.635643e-01f,-3.527224e-03f,-1.3259847e-01f,1.2851611e-01f,-1.0133986e-01f,-1.2488362e-01f,-8.9647084e-02f,1.331566e-01f,-8.337289e-02f,-1.5982634e-01f,1.6642678e-01f,1.9302273e-01f,5.250898e-02f,2.1393448e-02f,3.6245167e-02f,-6.8553716e-02f,2.2336006e-02f,1.8107086e-02f,1.8224382e-01f,1.1719793e-01f,1.3236943e-01f,-2.5443512e-01f,1.1529657e-01f,4.2985797e-02f,1.9103837e-01f,-2.4054614e-01f,-1.5776418e-01f,-1.5853682e-01f,7.071245e-02f,-1.0044202e-01f,-2.5044215e-01f,7.5129926e-02f,-6.443012e-02f,-1.992312e-01f,-1.14759386e-01f,-7.24953e-02f,2.254312e-01f,-8.7069646e-02f,-2.7210861e-02f,-2.507519e-01f,1.2915811e-01f,1.6386631e-01f,-1.835306e-01f,2.4110043e-01f,-2.098821e-01f,-8.468698e-02f,2.1599668e-01f,1.4653552e-01f,1.3099256e-01f,-2.3369463e-01f,1.3181427e-01f,7.5111836e-02f,1.6096237e-01f,-1.1935592e-02f,-1.2242682e-01f,2.6776165e-02f,1.5417549e-01f,1.14899665e-01f,2.8070712e-01f,-1.5350099e-01f,1.6107428e-01f,2.7872163e-01f,1.017009e-01f,-4.7934562e-02f,-1.8470648e-01f,2.590379e-01f,1.3635829e-01f,1.5972343e-01f,3.9356083e-02f,-1.6671054e-01f,7.9595804e-02f,1.99664e-01f,1.3562122e-01f,-7.386045e-02f,2.4597955e-01f,-4.2116597e-02f,-4.0968522e-02f,-1.4024486e-01f,-6.201832e-02f,-1.8805532e-01f,2.1840501e-01f,1.7124414e-01f,-1.3246922e-01f,2.1073258e-01f,-2.592437e-01f };
static float b_4  [4] = { 0.e+00f,0.e+00f,0.e+00f,0.e+00f };
static float w_3  [3][3][4][8] = { -4.2417437e-02f,-5.720052e-02f,-1.1796105e-01f,8.476637e-02f,-1.4738393e-01f,1.355296e-01f,6.338109e-02f,-1.861487e-01f,-2.1453685e-01f,-1.4385247e-01f,1.716505e-02f,1.632201e-01f,-1.2708545e-01f,-1.9298553e-01f,1.4826627e-01f,7.319312e-02f,8.69865e-02f,-4.2529404e-04f,-1.6373326e-01f,1.7111374e-01f,-1.7507234e-01f,1.02962e-01f,1.8907999e-01f,-1.1550636e-01f,1.5497588e-01f,-1.0003336e-01f,1.5137239e-01f,5.356206e-02f,1.0311931e-02f,1.2054981e-01f,-1.1424628e-01f,-6.510918e-02f,-7.316844e-02f,-1.1839022e-01f,7.3480457e-03f,3.3746466e-02f,-1.02208465e-01f,1.6058816e-01f,9.7334385e-05f,-1.0503961e-01f,3.4823522e-02f,7.798503e-02f,1.776859e-01f,-7.640144e-02f,5.401148e-02f,1.7331655e-01f,1.0090466e-01f,-1.7818958e-02f,2.2894101e-01f,2.9274866e-02f,1.03689745e-01f,2.1828033e-01f,-1.9642118e-01f,1.416726e-01f,-9.360537e-03f,1.8067111e-01f,-1.5556543e-01f,-1.1079715e-01f,-1.9670811e-01f,7.932128e-02f,-4.5079768e-02f,-8.966765e-02f,-6.7144826e-02f,-2.3406309e-01f,-4.3016225e-03f,7.7151254e-02f,-2.1098493e-01f,6.6516206e-02f,-3.792581e-02f,-1.5717779e-01f,3.02732e-03f,-1.818006e-01f,1.4999323e-01f,1.6620637e-01f,9.586339e-02f,1.985978e-01f,-1.4424242e-01f,-1.2803751e-01f,-2.6121825e-02f,-6.844783e-02f,-1.1262436e-01f,9.054075e-02f,6.256558e-02f,-1.2788966e-01f,7.302891e-02f,2.0259602e-01f,-8.4130466e-04f,2.2078262e-01f,-2.2283837e-01f,-1.040799e-01f,2.158536e-02f,-1.8130478e-01f,1.4304398e-01f,-2.0199125e-01f,-3.6371157e-02f,1.1986449e-02f,-1.496489e-01f,2.3206122e-01f,-1.6779453e-01f,8.279951e-02f,-1.6931057e-01f,-8.269858e-02f,-4.1527286e-02f,-1.4547946e-01f,2.5768742e-02f,-1.5706861e-01f,3.0295953e-02f,8.915514e-04f,1.7899384e-01f,-3.4523606e-02f,-4.0124357e-03f,2.0075347e-01f,-2.0804465e-02f,1.1778571e-01f,6.2281534e-02f,5.125791e-03f,1.5241952e-01f,2.1936317e-01f,-3.0674487e-03f,-8.357422e-02f,1.0671319e-01f,-1.7041972e-01f,1.00721315e-01f,1.1626512e-02f,-2.0804712e-01f,9.784256e-02f,-4.600385e-02f,9.138107e-03f,8.5541755e-03f,3.889282e-02f,-1.8047899e-01f,-1.0534346e-02f,1.8087788e-01f,-7.022013e-02f,-1.5337609e-01f,1.0864608e-01f,4.7973946e-02f,-4.5809135e-02f,9.56652e-02f,2.3511682e-01f,1.588961e-01f,1.230222e-02f,2.3049466e-01f,2.0306255e-01f,3.3767834e-02f,-1.280193e-01f,3.9295033e-02f,9.826915e-02f,1.3558237e-01f,7.360695e-02f,1.1333726e-01f,-3.826265e-02f,1.13493755e-01f,1.5820144e-01f,9.4898045e-04f,-2.3186791e-01f,-2.3041491e-01f,-1.8569642e-01f,1.2949432e-01f,1.8066223e-01f,1.4215766e-01f,-2.769138e-02f,-8.611725e-02f,2.08814e-02f,-1.7029485e-01f,-2.0979196e-01f,-2.0542155e-01f,1.10484466e-01f,1.1768766e-01f,-1.2626612e-01f,4.6522096e-02f,-2.103905e-01f,-1.8263201e-01f,-1.9170573e-01f,1.7445132e-02f,-1.3879248e-01f,1.4543535e-01f,-1.86456e-02f,-2.1337083e-01f,7.898004e-02f,-1.4799692e-01f,3.176771e-02f,-6.532627e-02f,1.8819551e-01f,1.3256235e-01f,-1.5220672e-01f,5.1358297e-02f,-1.7381422e-01f,-2.01789e-02f,-1.8096794e-01f,2.2000347e-01f,-1.9128853e-01f,1.8912174e-01f,-7.191141e-02f,-3.9514303e-02f,-6.4352676e-02f,-1.4844744e-01f,-1.6540357e-01f,-1.8163538e-01f,1.4957933e-01f,1.2505971e-01f,1.0897501e-01f,-9.781867e-02f,2.3115794e-01f,-2.0812193e-01f,2.0641908e-02f,1.05052486e-01f,1.1841269e-01f,1.6542609e-01f,2.0706697e-01f,8.5391924e-02f,8.099337e-02f,-6.784907e-02f,-1.894348e-01f,2.291127e-01f,1.06554314e-01f,-2.3053433e-01f,-2.1308373e-01f,1.5811177e-01f,1.0054849e-01f,4.0117055e-03f,4.2900607e-02f,2.2706448e-01f,-1.0533975e-01f,1.14871696e-01f,-1.807956e-01f,4.2347297e-02f,-1.777305e-01f,1.3259493e-01f,-2.2588435e-01f,1.0226308e-01f,-2.3302937e-01f,-3.682758e-02f,1.9907121e-01f,-8.898032e-02f,-1.3298972e-01f,1.7465703e-01f,5.839093e-02f,1.8424849e-01f,-8.82954e-02f,-1.6717306e-01f,3.816904e-02f,1.751335e-02f,1.5932469e-01f,1.8829902e-01f,-2.0424366e-02f,1.0949357e-01f,1.3024803e-01f,-1.3426721e-01f,-1.9577941e-01f,-1.2369527e-01f,-6.482686e-02f,9.8381534e-02f,-6.419179e-02f,-7.62831e-02f,-1.7245625e-01f,-5.9038132e-02f,-7.454024e-02f,-1.4295065e-01f,2.9177025e-02f,4.906501e-02f,2.05409e-01f,2.2998889e-01f,1.5129392e-01f,-1.6591984e-01f,1.741098e-01f,1.16289124e-01f,1.9213109e-01f,-1.0788424e-01f,-2.5097102e-02f,-9.701811e-02f,-1.1293389e-01f,1.645359e-01f,2.4689242e-02f,6.2718555e-02f,-2.3070662e-01f,-3.0004114e-02f,-2.0875345e-01f,3.602712e-02f,-1.32312e-01f,1.7676125e-01f,-1.9110388e-01f,-1.0728778e-01f,1.5980853e-01f,-2.3407966e-01f,-1.7181978e-02f,1.9267718e-01f,1.1070858e-01f };
static float b_3  [8] = { 0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f };
static float w_2  [3][3][8][8] = { -4.253587e-02f,-4.0220246e-02f,-1.7289454e-01f,-4.4977516e-03f,-1.00902826e-01f,-1.7318144e-01f,1.428509e-01f,-1.2711127e-01f,-1.8060336e-01f,6.5469354e-02f,-1.8929186e-01f,-9.828377e-02f,-4.165408e-02f,-7.892005e-02f,4.6114266e-02f,1.8668312e-01f,-1.7311102e-01f,-1.7038693e-01f,-1.3307288e-01f,1.5046525e-01f,1.2615025e-02f,8.2375646e-02f,-8.764604e-02f,1.6994074e-02f,-1.8945865e-01f,-3.410773e-02f,-9.423185e-02f,-1.12068355e-02f,8.110017e-03f,7.703915e-02f,-1.8735881e-01f,-1.2846008e-01f,1.4065874e-01f,3.904076e-02f,6.667709e-02f,-5.530989e-02f,-9.797483e-02f,-1.3767001e-01f,-6.0506016e-02f,4.1331947e-02f,-9.340748e-02f,1.6584927e-01f,-1.0723114e-01f,-7.515414e-02f,1.5322268e-01f,-5.3486913e-02f,-1.2554467e-01f,-1.2415192e-01f,-1.2197568e-01f,-7.257509e-02f,4.781741e-02f,8.7607294e-02f,8.819839e-02f,8.473697e-02f,-4.72371e-02f,-4.9364924e-02f,-3.308338e-02f,1.4975092e-01f,-1.1212777e-01f,-8.778834e-02f,-1.2455148e-01f,5.6711137e-03f,-1.4912486e-03f,1.24822855e-01f,1.8511477e-01f,1.8312308e-01f,2.6601478e-02f,-9.608805e-03f,-1.2416078e-01f,-8.799839e-02f,2.74311e-03f,-2.5901496e-02f,-1.1985565e-01f,-4.898177e-02f,-1.9933917e-01f,-6.047818e-02f,2.184394e-02f,1.9639313e-02f,-1.3180807e-01f,4.950586e-02f,3.8508043e-02f,5.814752e-02f,-1.20674185e-01f,-1.186089e-01f,-4.4994876e-02f,-7.966217e-02f,-1.5800849e-01f,5.80239e-02f,-3.0596644e-02f,-1.1249234e-01f,-4.0521204e-02f,-1.6058427e-01f,1.6803294e-01f,-1.3988212e-01f,-6.980954e-02f,1.7086977e-01f,-3.691323e-02f,1.8170857e-01f,1.8604735e-01f,-1.1513987e-01f,-1.7919254e-01f,-3.6164686e-02f,-8.244402e-02f,7.7207655e-02f,-1.7349003e-01f,-1.6804206e-01f,-1.6434853e-01f,-1.1707579e-01f,-1.1849482e-01f,-1.8040805e-01f,7.930267e-02f,-1.9213665e-01f,-1.599191e-01f,-1.9601326e-01f,7.936418e-03f,1.762782e-02f,4.2997092e-02f,2.0067221e-01f,-1.6380993e-01f,-3.8332805e-02f,-6.589311e-02f,1.2678769e-01f,-5.984181e-02f,-1.275377e-02f,4.7202557e-03f,2.1803305e-02f,-1.9786358e-01f,1.4394414e-01f,-8.9514025e-02f,-1.2981997e-01f,5.7929337e-02f,-1.1852607e-01f,1.8412167e-01f,-1.3112533e-01f,1.7877197e-01f,-1.5380725e-01f,1.2760922e-01f,-8.6520806e-02f,-9.308613e-02f,1.2125936e-01f,1.4548019e-01f,-1.647235e-01f,-3.149849e-02f,9.612024e-02f,8.1680745e-02f,1.5175936e-01f,-1.505507e-01f,-4.0681854e-02f,-3.469047e-02f,-1.2526625e-01f,-4.8173606e-02f,7.524428e-02f,-1.056392e-01f,-9.302154e-03f,-1.769139e-01f,-1.4634067e-01f,2.0022959e-02f,1.3701704e-01f,-1.6977412e-01f,1.396333e-01f,9.425706e-02f,1.4763477e-01f,-4.1739196e-02f,7.830319e-02f,1.1867961e-01f,1.0076743e-01f,-1.851524e-01f,1.656878e-01f,-1.2620689e-01f,-1.5857366e-01f,-1.1034374e-01f,5.2763253e-02f,-1.68559e-01f,-3.693454e-02f,-8.681992e-02f,-1.3836361e-01f,8.3919644e-02f,5.442694e-02f,5.9072435e-02f,1.271472e-01f,1.706247e-01f,-1.8864006e-02f,9.0298146e-02f,-9.745448e-02f,-9.544224e-02f,-3.804922e-02f,-5.503711e-02f,8.062878e-02f,1.8746579e-01f,6.523532e-02f,1.2036228e-01f,1.7048222e-01f,1.2240586e-01f,1.8945384e-01f,1.6198763e-01f,-4.7389433e-02f,1.9453925e-01f,-1.6458748e-01f,1.9377097e-02f,1.4758906e-01f,1.4331406e-01f,2.0206267e-01f,1.2165749e-01f,6.347829e-02f,1.451276e-01f,-4.307443e-02f,2.843821e-02f,-1.4738005e-01f,-1.3477647e-01f,-2.602394e-02f,-3.5974354e-02f,-6.87775e-02f,-1.9271958e-01f,-5.869779e-02f,-9.573955e-02f,-1.701856e-01f,-9.879774e-02f,-1.1539294e-01f,1.8601555e-01f,1.066505e-01f,-5.9421957e-03f,1.8691605e-01f,-1.6360387e-01f,2.0080253e-01f,9.126505e-02f,1.401087e-01f,1.1114672e-02f,9.4965905e-02f,-1.6083774e-01f,-2.3306772e-02f,1.9324738e-01f,9.446573e-02f,-1.5140495e-01f,-5.7806358e-02f,1.786688e-01f,9.043366e-04f,-1.0091232e-01f,8.633864e-02f,9.567031e-02f,-9.572804e-04f,7.7496916e-02f,6.65324e-02f,-8.644372e-02f,7.7772915e-02f,-1.2625667e-01f,-8.201356e-02f,7.409969e-02f,1.1360681e-01f,-1.0457675e-01f,1.0459027e-01f,-1.01713866e-01f,1.3488895e-01f,2.3131967e-03f,1.1913499e-01f,-1.4124036e-02f,1.4620551e-01f,-3.9631873e-02f,-1.8260396e-01f,1.8589175e-01f,8.6933225e-02f,-1.3110673e-01f,-1.04983896e-01f,-2.5757343e-02f,-4.828252e-02f,1.8073836e-01f,-4.6270818e-02f,-1.7726985e-01f,2.0234275e-01f,-1.9536284e-01f,-1.1816759e-01f,2.3178428e-02f,7.8359365e-02f,1.5474623e-01f,-1.3554975e-02f,2.0387626e-01f,1.1609423e-01f,-6.3521564e-03f,1.430279e-02f,-1.227838e-01f,1.6511181e-01f,-1.2678793e-01f,8.6162e-02f,1.3768315e-01f,-1.03899695e-01f,-8.0754064e-02f,1.4382362e-02f,4.3664172e-02f,-1.5122473e-01f,-6.155999e-02f,7.4661285e-02f,-1.9860224e-01f,1.1078003e-01f,-9.898622e-02f,-2.0303509e-01f,-1.7632586e-01f,1.8242496e-01f,1.16131306e-01f,2.922915e-02f,-1.6244888e-02f,-1.0936753e-01f,-2.0144644e-01f,4.1654855e-02f,-1.874708e-01f,-1.6061206e-01f,7.7875465e-02f,-1.7029689e-01f,-2.9598966e-02f,8.917466e-02f,-1.6488926e-01f,1.9420105e-01f,-2.9194847e-02f,1.8994355e-01f,8.719826e-02f,1.4802694e-02f,-4.5919687e-02f,1.4995858e-01f,-1.22740045e-01f,-1.203487e-01f,4.102467e-02f,-1.7958996e-01f,8.248821e-02f,-1.5753801e-01f,1.219745e-02f,-1.20408654e-01f,-4.4947326e-02f,1.651122e-01f,5.3159148e-02f,-9.576861e-02f,-6.8008855e-02f,1.4258051e-01f,6.743464e-02f,1.1283058e-01f,4.5094445e-02f,-4.0879846e-03f,-4.671842e-03f,-1.5908022e-01f,9.398702e-02f,-4.2698666e-02f,1.1985925e-01f,-1.1600736e-01f,-1.3913989e-01f,9.883472e-02f,-4.916638e-03f,1.5976423e-01f,8.16856e-02f,-1.4076105e-01f,1.4735031e-01f,-1.3364817e-01f,9.0850234e-02f,-1.842449e-01f,1.5564439e-01f,-7.4493244e-02f,1.038214e-01f,5.6400865e-02f,-1.810176e-01f,-1.6462913e-01f,-3.463678e-02f,-9.815066e-02f,2.3714066e-02f,-5.9899524e-02f,1.57338e-01f,1.0982776e-01f,1.5111333e-01f,3.0631736e-02f,7.745686e-02f,-1.6357982e-01f,-8.1730664e-02f,7.6518e-02f,5.3058743e-02f,-1.9692451e-01f,-4.044655e-02f,1.8113524e-01f,1.4817983e-01f,-1.8200023e-01f,-8.0720626e-02f,1.3373062e-01f,1.1501372e-01f,-1.3592952e-01f,-5.6552544e-02f,1.9513643e-01f,-1.5457341e-01f,6.669012e-02f,-1.3208741e-01f,-1.606146e-01f,5.7677984e-02f,1.264112e-01f,-5.7947204e-02f,-8.8554844e-02f,-4.262133e-02f,-1.741907e-01f,-3.5708085e-02f,-8.699536e-02f,6.889373e-02f,1.5617636e-01f,1.5059456e-01f,-1.6651396e-01f,-2.020757e-01f,1.1901051e-01f,9.412897e-02f,-8.408366e-02f,-1.6542168e-01f,-3.344235e-02f,-1.1992544e-01f,-1.8304321e-01f,-1.3066351e-02f,1.4359906e-01f,-1.3310656e-01f,1.5459225e-01f,4.0911272e-02f,-4.9752608e-02f,-1.627759e-01f,6.987208e-02f,-1.9536424e-01f,1.8028751e-01f,6.186211e-02f,-1.6216215e-01f,-1.502827e-01f,1.3425589e-01f,1.3341147e-01f,1.2142715e-01f,1.9053575e-01f,1.0986635e-01f,-6.156954e-02f,1.6638494e-01f,-1.8138231e-01f,-1.9033954e-02f,-1.3163298e-02f,1.4853802e-01f,1.5670446e-01f,7.1620345e-02f,1.0472113e-01f,-2.5129348e-02f,2.0221251e-01f,1.9788831e-02f,-1.17378116e-01f,2.0136249e-01f,-8.679597e-02f,1.5835708e-01f,-1.7112988e-01f,-1.3579595e-01f,-1.9979182e-01f,-1.5629584e-01f,1.5728831e-01f,1.5807182e-02f,-6.130001e-02f,1.3208342e-01f,1.8025154e-01f,-1.22278005e-01f,-1.5087453e-01f,1.4963636e-01f,1.8853661e-01f,1.70959e-01f,6.2589645e-02f,-1.15464285e-01f,1.8061927e-01f,-1.264242e-01f,1.6863146e-01f,1.2261987e-02f,6.0581744e-02f,-1.1554011e-01f,-1.4583477e-01f,-3.0703083e-02f,-1.7480487e-01f,-1.49899e-01f,-6.792158e-03f,9.104237e-03f,-5.899845e-02f,1.4940864e-01f,6.1937064e-02f,8.848408e-02f,1.620726e-01f,-1.8456684e-01f,1.13809496e-01f,1.1674899e-01f,7.600212e-02f,1.0185689e-01f,1.7428991e-01f,1.13453805e-01f,-9.800447e-02f,-4.4586986e-02f,1.7850813e-01f,7.6369315e-02f,7.785055e-02f,-1.1513559e-01f,-5.5246383e-02f,-8.485436e-03f,7.478565e-02f,-1.945115e-01f,1.1562717e-01f,4.9348265e-02f,-1.1178759e-01f,9.1483325e-02f,5.587554e-02f,-3.656964e-02f,-1.4384079e-01f,-1.2379525e-01f,1.0297063e-01f,-1.0444613e-01f,-1.2607792e-01f,-2.0224634e-01f,-1.95136e-01f,-1.963823e-01f,-1.9592275e-01f,-1.3175789e-01f,-1.4000276e-01f,-7.057409e-02f,8.03307e-02f,1.8992096e-02f,-1.177309e-01f,1.7697936e-01f,1.9797546e-01f,7.314244e-02f,6.792158e-03f,-2.0292684e-01f,4.4015303e-02f,7.9926044e-02f,-6.0269684e-03f,1.7826042e-01f,-1.7077884e-01f,-1.4706138e-01f,6.1158597e-02f,1.2643844e-01f,5.369824e-02f,-1.6157426e-01f,1.9027868e-01f,-1.947289e-01f,1.9506001e-01f,-1.6206843e-01f,-1.1391857e-01f,-5.655308e-02f,4.441957e-02f,-1.6047293e-01f,1.1110684e-01f,4.8843324e-02f,-1.6819696e-01f,1.2646854e-02f,-9.453777e-02f,1.5260115e-01f,-1.3376965e-01f,-1.102376e-01f,1.1140463e-01f,-4.073359e-02f,-1.1352977e-01f,1.0477784e-01f,-1.0085475e-01f,8.52724e-02f,9.115332e-02f,4.4851005e-02f,-8.7469086e-02f,-6.3511536e-02f,-1.8332694e-01f,-1.8505365e-01f,-1.2506813e-01f,-2.539979e-02f,1.9245952e-02f,1.16022825e-01f,8.497134e-02f,-1.8981922e-01f,1.687819e-01f,1.2374073e-02f,1.7193168e-02f,-7.410985e-02f,1.5564746e-01f,-3.0161709e-02f,1.2623698e-01f,1.989418e-01f,1.8010694e-01f,2.0316303e-01f,-1.2787637e-01f,-1.15126e-01f,-1.31329e-01f,-1.405473e-01f,4.8395723e-02f,-1.896157e-01f,-1.3888353e-01f,1.1968285e-01f };
static float b_2  [8] = { 0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f };
static float w_1  [1][1][8][8] = { 3.9445156e-01f,-2.000063e-01f,-1.4031991e-01f,-4.1614002e-01f,-5.8258474e-02f,-3.5023254e-01f,-3.1154636e-01f,1.1350322e-01f,1.17002845e-01f,-2.5177166e-01f,-4.9433812e-01f,-3.9925426e-01f,5.4512978e-02f,-2.742472e-01f,-2.0518291e-01f,-3.2608047e-01f,5.9031314e-01f,-5.3184676e-01f,-3.5138828e-01f,-6.64112e-02f,-3.4303847e-01f,-6.0961175e-01f,-4.6499366e-01f,5.083081e-01f,3.5877317e-01f,-3.4124851e-03f,-8.8302255e-02f,4.3758196e-01f,5.289888e-01f,-3.264129e-02f,2.4627107e-01f,5.287691e-01f,2.2574598e-01f,-5.7743925e-01f,-3.2332352e-01f,3.590272e-01f,-5.744883e-01f,-1.5309998e-01f,2.8990686e-02f,-2.061579e-01f,-4.9941647e-01f,4.097739e-01f,1.231724e-02f,-5.157058e-01f,-1.4468098e-01f,-1.5218076e-01f,-2.354616e-02f,-2.061373e-02f,3.581484e-01f,4.4742233e-01f,-4.1760218e-01f,-2.6410568e-01f,5.3103036e-01f,5.092184e-01f,2.5647318e-01f,5.9712034e-01f,4.369232e-01f,-1.3022423e-02f,-2.2771814e-01f,3.152615e-02f,4.9639946e-01f,5.6523794e-01f,-4.883166e-01f,3.778864e-01f };
static float b_1  [8] = { 0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f,0.e+00f };
static float w_0  [128][4] = { 6.959197e-02f,9.106016e-02f,2.8893918e-02f,-9.898679e-02f,1.8162724e-01f,-1.1372153e-01f,-5.6161553e-02f,-2.13111e-01f,-1.2331268e-01f,1.5492928e-01f,-5.6742042e-02f,1.844545e-01f,1.1933944e-01f,-7.384303e-02f,-1.8236104e-01f,1.1272803e-01f,8.993283e-02f,1.452645e-01f,1.8833348e-01f,-1.8758178e-03f,1.3022357e-01f,-5.026683e-03f,-1.7882925e-01f,1.0937512e-01f,1.08435154e-01f,2.1161178e-01f,1.9602269e-01f,-6.836511e-02f,-3.813556e-02f,9.775776e-02f,4.5969933e-02f,8.715984e-02f,1.779924e-01f,1.3039085e-01f,-8.455871e-02f,-1.9696921e-02f,-1.01637624e-01f,1.5643889e-01f,-1.0903486e-01f,-2.2046015e-02f,1.2986869e-02f,1.3457826e-01f,9.2318565e-02f,-8.647174e-03f,-1.9868842e-01f,-2.1103044e-01f,-1.9282338e-01f,-8.8645115e-02f,4.8047453e-02f,4.4114113e-02f,-1.7378132e-01f,3.5056517e-02f,-1.4718562e-01f,-2.0028761e-01f,2.1874204e-02f,-1.4761686e-02f,1.2709624e-01f,-1.4340658e-01f,-1.6866285e-01f,1.7745692e-01f,-7.395175e-02f,9.995192e-02f,-1.038012e-01f,-6.0863495e-03f,1.23723954e-01f,-6.4804e-02f,-1.0517053e-01f,-8.538781e-02f,1.0873613e-01f,-4.9543053e-02f,4.909253e-02f,1.1576566e-01f,3.8318515e-02f,-1.0400035e-01f,7.047948e-02f,2.4920925e-02f,-1.1168397e-01f,1.8694392e-01f,1.98277e-01f,-2.0549332e-01f,-7.891241e-02f,-1.9300246e-01f,-1.5210804e-01f,1.137968e-01f,1.7120776e-01f,1.5083519e-01f,3.5742685e-02f,3.5186097e-02f,1.7402217e-01f,1.0771072e-01f,1.6451171e-01f,2.177298e-03f,-1.4685974e-01f,-1.7007366e-01f,6.2525034e-02f,-2.7453527e-02f,7.5074315e-02f,2.6900172e-02f,-1.4929622e-02f,-1.1901908e-01f,-3.5433635e-02f,-1.8147907e-01f,6.1603278e-02f,-1.9188195e-02f,-1.1059558e-01f,-9.582245e-02f,1.5471894e-01f,-5.2256003e-02f,1.7920813e-01f,1.796469e-02f,-8.855987e-02f,-1.8164061e-01f,8.050388e-02f,-7.156269e-02f,-2.4362355e-03f,-1.8147591e-01f,3.327027e-02f,2.7226105e-02f,-1.9402812e-01f,-1.428813e-01f,1.5289405e-01f,6.8597406e-02f,6.854871e-02f,-8.3202034e-02f,-6.188406e-02f,-2.0885497e-02f,4.6934098e-02f,-1.2522545e-01f,1.3531628e-01f,5.2184284e-02f,1.5120834e-01f,-1.09453656e-01f,6.210524e-02f,-7.7212006e-02f,-4.545462e-02f,4.155004e-02f,6.9722295e-02f,1.2749344e-01f,-1.6946648e-01f,3.1855345e-02f,-1.6263387e-01f,-6.219159e-02f,-1.8539712e-01f,-1.1877408e-01f,1.6687277e-01f,1.7224592e-01f,2.775222e-03f,-1.1903855e-01f,1.7608234e-01f,1.9141436e-02f,2.0240888e-01f,-1.6798705e-01f,1.00805074e-01f,-8.216436e-02f,1.02598906e-01f,-1.3257007e-01f,-1.9123435e-02f,-1.5468751e-01f,-6.0425118e-02f,-8.0588356e-02f,-1.5856117e-02f,1.05282456e-01f,-2.2432342e-02f,1.2742263e-01f,-2.1143216e-01f,2.9955178e-02f,-7.63984e-02f,8.863205e-02f,1.3064787e-02f,1.9829854e-01f,-1.2939411e-01f,-3.3735022e-02f,5.245924e-02f,2.0694867e-01f,-2.7282327e-02f,-6.550394e-02f,-7.1597546e-03f,1.0033995e-01f,8.329228e-03f,-1.18330784e-01f,-3.2247856e-02f,-9.1961384e-02f,1.3123855e-02f,-3.3832923e-02f,7.582e-02f,-1.908187e-01f,-1.6878997e-01f,-1.4357036e-01f,1.6617289e-01f,3.5579264e-02f,-1.8289287e-01f,3.368616e-03f,-2.0396441e-01f,7.9236895e-02f,-1.8153411e-01f,1.5473148e-01f,-1.8089487e-01f,7.913846e-02f,3.1497896e-02f,-1.9936949e-01f,2.119143e-01f,-5.9453726e-02f,-5.8975667e-02f,6.82323e-02f,-7.4058905e-02f,-2.329214e-02f,-2.9743314e-03f,-1.8939726e-01f,2.0211601e-01f,-7.246505e-02f,2.1036965e-01f,1.0288593e-01f,2.0180851e-01f,1.9153929e-01f,-1.0272013e-01f,-2.0957972e-01f,-1.0702897e-01f,6.3844144e-02f,-3.940858e-02f,-1.066454e-01f,-9.579963e-02f,1.20544225e-01f,-1.413519e-01f,-2.6024565e-02f,-2.4858251e-02f,-1.5578538e-02f,1.763311e-01f,-1.1272458e-01f,-5.8035553e-02f,-4.0515825e-02f,-1.9341156e-02f,-7.5754523e-03f,-2.856347e-02f,9.009585e-02f,8.091906e-02f,9.648383e-03f,1.7964828e-01f,2.538465e-02f,1.946382e-01f,7.322252e-02f,-5.8820695e-03f,1.749965e-01f,1.4804989e-01f,1.6427562e-02f,3.8789213e-04f,-4.4004157e-02f,3.264317e-02f,5.717817e-02f,-1.00759014e-01f,-1.5102097e-01f,9.007874e-02f,-5.7019383e-02f,-8.284937e-02f,-9.696107e-02f,-3.3662528e-02f,-1.6944462e-01f,-8.835762e-02f,5.3976834e-02f,1.2694335e-01f,-2.0947643e-01f,1.0869399e-01f,-1.4742294e-01f,5.129972e-02f,-2.0602562e-01f,1.1813274e-01f,-1.2505278e-01f,5.7821184e-02f,-1.13717765e-01f,-1.2993789e-01f,-5.6922585e-03f,-9.279079e-02f,-5.741109e-02f,1.7378372e-01f,-1.895523e-01f,1.6338974e-01f,2.023217e-02f,6.722829e-02f,9.202623e-02f,9.523231e-02f,1.2335864e-01f,1.922516e-01f,3.747797e-02f,1.3911173e-02f,1.107052e-03f,-4.408315e-02f,-1.2823734e-01f,5.254507e-03f,-1.5461013e-02f,2.1252173e-01f,-1.2030684e-01f,2.0473349e-01f,1.0764033e-02f,-6.749627e-02f,-1.7477892e-01f,-1.8102378e-01f,-3.9632335e-02f,1.2679657e-01f,-1.3861348e-01f,6.355235e-02f,-1.9714859e-01f,-1.1754036e-01f,-4.3368056e-02f,7.9480946e-02f,-1.1943585e-01f,1.4759332e-01f,-1.2906519e-01f,2.0057672e-01f,1.5950376e-01f,-1.8752266e-01f,1.8171751e-01f,-9.515569e-03f,8.536017e-02f,-9.7828045e-02f,2.1024635e-01f,1.0576251e-01f,1.0915482e-01f,-1.3000956e-01f,-1.6530037e-01f,-3.8119704e-02f,-9.782657e-02f,-2.114625e-01f,-4.8884436e-02f,1.995534e-01f,1.16954386e-01f,1.457775e-01f,-1.6649988e-01f,-1.7936587e-01f,7.148558e-02f,2.0702705e-01f,-1.4386086e-01f,1.5497103e-02f,-1.3058513e-01f,-8.8228606e-02f,-4.3027937e-02f,1.4729023e-01f,1.4666104e-01f,-3.9480343e-02f,2.0844832e-01f,8.358437e-02f,-1.827932e-01f,-2.0540598e-01f,-3.5131872e-03f,-2.0477456e-01f,1.0808432e-01f,-1.2907708e-01f,-6.1798245e-03f,8.509514e-02f,-8.474231e-02f,6.7554355e-02f,-1.3553515e-01f,7.82544e-02f,1.3090032e-01f,-4.203786e-02f,1.735881e-03f,1.5783772e-01f,-2.0001805e-01f,1.9459963e-01f,1.6485214e-02f,-2.093005e-01f,7.139546e-02f,-1.4163664e-01f,9.65752e-02f,-5.2455172e-02f,-4.3534786e-02f,7.2552055e-02f,-1.836618e-01f,9.383893e-02f,5.3101927e-02f,8.20016e-02f,-1.9142345e-01f,2.3354009e-02f,1.4607304e-01f,-1.2260928e-01f,1.4589104e-01f,-2.0965277e-01f,-2.6173547e-02f,-2.0253216e-01f,-1.7801447e-01f,2.0504284e-01f,8.513844e-02f,8.738744e-02f,1.4786229e-02f,1.343886e-01f,1.8198523e-01f,1.4397052e-01f,-1.7335576e-01f,1.47591e-01f,6.8892986e-02f,-1.9018607e-01f,-3.458567e-02f,-1.9743416e-01f,-1.4333963e-01f,9.521049e-02f,5.7396352e-02f,-1.597938e-01f,1.5289736e-01f,-1.3593717e-01f,-1.492972e-01f,-1.3979813e-01f,3.7489384e-03f,-1.856471e-01f,-6.3694865e-03f,-7.7506825e-02f,-1.4982171e-01f,1.1204913e-01f,-2.0206405e-01f,1.7878166e-01f,-2.6248023e-02f,1.5580672e-01f,6.37207e-02f,4.8389137e-02f,4.7293723e-02f,-1.116005e-01f,-6.6688046e-02f,-4.19275e-02f,1.06137484e-01f,1.7561567e-01f,-2.0407559e-01f,-1.5112771e-01f,3.733039e-02f,-1.7952523e-01f,-1.17384516e-01f,7.5523704e-03f,-8.2928106e-02f,1.8919751e-01f,1.2554595e-01f,1.9951844e-01f,6.06969e-02f,1.6468582e-01f,1.2924054e-01f,-1.896134e-01f,1.1800033e-01f,-2.071375e-01f,6.813952e-02f,-1.1792749e-01f,-1.3657749e-01f,-1.755599e-01f,1.6187304e-01f,-2.4107724e-02f,-6.328075e-02f,-1.6488183e-01f,9.6384585e-02f,1.2663633e-02f,-1.8419206e-01f,-1.2238913e-01f,9.319821e-02f,5.323851e-02f,3.6350533e-02f,1.4195296e-01f,-6.632298e-02f,1.2493804e-01f,7.9037696e-02f,1.7045179e-01f,2.0085448e-01f,1.8293944e-01f,1.9712728e-01f,5.641541e-02f,-1.11984275e-01f,1.0366574e-02f,-1.0188248e-01f,1.5553665e-01f,1.6227832e-01f,-1.8250479e-01f,-1.7161068e-01f,-1.2876573e-01f,-2.1539077e-02f,4.1419715e-02f,-2.382195e-02f,-1.7845935e-01f,1.6146868e-01f,6.463498e-02f,2.1150738e-02f,1.4057353e-01f,2.1276733e-01f,-1.3317823e-02f,-8.902737e-02f,1.9891155e-01f,-1.7662786e-01f,-1.8744002e-01f,9.2948735e-02f,1.2609726e-01f,1.7777693e-01f,5.3177327e-02f,-2.0119123e-01f,-8.455104e-02f,-4.1744873e-02f,1.6295415e-01f,-4.593599e-02f,1.6417548e-02f,-2.762188e-02f,1.7658877e-01f,4.4409722e-02f,-2.4537861e-02f,-1.5049319e-01f,-1.5137693e-01f,-4.5945644e-02f,1.5646487e-01f,-1.4793679e-01f,-6.282759e-02f,1.416151e-01f,-2.2322282e-02f,1.0888678e-01f,2.0927647e-01f,-2.0230336e-01f,6.9643557e-04f,-1.7365658e-01f,7.4846625e-02f,-6.223774e-02f,-1.6974387e-01f,-1.4518556e-01f,1.4326686e-01f,-1.9485757e-02f };
static float b_0  [4] = { 0.e+00f,0.e+00f,0.e+00f,0.e+00f };

void dummy_model2::cnn(float x_0[16][16][1])
{
	static float x_1  [18][18][1] = { 0 };
	static float x_2 alignas(16) [16][16][4] = { 0 };
	static float x_3  [18][18][4] = { 0 };
	static float x_4 alignas(16) [16][16][4] = { 0 };
	static float x_5  [16][16][4] = { 0 };
	static float x_6  [10][10][4] = { 0 };
	static float x_7 alignas(16) [8][8][8] = { 0 };
	static float x_8  [8][8][8] = { 0 };
	static float x_9  [6][6][8] = { 0 };
	static float x_10 alignas(16) [4][4][8] = { 0 };
	static float x_11  [4][4][8] = { 0 };
	static float x_12 alignas(16) [4][4][8] = { 0 };
	float *x_13 ;
	for (int i_72 = 0; i_72 < 16; i_72 += 1) {
		for (int i_71 = 0; i_71 < 16; i_71 += 1) {
			for (int i_70 = 0; i_70 < 1; i_70 += 1) {
				x_1[i_72 + 1][i_71 + 1][i_70 + 0] = x_0[i_72 + 0][i_71 + 0][i_70 + 0] - 99;
			}
		}
	}
	for (int i_61 = 0; i_61 < 16; i_61 += 1) {
		for (int i_62 = 0; i_62 < 16; i_62 += 1) {
			for (int i_63 = 0; i_63 < 4; i_63 += 1) {
				x_2[i_61 + 0][i_62 + 0][i_63 + 0] = b_5[i_63 + 0];
			}
		}
	}
	for (int i_64 = 0; i_64 < 16; i_64 += 1) {
		for (int i_65 = 0; i_65 < 16; i_65 += 1) {
			for (int i_66 = 0; i_66 < 3; i_66 += 1) {
				for (int i_67 = 0; i_67 < 3; i_67 += 1) {
					for (int i_68 = 0; i_68 < 1; i_68 += 1) {
						for (int i_69 = 0; i_69 < 4; i_69 += 4) {
							{
							    __m128 w, x, y;
							    w = _mm_load_ps((float*)&w_5[i_66][i_67][i_68][i_69]);
							    x = _mm_load_ps1(&x_1[i_64 + i_66][i_65 + i_67][i_68]);
							    y = _mm_mul_ps(w, x);
							    x = _mm_load_ps((float*)&x_2[i_64 / 1 + 0][i_65 / 1 + 0][i_69 + 0]);
							    x = _mm_add_ps(x, y);
							    _mm_store_ps((float*)&x_2[i_64 / 1 + 0][i_65 / 1 + 0][i_69 + 0], x);
							}
						}
					}
				}
			}
		}
	}
	for (int i_58 = 0; i_58 < 16; i_58 += 1) {
		for (int i_59 = 0; i_59 < 16; i_59 += 1) {
			for (int i_60 = 0; i_60 < 4; i_60 += 1) {
				x_3[i_58 + 1][i_59 + 1][i_60 + 0] = x_2[i_58 + 0][i_59 + 0][i_60 + 0] < 0 ? 0 : x_2[i_58 + 0][i_59 + 0][i_60 + 0];
			}
		}
	}
	for (int i_49 = 0; i_49 < 16; i_49 += 1) {
		for (int i_50 = 0; i_50 < 16; i_50 += 1) {
			for (int i_51 = 0; i_51 < 4; i_51 += 1) {
				x_4[i_49 + 0][i_50 + 0][i_51 + 0] = b_4[i_51 + 0];
			}
		}
	}
	for (int i_52 = 0; i_52 < 16; i_52 += 1) {
		for (int i_53 = 0; i_53 < 16; i_53 += 1) {
			for (int i_54 = 0; i_54 < 3; i_54 += 1) {
				for (int i_55 = 0; i_55 < 3; i_55 += 1) {
					for (int i_56 = 0; i_56 < 4; i_56 += 1) {
						for (int i_57 = 0; i_57 < 4; i_57 += 4) {
							{
							    __m128 w, x, y;
							    w = _mm_load_ps((float*)&w_4[i_54][i_55][i_56][i_57]);
							    x = _mm_load_ps1(&x_3[i_52 + i_54][i_53 + i_55][i_56]);
							    y = _mm_mul_ps(w, x);
							    x = _mm_load_ps((float*)&x_4[i_52 / 1 + 0][i_53 / 1 + 0][i_57 + 0]);
							    x = _mm_add_ps(x, y);
							    _mm_store_ps((float*)&x_4[i_52 / 1 + 0][i_53 / 1 + 0][i_57 + 0], x);
							}
						}
					}
				}
			}
		}
	}
	for (int i_46 = 0; i_46 < 16; i_46 += 1) {
		for (int i_47 = 0; i_47 < 16; i_47 += 1) {
			for (int i_48 = 0; i_48 < 4; i_48 += 1) {
				x_5[i_46 + 0][i_47 + 0][i_48 + 0] = x_4[i_46 + 0][i_47 + 0][i_48 + 0] < 0 ? 0 : x_4[i_46 + 0][i_47 + 0][i_48 + 0];
			}
		}
	}
	for (int i_41 = 0; i_41 < 15; i_41 += 2) {
		for (int i_42 = 0; i_42 < 15; i_42 += 2) {
			for (int i_43 = 0; i_43 < 4; i_43 += 1) {
				x_6[i_41 / 2 + 1][i_42 / 2 + 1][i_43 + 0] = x_5[i_41][i_42][i_43];
				for (int i_44 = 0; i_44 < 2; i_44 += 1) {
					for (int i_45 = 0; i_45 < 2; i_45 += 1) {
						x_6[i_41 / 2 + 1][i_42 / 2 + 1][i_43 + 0] = x_5[i_41 + i_44][i_42 + i_45][i_43] > x_6[i_41 / 2 + 1][i_42 / 2 + 1][i_43 + 0] ? x_5[i_41 + i_44][i_42 + i_45][i_43] : x_6[i_41 / 2 + 1][i_42 / 2 + 1][i_43 + 0];
					}
				}
			}
		}
	}
	for (int i_32 = 0; i_32 < 8; i_32 += 1) {
		for (int i_33 = 0; i_33 < 8; i_33 += 1) {
			for (int i_34 = 0; i_34 < 8; i_34 += 1) {
				x_7[i_32 + 0][i_33 + 0][i_34 + 0] = b_3[i_34 + 0];
			}
		}
	}
	for (int i_35 = 0; i_35 < 8; i_35 += 1) {
		for (int i_36 = 0; i_36 < 8; i_36 += 1) {
			for (int i_37 = 0; i_37 < 3; i_37 += 1) {
				for (int i_38 = 0; i_38 < 3; i_38 += 1) {
					for (int i_39 = 0; i_39 < 4; i_39 += 1) {
						for (int i_40 = 0; i_40 < 8; i_40 += 4) {
							{
							    __m128 w, x, y;
							    w = _mm_load_ps((float*)&w_3[i_37][i_38][i_39][i_40]);
							    x = _mm_load_ps1(&x_6[i_35 + i_37][i_36 + i_38][i_39]);
							    y = _mm_mul_ps(w, x);
							    x = _mm_load_ps((float*)&x_7[i_35 / 1 + 0][i_36 / 1 + 0][i_40 + 0]);
							    x = _mm_add_ps(x, y);
							    _mm_store_ps((float*)&x_7[i_35 / 1 + 0][i_36 / 1 + 0][i_40 + 0], x);
							}
						}
					}
				}
			}
		}
	}
	for (int i_29 = 0; i_29 < 8; i_29 += 1) {
		for (int i_30 = 0; i_30 < 8; i_30 += 1) {
			for (int i_31 = 0; i_31 < 8; i_31 += 1) {
				x_8[i_29 + 0][i_30 + 0][i_31 + 0] = x_7[i_29 + 0][i_30 + 0][i_31 + 0] < 0 ? 0 : x_7[i_29 + 0][i_30 + 0][i_31 + 0];
			}
		}
	}
	for (int i_24 = 0; i_24 < 7; i_24 += 2) {
		for (int i_25 = 0; i_25 < 7; i_25 += 2) {
			for (int i_26 = 0; i_26 < 8; i_26 += 1) {
				x_9[i_24 / 2 + 1][i_25 / 2 + 1][i_26 + 0] = x_8[i_24][i_25][i_26];
				for (int i_27 = 0; i_27 < 2; i_27 += 1) {
					for (int i_28 = 0; i_28 < 2; i_28 += 1) {
						x_9[i_24 / 2 + 1][i_25 / 2 + 1][i_26 + 0] = x_8[i_24 + i_27][i_25 + i_28][i_26] > x_9[i_24 / 2 + 1][i_25 / 2 + 1][i_26 + 0] ? x_8[i_24 + i_27][i_25 + i_28][i_26] : x_9[i_24 / 2 + 1][i_25 / 2 + 1][i_26 + 0];
					}
				}
			}
		}
	}
	for (int i_15 = 0; i_15 < 4; i_15 += 1) {
		for (int i_16 = 0; i_16 < 4; i_16 += 1) {
			for (int i_17 = 0; i_17 < 8; i_17 += 1) {
				x_10[i_15 + 0][i_16 + 0][i_17 + 0] = b_2[i_17 + 0];
			}
		}
	}
	for (int i_18 = 0; i_18 < 4; i_18 += 1) {
		for (int i_19 = 0; i_19 < 4; i_19 += 1) {
			for (int i_20 = 0; i_20 < 3; i_20 += 1) {
				for (int i_21 = 0; i_21 < 3; i_21 += 1) {
					for (int i_22 = 0; i_22 < 8; i_22 += 1) {
						for (int i_23 = 0; i_23 < 8; i_23 += 4) {
							{
							    __m128 w, x, y;
							    w = _mm_load_ps((float*)&w_2[i_20][i_21][i_22][i_23]);
							    x = _mm_load_ps1(&x_9[i_18 + i_20][i_19 + i_21][i_22]);
							    y = _mm_mul_ps(w, x);
							    x = _mm_load_ps((float*)&x_10[i_18 / 1 + 0][i_19 / 1 + 0][i_23 + 0]);
							    x = _mm_add_ps(x, y);
							    _mm_store_ps((float*)&x_10[i_18 / 1 + 0][i_19 / 1 + 0][i_23 + 0], x);
							}
						}
					}
				}
			}
		}
	}
	for (int i_12 = 0; i_12 < 4; i_12 += 1) {
		for (int i_13 = 0; i_13 < 4; i_13 += 1) {
			for (int i_14 = 0; i_14 < 8; i_14 += 1) {
				x_11[i_12 + 0][i_13 + 0][i_14 + 0] = x_10[i_12 + 0][i_13 + 0][i_14 + 0] < 0 ? 0 : x_10[i_12 + 0][i_13 + 0][i_14 + 0];
			}
		}
	}
	for (int i_3 = 0; i_3 < 4; i_3 += 1) {
		for (int i_4 = 0; i_4 < 4; i_4 += 1) {
			for (int i_5 = 0; i_5 < 8; i_5 += 1) {
				x_12[i_3 + 0][i_4 + 0][i_5 + 0] = b_1[i_5 + 0];
			}
		}
	}
	for (int i_6 = 0; i_6 < 4; i_6 += 1) {
		for (int i_7 = 0; i_7 < 4; i_7 += 1) {
			for (int i_8 = 0; i_8 < 1; i_8 += 1) {
				for (int i_9 = 0; i_9 < 1; i_9 += 1) {
					for (int i_10 = 0; i_10 < 8; i_10 += 1) {
						for (int i_11 = 0; i_11 < 8; i_11 += 4) {
							{
							    __m128 w, x, y;
							    w = _mm_load_ps((float*)&w_1[i_8][i_9][i_10][i_11]);
							    x = _mm_load_ps1(&x_11[i_6 + i_8][i_7 + i_9][i_10]);
							    y = _mm_mul_ps(w, x);
							    x = _mm_load_ps((float*)&x_12[i_6 / 1 + 0][i_7 / 1 + 0][i_11 + 0]);
							    x = _mm_add_ps(x, y);
							    _mm_store_ps((float*)&x_12[i_6 / 1 + 0][i_7 / 1 + 0][i_11 + 0], x);
							}
						}
					}
				}
			}
		}
	}
	x_13 = (float*)x_12;
	for (int i_0 = 0; i_0 < 4; i_0 += 1) {
		scores[i_0 + 0] = b_0[i_0 + 0];
	}
	for (int i_1 = 0; i_1 < 128; i_1 += 1) {
		for (int i_2 = 0; i_2 < 4; i_2 += 4) {
			scores[i_2 + 0] +=  x_13[i_1] * w_0[i_1][i_2];
			scores[(i_2 + 1) + 0] +=  x_13[i_1] * w_0[i_1][(i_2 + 1)];
			scores[(i_2 + 2) + 0] +=  x_13[i_1] * w_0[i_1][(i_2 + 2)];
			scores[(i_2 + 3) + 0] +=  x_13[i_1] * w_0[i_1][(i_2 + 3)];
		}
	}
}

void dummy_model2::predict(const BallCandidates::PatchYUVClassified& patch, double meanBrightnessOffset){

    ASSERT(patch.size() == 16);

    for(size_t x=0; x < patch.size(); x++) {
        for(size_t y=0; y < patch.size(); y++) {
            // TODO: check
            // .pixel.y accesses the brightness channel of the pixel
            // subtract the mean brightness calculated on the dataset and the offset from the module parameters
            float value = (static_cast<float>((patch.data[patch.size() * x + y].pixel.y)) / 255.0f) - -0.000000f - static_cast<float>(meanBrightnessOffset);
            in_step[y][x][0] = value;
        }
    }
    cnn(in_step);
}

double dummy_model2::getRadius() {
    return scores[0];
}

Vector2d dummy_model2::getCenter() {
    return Vector2d(scores[1], scores[2]);
}

double dummy_model2::getBallConfidence() {
    return scores[3];
}