#include "dummy_model2.h"

#if WIN32
	#define alignas(x) __declspec(align(x))
#endif

#include <emmintrin.h>
void dummy_model2::cnn(float x0[16][16][1])
{
	__m128 w, x, y;

 	// Convolution Layer
	alignas(16) static float x1 [16][16][4] = {};
	for (int i = 0; i < 16; i += 1)
	{
		for (int j = 0; j < 16; j += 1)
		{
			x1[i][j][0] = 0.0f;
			x1[i][j][1] = 0.0f;
			x1[i][j][2] = 0.0f;
			x1[i][j][3] = 0.0f;
		}
	}
	for (int ix = -1; ix < 15; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 1) / 1;
		for (int jx = -1; jx < 15; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 1) / 1;
			x_1 = ix + 0;
			if (x_1 >= 0 && x_1 < 16)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.23455852270126343f, -0.16197308897972107f, -0.2602453827857971f, 0.17989498376846313f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.041612058877944946f, -0.021744877099990845f, 0.31639665365219116f, -0.08846500515937805f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.12105514109134674f, -0.17537029087543488f, -0.2614867389202118f, -0.15569531917572021f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
			}
			x_1 = ix + 1;
			if (x_1 >= 0 && x_1 < 16)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.18553286790847778f, -0.3649568259716034f, -0.13164272904396057f, 0.141585111618042f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.31273823976516724f, -0.029049396514892578f, 0.3118011951446533f, -0.2737172842025757f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.31311607360839844f, 0.12439286708831787f, 0.24791771173477173f, -0.12876275181770325f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
			}
			x_1 = ix + 2;
			if (x_1 >= 0 && x_1 < 16)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.17308884859085083f, -0.33317095041275024f, -0.28392040729522705f, -0.10389938950538635f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.30459749698638916f, 0.2935205101966858f, -0.08973380923271179f, 0.047162264585494995f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.2402704954147339f, -0.32106637954711914f, 0.28586339950561523f, -0.051541924476623535f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
			}
		}
	}
	for (int i = 0; i < 16; i += 1)
	{
		for (int j = 0; j < 16; j += 1)
		{

			x = _mm_load_ps((float*)&x1[i][j][0]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x1[i][j][0], x);
		}
	}

 	// Convolution Layer
	alignas(16) static float x2 [16][16][4] = {};
	for (int i = 0; i < 16; i += 1)
	{
		for (int j = 0; j < 16; j += 1)
		{
			x2[i][j][0] = 0.0f;
			x2[i][j][1] = 0.0f;
			x2[i][j][2] = 0.0f;
			x2[i][j][3] = 0.0f;
		}
	}
	for (int ix = -1; ix < 15; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 1) / 1;
		for (int jx = -1; jx < 15; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 1) / 1;
			x_1 = ix + 0;
			if (x_1 >= 0 && x_1 < 16)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.03909550607204437f, -0.02472171187400818f, 0.10220760107040405f, 0.0948137640953064f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.09767076373100281f, -0.12707887589931488f, -0.0035262703895568848f, 0.11901327967643738f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.13857576251029968f, 0.12244418263435364f, -0.017808616161346436f, 0.24095219373703003f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.21592479944229126f, 0.2771769165992737f, 0.05412638187408447f, 0.2780246138572693f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.0876585841178894f, -0.0398479700088501f, 0.19168710708618164f, 0.047561049461364746f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1439979076385498f, 0.10803353786468506f, -0.017992854118347168f, -0.2770645320415497f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.04805514216423035f, -0.03936357796192169f, 0.09569725394248962f, 0.20149818062782288f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.003468811511993408f, 0.10258519649505615f, -0.23233649134635925f, -0.11124825477600098f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.12195330858230591f, 0.04061201214790344f, 0.08056426048278809f, 0.14379486441612244f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.19874833524227142f, -0.17946816980838776f, -0.07790477573871613f, -0.1836007982492447f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.22899480164051056f, -0.1267535388469696f, 0.05144307017326355f, -0.06844814121723175f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1768360137939453f, 0.15317758917808533f, -0.1646256148815155f, -0.202753484249115f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
			}
			x_1 = ix + 1;
			if (x_1 >= 0 && x_1 < 16)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.264504998922348f, -0.22201955318450928f, 0.28207284212112427f, -0.06473803520202637f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.12526491284370422f, -0.18609710037708282f, 0.2710109353065491f, 0.1359330713748932f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.2635642886161804f, -0.20049339532852173f, -0.26850876212120056f, -0.00422513484954834f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.10133986175060272f, 0.12851610779762268f, -0.13259847462177277f, -0.003527224063873291f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(-0.08337289094924927f, 0.13315659761428833f, -0.0896470844745636f, -0.12488362193107605f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.05250898003578186f, 0.1930227279663086f, 0.16642677783966064f, -0.1598263382911682f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.02233600616455078f, -0.06855371594429016f, 0.03624516725540161f, 0.02139344811439514f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.13236942887306213f, 0.1171979308128357f, 0.18224382400512695f, 0.018107086420059204f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.1910383701324463f, 0.04298579692840576f, 0.11529657244682312f, -0.25443512201309204f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.07071244716644287f, -0.15853682160377502f, -0.15776418149471283f, -0.24054613709449768f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.0644301176071167f, 0.0751299262046814f, -0.25044214725494385f, -0.10044202208518982f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.22543120384216309f, -0.07249529659748077f, -0.11475938558578491f, -0.19923120737075806f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
			}
			x_1 = ix + 2;
			if (x_1 >= 0 && x_1 < 16)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.1291581094264984f, -0.25075191259384155f, -0.027210861444473267f, -0.08706964552402496f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.20988209545612335f, 0.24110043048858643f, -0.18353059887886047f, 0.1638663113117218f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1309925615787506f, 0.14653551578521729f, 0.21599668264389038f, -0.08468697965145111f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.16096237301826477f, 0.07511183619499207f, 0.13181427121162415f, -0.2336946278810501f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.15417549014091492f, 0.026776164770126343f, -0.1224268227815628f, -0.011935591697692871f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.16107428073883057f, -0.1535009890794754f, 0.28070712089538574f, 0.1148996651172638f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.18470647931098938f, -0.04793456196784973f, 0.10170090198516846f, 0.2787216305732727f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.039356082677841187f, 0.1597234308719635f, 0.13635829091072083f, 0.25903791189193726f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 16)
				{

					w = _mm_set_ps(0.13562121987342834f, 0.19966399669647217f, 0.07959580421447754f, -0.16671054065227509f);
					x = _mm_load_ps1(&x1[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.04096852242946625f, -0.042116597294807434f, 0.24597954750061035f, -0.07386045157909393f);
					x = _mm_load_ps1(&x1[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.21840500831604004f, -0.18805532157421112f, -0.062018319964408875f, -0.14024485647678375f);
					x = _mm_load_ps1(&x1[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.2592436969280243f, 0.2107325792312622f, -0.13246922194957733f, 0.17124414443969727f);
					x = _mm_load_ps1(&x1[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x2[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
				}
			}
		}
	}
	for (int i = 0; i < 16; i += 1)
	{
		for (int j = 0; j < 16; j += 1)
		{

			x = _mm_load_ps((float*)&x2[i][j][0]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x2[i][j][0], x);
		}
	}

 	// Maxpool Layer 
	static float x3[8][8][4] = {};
	for (int ix = 0; ix < 15; ix += 2)
	{
		int x_out_1;
		x_out_1 = ix / 2;
	for (int jx = 0; jx < 15; jx += 2)
	{
		int x_out_2;
		x_out_2 = jx / 2;
		x = _mm_load_ps((float*)&x2[ix][jx][0]);
		y = _mm_load_ps((float*)&x2[ix + 0][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x2[ix + 0][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);
		y = _mm_load_ps((float*)&x2[ix + 1][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x2[ix + 1][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);
		}
	}

 	// Convolution Layer
	alignas(16) static float x4 [8][8][8] = {};
	for (int i = 0; i < 8; i += 1)
	{
		for (int j = 0; j < 8; j += 1)
		{
			x4[i][j][0] = 0.0f;
			x4[i][j][1] = 0.0f;
			x4[i][j][2] = 0.0f;
			x4[i][j][3] = 0.0f;
			x4[i][j][4] = 0.0f;
			x4[i][j][5] = 0.0f;
			x4[i][j][6] = 0.0f;
			x4[i][j][7] = 0.0f;
		}
	}
	for (int ix = -1; ix < 7; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 1) / 1;
		for (int jx = -1; jx < 7; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 1) / 1;
			x_1 = ix + 0;
			if (x_1 >= 0 && x_1 < 8)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(0.08476637303829193f, -0.11796104907989502f, -0.05720052123069763f, -0.042417436838150024f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.18614870309829712f, 0.06338109076023102f, 0.13552959263324738f, -0.1473839282989502f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1632200926542282f, 0.01716504991054535f, -0.14385247230529785f, -0.2145368456840515f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.07319311797618866f, 0.1482662707567215f, -0.19298553466796875f, -0.12708544731140137f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1711137443780899f, -0.16373325884342194f, -0.00042529404163360596f, 0.0869864970445633f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1155063584446907f, 0.18907998502254486f, 0.10296200215816498f, -0.1750723421573639f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.05356205999851227f, 0.15137238800525665f, -0.10003335773944855f, 0.15497587621212006f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.06510917842388153f, -0.11424627900123596f, 0.12054981291294098f, 0.010311931371688843f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(0.03374646604061127f, 0.007348045706748962f, -0.11839021742343903f, -0.07316844165325165f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.10503961145877838f, 9.733438491821289e-05f, 0.16058816015720367f, -0.1022084653377533f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.07640144228935242f, 0.1776859015226364f, 0.07798503339290619f, 0.034823521971702576f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.017818957567214966f, 0.10090465843677521f, 0.17331655323505402f, 0.05401147902011871f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.21828033030033112f, 0.10368974506855011f, 0.029274865984916687f, 0.22894100844860077f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1806711107492447f, -0.009360536932945251f, 0.14167259633541107f, -0.19642117619514465f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.07932128012180328f, -0.19670811295509338f, -0.11079715192317963f, -0.15556542575359344f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.23406308889389038f, -0.06714482605457306f, -0.08966764807701111f, -0.04507976770401001f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(0.0665162056684494f, -0.21098493039608002f, 0.07715125381946564f, -0.00430162250995636f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.18180060386657715f, 0.00302731990814209f, -0.15717779099941254f, -0.03792580962181091f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.19859780371189117f, 0.09586338698863983f, 0.16620637476444244f, 0.14999322593212128f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.06844782829284668f, -0.026121824979782104f, -0.12803751230239868f, -0.14424242079257965f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.12788966298103333f, 0.06256558001041412f, 0.09054075181484222f, -0.11262436211109161f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.22078262269496918f, -0.0008413046598434448f, 0.2025960236787796f, 0.07302890717983246f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.18130478262901306f, 0.021585360169410706f, -0.10407990217208862f, -0.2228383719921112f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.011986449360847473f, -0.03637115657329559f, -0.2019912451505661f, 0.14304398000240326f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
			}
			x_1 = ix + 1;
			if (x_1 >= 0 && x_1 < 8)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(0.0827995091676712f, -0.16779452562332153f, 0.2320612221956253f, -0.14964890480041504f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1454794555902481f, -0.04152728617191315f, -0.08269858360290527f, -0.1693105697631836f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.0008915513753890991f, 0.030295953154563904f, -0.15706861019134521f, 0.025768741965293884f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.20075346529483795f, -0.004012435674667358f, -0.034523606300354004f, 0.1789938360452652f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.00512579083442688f, 0.062281534075737f, 0.11778570711612701f, -0.02080446481704712f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.08357422053813934f, -0.0030674487352371216f, 0.21936316788196564f, 0.15241952240467072f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.011626511812210083f, 0.1007213145494461f, -0.17041972279548645f, 0.1067131906747818f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.009138107299804688f, -0.04600384831428528f, 0.09784255921840668f, -0.20804712176322937f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(-0.010534346103668213f, -0.1804789900779724f, 0.03889282047748566f, 0.00855417549610138f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.10864607989788055f, -0.15337608754634857f, -0.07022012770175934f, 0.18087787926197052f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.23511682450771332f, 0.09566520154476166f, -0.04580913484096527f, 0.04797394573688507f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.20306254923343658f, 0.2304946631193161f, 0.012302219867706299f, 0.15889610350131989f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.09826914966106415f, 0.03929503262042999f, -0.1280193030834198f, 0.033767834305763245f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.03826265037059784f, 0.11333726346492767f, 0.0736069530248642f, 0.13558237254619598f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.23186790943145752f, 0.000948980450630188f, 0.1582014411687851f, 0.11349375545978546f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.18066222965717316f, 0.1294943243265152f, -0.18569642305374146f, -0.23041491210460663f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(0.020881399512290955f, -0.08611725270748138f, -0.027691379189491272f, 0.1421576589345932f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.11048446595668793f, -0.2054215520620346f, -0.20979195833206177f, -0.170294851064682f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.21039049327373505f, 0.046522095799446106f, -0.12626612186431885f, 0.1176876574754715f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.138792484998703f, 0.01744513213634491f, -0.19170573353767395f, -0.18263201415538788f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.07898004353046417f, -0.21337082982063293f, -0.018645599484443665f, 0.14543534815311432f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.18819551169872284f, -0.0653262734413147f, 0.03176771104335785f, -0.1479969173669815f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.17381422221660614f, 0.05135829746723175f, -0.15220671892166138f, 0.13256235420703888f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1912885308265686f, 0.22000347077846527f, -0.18096794188022614f, -0.0201788991689682f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
			}
			x_1 = ix + 2;
			if (x_1 >= 0 && x_1 < 8)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(-0.06435267627239227f, -0.03951430320739746f, -0.07191140949726105f, 0.18912173807621002f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.14957933127880096f, -0.18163537979125977f, -0.1654035747051239f, -0.1484474390745163f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.23115794360637665f, -0.09781867265701294f, 0.10897500813007355f, 0.12505970895290375f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.11841268837451935f, 0.10505248606204987f, 0.020641908049583435f, -0.20812192559242249f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.08099336922168732f, 0.08539192378520966f, 0.20706696808338165f, 0.1654260903596878f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.106554314494133f, 0.22911269962787628f, -0.18943479657173157f, -0.0678490698337555f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.10054849088191986f, 0.15811176598072052f, -0.21308372914791107f, -0.23053433001041412f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.10533975064754486f, 0.22706447541713715f, 0.042900606989860535f, 0.00401170551776886f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(-0.1777305006980896f, 0.042347297072410583f, -0.1807955950498581f, 0.11487169563770294f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.23302936553955078f, 0.10226307809352875f, -0.225884348154068f, 0.13259492814540863f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.13298971951007843f, -0.08898031711578369f, 0.19907121360301971f, -0.03682757914066315f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.08829540014266968f, 0.18424849212169647f, 0.05839093029499054f, 0.17465703189373016f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.15932469069957733f, 0.017513349652290344f, 0.03816904127597809f, -0.16717305779457092f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1302480250597f, 0.10949356853961945f, -0.020424365997314453f, 0.1882990151643753f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.0648268610239029f, -0.1236952692270279f, -0.19577941298484802f, -0.13426721096038818f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.17245624959468842f, -0.07628309726715088f, -0.0641917884349823f, 0.09838153421878815f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 8)
				{

					w = _mm_set_ps(0.029177024960517883f, -0.1429506540298462f, -0.07454024255275726f, -0.059038132429122925f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.15129391849040985f, 0.22998888790607452f, 0.2054090052843094f, 0.049065008759498596f);
					x = _mm_load_ps1(&x3[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.19213108718395233f, 0.11628912389278412f, 0.1741098016500473f, -0.16591984033584595f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.11293388903141022f, -0.09701810777187347f, -0.02509710192680359f, -0.1078842431306839f);
					x = _mm_load_ps1(&x3[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.2307066172361374f, 0.06271855533123016f, 0.024689242243766785f, 0.16453589498996735f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.13231199979782104f, 0.03602711856365204f, -0.20875345170497894f, -0.030004113912582397f);
					x = _mm_load_ps1(&x3[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.15980853140354156f, -0.10728777945041656f, -0.19110387563705444f, 0.17676125466823578f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1107085794210434f, 0.19267718493938446f, -0.01718197762966156f, -0.23407965898513794f);
					x = _mm_load_ps1(&x3[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x4[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
				}
			}
		}
	}
	for (int i = 0; i < 8; i += 1)
	{
		for (int j = 0; j < 8; j += 1)
		{

			x = _mm_load_ps((float*)&x4[i][j][0]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x4[i][j][0], x);

			x = _mm_load_ps((float*)&x4[i][j][4]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x4[i][j][4], x);
		}
	}

 	// Maxpool Layer 
	static float x5[4][4][8] = {};
	for (int ix = 0; ix < 7; ix += 2)
	{
		int x_out_1;
		x_out_1 = ix / 2;
	for (int jx = 0; jx < 7; jx += 2)
	{
		int x_out_2;
		x_out_2 = jx / 2;
		x = _mm_load_ps((float*)&x4[ix][jx][0]);
		y = _mm_load_ps((float*)&x4[ix + 0][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x4[ix + 0][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);
		y = _mm_load_ps((float*)&x4[ix + 1][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x4[ix + 1][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);
		x = _mm_load_ps((float*)&x4[ix][jx][4]);
		y = _mm_load_ps((float*)&x4[ix + 0][jx + 0][4]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x4[ix + 0][jx + 1][4]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);
		y = _mm_load_ps((float*)&x4[ix + 1][jx + 0][4]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x4[ix + 1][jx + 1][4]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);
		}
	}

 	// Convolution Layer
	alignas(16) static float x6 [4][4][8] = {};
	for (int i = 0; i < 4; i += 1)
	{
		for (int j = 0; j < 4; j += 1)
		{
			x6[i][j][0] = 0.0f;
			x6[i][j][1] = 0.0f;
			x6[i][j][2] = 0.0f;
			x6[i][j][3] = 0.0f;
			x6[i][j][4] = 0.0f;
			x6[i][j][5] = 0.0f;
			x6[i][j][6] = 0.0f;
			x6[i][j][7] = 0.0f;
		}
	}
	for (int ix = -1; ix < 3; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 1) / 1;
		for (int jx = -1; jx < 3; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 1) / 1;
			x_1 = ix + 0;
			if (x_1 >= 0 && x_1 < 4)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.004497751593589783f, -0.17289453744888306f, -0.040220245718955994f, -0.042535871267318726f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1271112710237503f, 0.14285090565681458f, -0.1731814444065094f, -0.10090282559394836f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.09828376770019531f, -0.18929186463356018f, 0.06546935439109802f, -0.18060335516929626f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.18668311834335327f, 0.04611426591873169f, -0.07892005145549774f, -0.0416540801525116f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1504652500152588f, -0.1330728828907013f, -0.1703869253396988f, -0.17311102151870728f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.01699407398700714f, -0.08764603734016418f, 0.08237564563751221f, 0.012615025043487549f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.011206835508346558f, -0.09423185139894485f, -0.03410772979259491f, -0.18945865333080292f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1284600794315338f, -0.1873588114976883f, 0.07703915238380432f, 0.008110016584396362f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.05530989170074463f, 0.06667709350585938f, 0.039040759205818176f, 0.14065873622894287f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.041331946849823f, -0.06050601601600647f, -0.13767001032829285f, -0.09797482937574387f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.0751541405916214f, -0.10723114013671875f, 0.16584926843643188f, -0.09340748190879822f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.12415192276239395f, -0.12554466724395752f, -0.053486913442611694f, 0.1532226800918579f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.08760729432106018f, 0.0478174090385437f, -0.07257509231567383f, -0.12197568267583847f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.04936492443084717f, -0.0472370982170105f, 0.08473697304725647f, 0.08819839358329773f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.08778834342956543f, -0.11212777346372604f, 0.14975091814994812f, -0.03308337926864624f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.12482285499572754f, -0.001491248607635498f, 0.005671113729476929f, -0.12455148249864578f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.009608805179595947f, 0.026601478457450867f, 0.18312308192253113f, 0.1851147711277008f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.025901496410369873f, 0.0027431100606918335f, -0.0879983901977539f, -0.1241607815027237f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.06047818064689636f, -0.19933916628360748f, -0.04898177087306976f, -0.11985564976930618f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.04950585961341858f, -0.1318080723285675f, 0.019639313220977783f, 0.021843940019607544f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.11860889941453934f, -0.12067418545484543f, 0.05814751982688904f, 0.03850804269313812f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.05802389979362488f, -0.15800848603248596f, -0.07966216653585434f, -0.04499487578868866f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.16058427095413208f, -0.040521204471588135f, -0.11249233782291412f, -0.030596643686294556f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.17086976766586304f, -0.06980954110622406f, -0.13988211750984192f, 0.1680329442024231f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.11513987183570862f, 0.18604734539985657f, 0.18170857429504395f, -0.03691323101520538f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.07720765471458435f, -0.08244401961565018f, -0.03616468608379364f, -0.17919254302978516f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.11707579344511032f, -0.1643485277891159f, -0.16804206371307373f, -0.1734900325536728f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.19213664531707764f, 0.07930266857147217f, -0.1804080456495285f, -0.11849482357501984f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.017627820372581482f, 0.007936418056488037f, -0.19601325690746307f, -0.15991909801959991f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.038332805037498474f, -0.16380992531776428f, 0.2006722092628479f, 0.0429970920085907f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.012753769755363464f, -0.059841811656951904f, 0.1267876923084259f, -0.06589311361312866f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1439441442489624f, -0.19786357879638672f, 0.02180330455303192f, 0.004720255732536316f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.11852607131004333f, 0.05792933702468872f, -0.12981997430324554f, -0.08951402455568314f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.15380725264549255f, 0.17877197265625f, -0.1311253309249878f, 0.18412166833877563f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.12125936150550842f, -0.0930861309170723f, -0.08652080595493317f, 0.1276092231273651f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.09612023830413818f, -0.031498491764068604f, -0.1647235006093979f, 0.1454801857471466f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.040681853890419006f, -0.1505506932735443f, 0.15175935626029968f, 0.08168074488639832f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.07524427771568298f, -0.048173606395721436f, -0.12526625394821167f, -0.03469046950340271f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.14634066820144653f, -0.17691390216350555f, -0.009302154183387756f, -0.10563919693231583f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.13963329792022705f, -0.1697741150856018f, 0.13701704144477844f, 0.020022958517074585f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.07830318808555603f, -0.041739195585250854f, 0.14763477444648743f, 0.09425705671310425f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.16568779945373535f, -0.18515239655971527f, 0.10076743364334106f, 0.11867961287498474f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.05276325345039368f, -0.11034373939037323f, -0.15857365727424622f, -0.1262068897485733f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.13836361467838287f, -0.08681991696357727f, -0.036934539675712585f, -0.1685589998960495f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.12714719772338867f, 0.05907243490219116f, 0.05442693829536438f, 0.08391964435577393f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.0974544808268547f, 0.09029814600944519f, -0.01886400580406189f, 0.17062470316886902f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.08062878251075745f, -0.055037111043930054f, -0.03804922103881836f, -0.09544224292039871f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.17048221826553345f, 0.1203622817993164f, 0.06523531675338745f, 0.18746578693389893f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
			}
			x_1 = ix + 1;
			if (x_1 >= 0 && x_1 < 4)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.0473894327878952f, 0.16198763251304626f, 0.1894538402557373f, 0.12240585684776306f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1475890576839447f, 0.019377097487449646f, -0.16458748281002045f, 0.19453924894332886f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.06347829103469849f, 0.12165749073028564f, 0.2020626664161682f, 0.14331406354904175f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1473800539970398f, 0.028438210487365723f, -0.04307442903518677f, 0.14512759447097778f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.06877750158309937f, -0.035974353551864624f, -0.02602393925189972f, -0.13477647304534912f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.17018559575080872f, -0.09573955088853836f, -0.058697789907455444f, -0.19271957874298096f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.10665050148963928f, 0.1860155463218689f, -0.11539293825626373f, -0.0987977385520935f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.2008025348186493f, -0.16360387206077576f, 0.1869160532951355f, -0.005942195653915405f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.09496590495109558f, 0.01111467182636261f, 0.14010870456695557f, 0.09126505255699158f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.09446573257446289f, 0.19324737787246704f, -0.023306772112846375f, -0.16083773970603943f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.0009043365716934204f, 0.1786687970161438f, -0.057806357741355896f, -0.1514049470424652f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.0009572803974151611f, 0.09567031264305115f, 0.08633863925933838f, -0.10091231763362885f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.07777291536331177f, -0.08644372224807739f, 0.06653240323066711f, 0.07749691605567932f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.11360681056976318f, 0.07409968972206116f, -0.0820135623216629f, -0.12625667452812195f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.13488894701004028f, -0.1017138659954071f, 0.1045902669429779f, -0.10457675158977509f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.14620551466941833f, -0.014124035835266113f, 0.11913499236106873f, 0.0023131966590881348f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.08693322539329529f, 0.1858917474746704f, -0.18260395526885986f, -0.03963187336921692f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.04828251898288727f, -0.02575734257698059f, -0.10498389601707458f, -0.13110673427581787f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.20234274864196777f, -0.177269846200943f, -0.04627081751823425f, 0.18073835968971252f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.07835936546325684f, 0.023178428411483765f, -0.11816758662462234f, -0.19536283612251282f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.11609423160552979f, 0.20387625694274902f, -0.013554975390434265f, 0.15474623441696167f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.16511180996894836f, -0.12278380244970322f, 0.01430279016494751f, -0.006352156400680542f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.10389969497919083f, 0.13768315315246582f, 0.08616200089454651f, -0.126787930727005f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.15122473239898682f, 0.043664172291755676f, 0.014382362365722656f, -0.08075406402349472f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1107800304889679f, -0.19860224425792694f, 0.07466128468513489f, -0.06155999004840851f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.18242496252059937f, -0.17632585763931274f, -0.20303508639335632f, -0.09898622334003448f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.10936752706766129f, -0.016244888305664062f, 0.029229149222373962f, 0.11613130569458008f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1606120616197586f, -0.18747079372406006f, 0.04165485501289368f, -0.20144644379615784f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.08917465806007385f, -0.029598966240882874f, -0.17029689252376556f, 0.07787546515464783f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.18994355201721191f, -0.029194846749305725f, 0.1942010521888733f, -0.16488926112651825f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.14995858073234558f, -0.04591968655586243f, 0.014802694320678711f, 0.08719825744628906f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.17958995699882507f, 0.041024670004844666f, -0.12034869939088821f, -0.12274004518985748f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.12040865421295166f, 0.012197449803352356f, -0.1575380116701126f, 0.08248820900917053f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.09576860815286636f, 0.05315914750099182f, 0.1651121973991394f, -0.04494732618331909f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.11283057928085327f, 0.0674346387386322f, 0.14258050918579102f, -0.06800885498523712f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.15908022224903107f, -0.004671841859817505f, -0.004087984561920166f, 0.04509444534778595f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.11600735783576965f, 0.1198592483997345f, -0.04269866645336151f, 0.09398701786994934f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.15976423025131226f, -0.004916638135910034f, 0.09883472323417664f, -0.13913989067077637f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.13364817202091217f, 0.14735031127929688f, -0.14076104760169983f, 0.08168560266494751f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.07449324429035187f, 0.15564438700675964f, -0.18424490094184875f, 0.09085023403167725f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.16462913155555725f, -0.18101760745048523f, 0.05640086531639099f, 0.10382139682769775f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.059899523854255676f, 0.023714065551757812f, -0.09815066307783127f, -0.03463678061962128f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.030631735920906067f, 0.1511133313179016f, 0.10982775688171387f, 0.1573379933834076f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.0765179991722107f, -0.0817306637763977f, -0.1635798215866089f, 0.07745686173439026f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.18113523721694946f, -0.04044654965400696f, -0.19692450761795044f, 0.053058743476867676f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.13373062014579773f, -0.08072062581777573f, -0.18200023472309113f, 0.148179829120636f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1951364278793335f, -0.056552544236183167f, -0.13592952489852905f, 0.1150137186050415f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.16061459481716156f, -0.13208740949630737f, 0.0666901171207428f, -0.15457341074943542f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
			}
			x_1 = ix + 2;
			if (x_1 >= 0 && x_1 < 4)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.08855484426021576f, -0.057947203516960144f, 0.12641119956970215f, 0.0576779842376709f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.08699536323547363f, -0.03570808470249176f, -0.1741907000541687f, -0.04262132942676544f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.16651396453380585f, 0.150594562292099f, 0.15617635846138f, 0.06889373064041138f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.0840836614370346f, 0.09412896633148193f, 0.11901050806045532f, -0.20207570493221283f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.1830432116985321f, -0.11992543935775757f, -0.03344234824180603f, -0.16542167961597443f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.15459224581718445f, -0.1331065595149994f, 0.14359906315803528f, -0.013066351413726807f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.06987208127975464f, -0.1627759039402008f, -0.0497526079416275f, 0.040911272168159485f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1621621549129486f, 0.06186211109161377f, 0.18028751015663147f, -0.19536423683166504f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.12142714858055115f, 0.1334114670753479f, 0.13425588607788086f, -0.15028269588947296f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.16638493537902832f, -0.061569541692733765f, 0.10986635088920593f, 0.1905357539653778f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1485380232334137f, -0.01316329836845398f, -0.019033953547477722f, -0.18138231337070465f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.025129348039627075f, 0.10472112894058228f, 0.07162034511566162f, 0.15670445561408997f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.2013624906539917f, -0.1173781156539917f, 0.01978883147239685f, 0.20221251249313354f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1357959508895874f, -0.17112988233566284f, 0.15835708379745483f, -0.08679597079753876f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.01580718159675598f, 0.1572883129119873f, -0.15629583597183228f, -0.199791818857193f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.12227800488471985f, 0.18025153875350952f, 0.13208341598510742f, -0.06130000948905945f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.1709589958190918f, 0.1885366141796112f, 0.14963635802268982f, -0.150874525308609f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.12642419338226318f, 0.1806192696094513f, -0.11546428501605988f, 0.06258964538574219f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.11554010957479477f, 0.060581743717193604f, 0.01226198673248291f, 0.16863146424293518f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.14989900588989258f, -0.17480486631393433f, -0.03070308268070221f, -0.1458347737789154f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.14940863847732544f, -0.05899845063686371f, 0.009104236960411072f, -0.006792157888412476f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.18456684052944183f, 0.16207259893417358f, 0.08848407864570618f, 0.06193706393241882f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.10185688734054565f, 0.07600212097167969f, 0.11674898862838745f, 0.1138094961643219f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.04458698630332947f, -0.09800446778535843f, 0.11345380544662476f, 0.17428991198539734f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.11513558775186539f, 0.07785055041313171f, 0.07636931538581848f, 0.17850813269615173f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1945115029811859f, 0.07478564977645874f, -0.00848543643951416f, -0.05524638295173645f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.09148332476615906f, -0.1117875874042511f, 0.049348264932632446f, 0.11562716960906982f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.12379524856805801f, -0.14384078979492188f, -0.036569640040397644f, 0.055875539779663086f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.20224633812904358f, -0.12607792019844055f, -0.10444612801074982f, 0.10297062993049622f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.13175788521766663f, -0.19592274725437164f, -0.1963822990655899f, -0.19513599574565887f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.018992096185684204f, 0.08033069968223572f, -0.070574089884758f, -0.1400027573108673f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.07314243912696838f, 0.19797545671463013f, 0.17697936296463013f, -0.11773090064525604f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.07992604374885559f, 0.0440153032541275f, -0.2029268443584442f, 0.006792157888412476f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.14706137776374817f, -0.17077884078025818f, 0.1782604157924652f, -0.006026968359947205f);
					x = _mm_load_ps1(&x5[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.16157425940036774f, 0.05369824171066284f, 0.1264384388923645f, 0.061158597469329834f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1620684266090393f, 0.19506001472473145f, -0.19472889602184296f, 0.19027867913246155f);
					x = _mm_load_ps1(&x5[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.16047292947769165f, 0.04441957175731659f, -0.056553080677986145f, -0.11391857266426086f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.012646853923797607f, -0.16819696128368378f, 0.048843324184417725f, 0.11110684275627136f);
					x = _mm_load_ps1(&x5[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.11023759841918945f, -0.13376964628696442f, 0.15260115265846252f, -0.09453777223825455f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.10477784276008606f, -0.11352977156639099f, -0.040733590722084045f, 0.11140462756156921f);
					x = _mm_load_ps1(&x5[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.04485100507736206f, 0.09115332365036011f, 0.0852724015712738f, -0.10085474699735641f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.18505364656448364f, -0.18332694470882416f, -0.06351153552532196f, -0.08746908605098724f);
					x = _mm_load_ps1(&x5[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.11602282524108887f, 0.019245952367782593f, -0.025399789214134216f, -0.12506812810897827f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.012374073266983032f, 0.16878190636634827f, -0.18981921672821045f, 0.0849713385105133f);
					x = _mm_load_ps1(&x5[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.03016170859336853f, 0.15564745664596558f, -0.07410985231399536f, 0.01719316840171814f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.2031630277633667f, 0.18010693788528442f, 0.19894179701805115f, 0.12623697519302368f);
					x = _mm_load_ps1(&x5[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.14054730534553528f, -0.1313289999961853f, -0.11512599885463715f, -0.1278763711452484f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.11968284845352173f, -0.1388835310935974f, -0.18961569666862488f, 0.04839572310447693f);
					x = _mm_load_ps1(&x5[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x6[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
				}
			}
		}
	}
	for (int i = 0; i < 4; i += 1)
	{
		for (int j = 0; j < 4; j += 1)
		{

			x = _mm_load_ps((float*)&x6[i][j][0]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x6[i][j][0], x);

			x = _mm_load_ps((float*)&x6[i][j][4]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x6[i][j][4], x);
		}
	}

 	// Convolution Layer
	alignas(16) static float x7 [4][4][8] = {};
	for (int i = 0; i < 4; i += 1)
	{
		for (int j = 0; j < 4; j += 1)
		{
			x7[i][j][0] = 0.0f;
			x7[i][j][1] = 0.0f;
			x7[i][j][2] = 0.0f;
			x7[i][j][3] = 0.0f;
			x7[i][j][4] = 0.0f;
			x7[i][j][5] = 0.0f;
			x7[i][j][6] = 0.0f;
			x7[i][j][7] = 0.0f;
		}
	}
	for (int ix = -0; ix < 4; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 0) / 1;
		for (int jx = -0; jx < 4; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 0) / 1;
			x_1 = ix + 0;
			if (x_1 >= 0 && x_1 < 4)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.41614001989364624f, -0.14031991362571716f, -0.20000630617141724f, 0.3944515585899353f);
					x = _mm_load_ps1(&x6[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.11350321769714355f, -0.31154635548591614f, -0.3502325415611267f, -0.05825847387313843f);
					x = _mm_load_ps1(&x6[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.3992542624473572f, -0.49433812499046326f, -0.2517716586589813f, 0.11700284481048584f);
					x = _mm_load_ps1(&x6[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.32608047127723694f, -0.20518290996551514f, -0.2742471992969513f, 0.054512977600097656f);
					x = _mm_load_ps1(&x6[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.06641119718551636f, -0.35138827562332153f, -0.5318467617034912f, 0.5903131365776062f);
					x = _mm_load_ps1(&x6[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.5083081126213074f, -0.4649936556816101f, -0.6096117496490479f, -0.3430384695529938f);
					x = _mm_load_ps1(&x6[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.43758195638656616f, -0.08830225467681885f, -0.003412485122680664f, 0.3587731719017029f);
					x = _mm_load_ps1(&x6[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.5287690758705139f, 0.2462710738182068f, -0.03264129161834717f, 0.528988778591156f);
					x = _mm_load_ps1(&x6[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.3590272068977356f, -0.323323518037796f, -0.5774392485618591f, 0.22574597597122192f);
					x = _mm_load_ps1(&x6[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.2061578929424286f, 0.02899068593978882f, -0.15309998393058777f, -0.5744882822036743f);
					x = _mm_load_ps1(&x6[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.5157058238983154f, 0.012317240238189697f, 0.40977388620376587f, -0.4994164705276489f);
					x = _mm_load_ps1(&x6[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.02061372995376587f, -0.023546159267425537f, -0.1521807610988617f, -0.14468097686767578f);
					x = _mm_load_ps1(&x6[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.2641056776046753f, -0.41760218143463135f, 0.4474223256111145f, 0.35814839601516724f);
					x = _mm_load_ps1(&x6[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.5971203446388245f, 0.256473183631897f, 0.5092183947563171f, 0.5310303568840027f);
					x = _mm_load_ps1(&x6[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.031526148319244385f, -0.22771814465522766f, -0.013022422790527344f, 0.43692320585250854f);
					x = _mm_load_ps1(&x6[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.37788641452789307f, -0.4883165955543518f, 0.5652379393577576f, 0.496399462223053f);
					x = _mm_load_ps1(&x6[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x7[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x7[x_out_1][x_out_2][4], x);
				}
			}
		}
	}

 	// Dense Layer
	scores[0] = 0.000000f
	 + 0.069592f * x7[0][0][0] + 0.181627f * x7[0][0][1] - 0.123313f * x7[0][0][2] + 0.119339f * x7[0][0][3]
	 + 0.089933f * x7[0][0][4] + 0.130224f * x7[0][0][5] + 0.108435f * x7[0][0][6] - 0.038136f * x7[0][0][7]
	 + 0.177992f * x7[0][1][0] - 0.101638f * x7[0][1][1] + 0.012987f * x7[0][1][2] - 0.198688f * x7[0][1][3]
	 + 0.048047f * x7[0][1][4] - 0.147186f * x7[0][1][5] + 0.127096f * x7[0][1][6] - 0.073952f * x7[0][1][7]
	 + 0.123724f * x7[0][2][0] + 0.108736f * x7[0][2][1] + 0.038319f * x7[0][2][2] - 0.111684f * x7[0][2][3]
	 - 0.078912f * x7[0][2][4] + 0.171208f * x7[0][2][5] + 0.174022f * x7[0][2][6] - 0.146860f * x7[0][2][7]
	 + 0.075074f * x7[0][3][0] - 0.035434f * x7[0][3][1] - 0.110596f * x7[0][3][2] + 0.179208f * x7[0][3][3]
	 + 0.080504f * x7[0][3][4] + 0.033270f * x7[0][3][5] + 0.152894f * x7[0][3][6] - 0.061884f * x7[0][3][7]
	 + 0.135316f * x7[1][0][0] + 0.062105f * x7[1][0][1] + 0.069722f * x7[1][0][2] - 0.162634f * x7[1][0][3]
	 + 0.166873f * x7[1][0][4] + 0.176082f * x7[1][0][5] + 0.100805f * x7[1][0][6] - 0.019123f * x7[1][0][7]
	 - 0.015856f * x7[1][1][0] - 0.211432f * x7[1][1][1] + 0.013065f * x7[1][1][2] + 0.052459f * x7[1][1][3]
	 - 0.007160f * x7[1][1][4] - 0.032248f * x7[1][1][5] + 0.075820f * x7[1][1][6] + 0.166173f * x7[1][1][7]
	 - 0.203964f * x7[1][2][0] - 0.180895f * x7[1][2][1] + 0.211914f * x7[1][2][2] - 0.074059f * x7[1][2][3]
	 + 0.202116f * x7[1][2][4] + 0.201809f * x7[1][2][5] - 0.107029f * x7[1][2][6] - 0.095800f * x7[1][2][7]
	 - 0.024858f * x7[1][3][0] - 0.058036f * x7[1][3][1] - 0.028563f * x7[1][3][2] + 0.179648f * x7[1][3][3]
	 - 0.005882f * x7[1][3][4] + 0.000388f * x7[1][3][5] - 0.100759f * x7[1][3][6] - 0.082849f * x7[1][3][7]
	 - 0.088358f * x7[2][0][0] + 0.108694f * x7[2][0][1] + 0.118133f * x7[2][0][2] - 0.129938f * x7[2][0][3]
	 + 0.173784f * x7[2][0][4] + 0.067228f * x7[2][0][5] + 0.192252f * x7[2][0][6] - 0.044083f * x7[2][0][7]
	 + 0.212522f * x7[2][1][0] - 0.067496f * x7[2][1][1] + 0.126797f * x7[2][1][2] - 0.117540f * x7[2][1][3]
	 + 0.147593f * x7[2][1][4] - 0.187523f * x7[2][1][5] - 0.097828f * x7[2][1][6] - 0.130010f * x7[2][1][7]
	 - 0.211462f * x7[2][2][0] + 0.145777f * x7[2][2][1] + 0.207027f * x7[2][2][2] - 0.088229f * x7[2][2][3]
	 - 0.039480f * x7[2][2][4] - 0.205406f * x7[2][2][5] - 0.129077f * x7[2][2][6] + 0.067554f * x7[2][2][7]
	 - 0.042038f * x7[2][3][0] + 0.194600f * x7[2][3][1] - 0.141637f * x7[2][3][2] + 0.072552f * x7[2][3][3]
	 + 0.082002f * x7[2][3][4] - 0.122609f * x7[2][3][5] - 0.202532f * x7[2][3][6] + 0.087387f * x7[2][3][7]
	 + 0.143971f * x7[3][0][0] - 0.190186f * x7[3][0][1] + 0.095210f * x7[3][0][2] - 0.135937f * x7[3][0][3]
	 - 0.185647f * x7[3][0][4] + 0.112049f * x7[3][0][5] + 0.155807f * x7[3][0][6] - 0.111601f * x7[3][0][7]
	 + 0.175616f * x7[3][1][0] - 0.179525f * x7[3][1][1] + 0.189198f * x7[3][1][2] + 0.164686f * x7[3][1][3]
	 - 0.207137f * x7[3][1][4] - 0.175560f * x7[3][1][5] - 0.164882f * x7[3][1][6] - 0.122389f * x7[3][1][7]
	 + 0.141953f * x7[3][2][0] + 0.170452f * x7[3][2][1] + 0.056415f * x7[3][2][2] + 0.155537f * x7[3][2][3]
	 - 0.128766f * x7[3][2][4] - 0.178459f * x7[3][2][5] + 0.140574f * x7[3][2][6] + 0.198912f * x7[3][2][7]
	 + 0.126097f * x7[3][3][0] - 0.084551f * x7[3][3][1] + 0.016418f * x7[3][3][2] - 0.024538f * x7[3][3][3]
	 + 0.156465f * x7[3][3][4] - 0.022322f * x7[3][3][5] + 0.000696f * x7[3][3][6] - 0.169744f * x7[3][3][7];

	scores[1] = 0.000000f
	 + 0.091060f * x7[0][0][0] - 0.113722f * x7[0][0][1] + 0.154929f * x7[0][0][2] - 0.073843f * x7[0][0][3]
	 + 0.145265f * x7[0][0][4] - 0.005027f * x7[0][0][5] + 0.211612f * x7[0][0][6] + 0.097758f * x7[0][0][7]
	 + 0.130391f * x7[0][1][0] + 0.156439f * x7[0][1][1] + 0.134578f * x7[0][1][2] - 0.211030f * x7[0][1][3]
	 + 0.044114f * x7[0][1][4] - 0.200288f * x7[0][1][5] - 0.143407f * x7[0][1][6] + 0.099952f * x7[0][1][7]
	 - 0.064804f * x7[0][2][0] - 0.049543f * x7[0][2][1] - 0.104000f * x7[0][2][2] + 0.186944f * x7[0][2][3]
	 - 0.193002f * x7[0][2][4] + 0.150835f * x7[0][2][5] + 0.107711f * x7[0][2][6] - 0.170074f * x7[0][2][7]
	 + 0.026900f * x7[0][3][0] - 0.181479f * x7[0][3][1] - 0.095822f * x7[0][3][2] + 0.017965f * x7[0][3][3]
	 - 0.071563f * x7[0][3][4] + 0.027226f * x7[0][3][5] + 0.068597f * x7[0][3][6] - 0.020885f * x7[0][3][7]
	 + 0.052184f * x7[1][0][0] - 0.077212f * x7[1][0][1] + 0.127493f * x7[1][0][2] - 0.062192f * x7[1][0][3]
	 + 0.172246f * x7[1][0][4] + 0.019141f * x7[1][0][5] - 0.082164f * x7[1][0][6] - 0.154688f * x7[1][0][7]
	 + 0.105282f * x7[1][1][0] + 0.029955f * x7[1][1][1] + 0.198299f * x7[1][1][2] + 0.206949f * x7[1][1][3]
	 + 0.100340f * x7[1][1][4] - 0.091961f * x7[1][1][5] - 0.190819f * x7[1][1][6] + 0.035579f * x7[1][1][7]
	 + 0.079237f * x7[1][2][0] + 0.079138f * x7[1][2][1] - 0.059454f * x7[1][2][2] - 0.023292f * x7[1][2][3]
	 - 0.072465f * x7[1][2][4] + 0.191539f * x7[1][2][5] + 0.063844f * x7[1][2][6] + 0.120544f * x7[1][2][7]
	 - 0.015579f * x7[1][3][0] - 0.040516f * x7[1][3][1] + 0.090096f * x7[1][3][2] + 0.025385f * x7[1][3][3]
	 + 0.174996f * x7[1][3][4] - 0.044004f * x7[1][3][5] - 0.151021f * x7[1][3][6] - 0.096961f * x7[1][3][7]
	 + 0.053977f * x7[2][0][0] - 0.147423f * x7[2][0][1] - 0.125053f * x7[2][0][2] - 0.005692f * x7[2][0][3]
	 - 0.189552f * x7[2][0][4] + 0.092026f * x7[2][0][5] + 0.037478f * x7[2][0][6] - 0.128237f * x7[2][0][7]
	 - 0.120307f * x7[2][1][0] - 0.174779f * x7[2][1][1] - 0.138613f * x7[2][1][2] - 0.043368f * x7[2][1][3]
	 - 0.129065f * x7[2][1][4] + 0.181718f * x7[2][1][5] + 0.210246f * x7[2][1][6] - 0.165300f * x7[2][1][7]
	 - 0.048884f * x7[2][2][0] - 0.166500f * x7[2][2][1] - 0.143861f * x7[2][2][2] - 0.043028f * x7[2][2][3]
	 + 0.208448f * x7[2][2][4] - 0.003513f * x7[2][2][5] - 0.006180f * x7[2][2][6] - 0.135535f * x7[2][2][7]
	 + 0.001736f * x7[2][3][0] + 0.016485f * x7[2][3][1] + 0.096575f * x7[2][3][2] - 0.183662f * x7[2][3][3]
	 - 0.191423f * x7[2][3][4] + 0.145891f * x7[2][3][5] - 0.178014f * x7[2][3][6] + 0.014786f * x7[2][3][7]
	 - 0.173356f * x7[3][0][0] - 0.034586f * x7[3][0][1] + 0.057396f * x7[3][0][2] - 0.149297f * x7[3][0][3]
	 - 0.006369f * x7[3][0][4] - 0.202064f * x7[3][0][5] + 0.063721f * x7[3][0][6] - 0.066688f * x7[3][0][7]
	 - 0.204076f * x7[3][1][0] - 0.117385f * x7[3][1][1] + 0.125546f * x7[3][1][2] + 0.129241f * x7[3][1][3]
	 + 0.068140f * x7[3][1][4] + 0.161873f * x7[3][1][5] + 0.096385f * x7[3][1][6] + 0.093198f * x7[3][1][7]
	 - 0.066323f * x7[3][2][0] + 0.200854f * x7[3][2][1] - 0.111984f * x7[3][2][2] + 0.162278f * x7[3][2][3]
	 - 0.021539f * x7[3][2][4] + 0.161469f * x7[3][2][5] + 0.212767f * x7[3][2][6] - 0.176628f * x7[3][2][7]
	 + 0.177777f * x7[3][3][0] - 0.041745f * x7[3][3][1] - 0.027622f * x7[3][3][2] - 0.150493f * x7[3][3][3]
	 - 0.147937f * x7[3][3][4] + 0.108887f * x7[3][3][5] - 0.173657f * x7[3][3][6] - 0.145186f * x7[3][3][7];

	scores[2] = 0.000000f
	 + 0.028894f * x7[0][0][0] - 0.056162f * x7[0][0][1] - 0.056742f * x7[0][0][2] - 0.182361f * x7[0][0][3]
	 + 0.188333f * x7[0][0][4] - 0.178829f * x7[0][0][5] + 0.196023f * x7[0][0][6] + 0.045970f * x7[0][0][7]
	 - 0.084559f * x7[0][1][0] - 0.109035f * x7[0][1][1] + 0.092319f * x7[0][1][2] - 0.192823f * x7[0][1][3]
	 - 0.173781f * x7[0][1][4] + 0.021874f * x7[0][1][5] - 0.168663f * x7[0][1][6] - 0.103801f * x7[0][1][7]
	 - 0.105171f * x7[0][2][0] + 0.049093f * x7[0][2][1] + 0.070479f * x7[0][2][2] + 0.198277f * x7[0][2][3]
	 - 0.152108f * x7[0][2][4] + 0.035743f * x7[0][2][5] + 0.164512f * x7[0][2][6] + 0.062525f * x7[0][2][7]
	 - 0.014930f * x7[0][3][0] + 0.061603f * x7[0][3][1] + 0.154719f * x7[0][3][2] - 0.088560f * x7[0][3][3]
	 - 0.002436f * x7[0][3][4] - 0.194028f * x7[0][3][5] + 0.068549f * x7[0][3][6] + 0.046934f * x7[0][3][7]
	 + 0.151208f * x7[1][0][0] - 0.045455f * x7[1][0][1] - 0.169466f * x7[1][0][2] - 0.185397f * x7[1][0][3]
	 + 0.002775f * x7[1][0][4] + 0.202409f * x7[1][0][5] + 0.102599f * x7[1][0][6] - 0.060425f * x7[1][0][7]
	 - 0.022432f * x7[1][1][0] - 0.076398f * x7[1][1][1] - 0.129394f * x7[1][1][2] - 0.027282f * x7[1][1][3]
	 + 0.008329f * x7[1][1][4] + 0.013124f * x7[1][1][5] - 0.168790f * x7[1][1][6] - 0.182893f * x7[1][1][7]
	 - 0.181534f * x7[1][2][0] + 0.031498f * x7[1][2][1] - 0.058976f * x7[1][2][2] - 0.002974f * x7[1][2][3]
	 + 0.210370f * x7[1][2][4] - 0.102720f * x7[1][2][5] - 0.039409f * x7[1][2][6] - 0.141352f * x7[1][2][7]
	 + 0.176331f * x7[1][3][0] - 0.019341f * x7[1][3][1] + 0.080919f * x7[1][3][2] + 0.194638f * x7[1][3][3]
	 + 0.148050f * x7[1][3][4] + 0.032643f * x7[1][3][5] + 0.090079f * x7[1][3][6] - 0.033663f * x7[1][3][7]
	 + 0.126943f * x7[2][0][0] + 0.051300f * x7[2][0][1] + 0.057821f * x7[2][0][2] - 0.092791f * x7[2][0][3]
	 + 0.163390f * x7[2][0][4] + 0.095232f * x7[2][0][5] + 0.013911f * x7[2][0][6] + 0.005255f * x7[2][0][7]
	 + 0.204733f * x7[2][1][0] - 0.181024f * x7[2][1][1] + 0.063552f * x7[2][1][2] + 0.079481f * x7[2][1][3]
	 + 0.200577f * x7[2][1][4] - 0.009516f * x7[2][1][5] + 0.105763f * x7[2][1][6] - 0.038120f * x7[2][1][7]
	 + 0.199553f * x7[2][2][0] - 0.179366f * x7[2][2][1] + 0.015497f * x7[2][2][2] + 0.147290f * x7[2][2][3]
	 + 0.083584f * x7[2][2][4] - 0.204775f * x7[2][2][5] + 0.085095f * x7[2][2][6] + 0.078254f * x7[2][2][7]
	 + 0.157838f * x7[2][3][0] - 0.209301f * x7[2][3][1] - 0.052455f * x7[2][3][2] + 0.093839f * x7[2][3][3]
	 + 0.023354f * x7[2][3][4] - 0.209653f * x7[2][3][5] + 0.205043f * x7[2][3][6] + 0.134389f * x7[2][3][7]
	 + 0.147591f * x7[3][0][0] - 0.197434f * x7[3][0][1] - 0.159794f * x7[3][0][2] - 0.139798f * x7[3][0][3]
	 - 0.077507f * x7[3][0][4] + 0.178782f * x7[3][0][5] + 0.048389f * x7[3][0][6] - 0.041928f * x7[3][0][7]
	 - 0.151128f * x7[3][1][0] + 0.007552f * x7[3][1][1] + 0.199518f * x7[3][1][2] - 0.189613f * x7[3][1][3]
	 - 0.117927f * x7[3][1][4] - 0.024108f * x7[3][1][5] + 0.012664f * x7[3][1][6] + 0.053239f * x7[3][1][7]
	 + 0.124938f * x7[3][2][0] + 0.182939f * x7[3][2][1] + 0.010367f * x7[3][2][2] - 0.182505f * x7[3][2][3]
	 + 0.041420f * x7[3][2][4] + 0.064635f * x7[3][2][5] - 0.013318f * x7[3][2][6] - 0.187440f * x7[3][2][7]
	 + 0.053177f * x7[3][3][0] + 0.162954f * x7[3][3][1] + 0.176589f * x7[3][3][2] - 0.151377f * x7[3][3][3]
	 - 0.062828f * x7[3][3][4] + 0.209276f * x7[3][3][5] + 0.074847f * x7[3][3][6] + 0.143267f * x7[3][3][7];

	scores[3] = 0.000000f
	 - 0.098987f * x7[0][0][0] - 0.213111f * x7[0][0][1] + 0.184455f * x7[0][0][2] + 0.112728f * x7[0][0][3]
	 - 0.001876f * x7[0][0][4] + 0.109375f * x7[0][0][5] - 0.068365f * x7[0][0][6] + 0.087160f * x7[0][0][7]
	 - 0.019697f * x7[0][1][0] - 0.022046f * x7[0][1][1] - 0.008647f * x7[0][1][2] - 0.088645f * x7[0][1][3]
	 + 0.035057f * x7[0][1][4] - 0.014762f * x7[0][1][5] + 0.177457f * x7[0][1][6] - 0.006086f * x7[0][1][7]
	 - 0.085388f * x7[0][2][0] + 0.115766f * x7[0][2][1] + 0.024921f * x7[0][2][2] - 0.205493f * x7[0][2][3]
	 + 0.113797f * x7[0][2][4] + 0.035186f * x7[0][2][5] + 0.002177f * x7[0][2][6] - 0.027454f * x7[0][2][7]
	 - 0.119019f * x7[0][3][0] - 0.019188f * x7[0][3][1] - 0.052256f * x7[0][3][2] - 0.181641f * x7[0][3][3]
	 - 0.181476f * x7[0][3][4] - 0.142881f * x7[0][3][5] - 0.083202f * x7[0][3][6] - 0.125225f * x7[0][3][7]
	 - 0.109454f * x7[1][0][0] + 0.041550f * x7[1][0][1] + 0.031855f * x7[1][0][2] - 0.118774f * x7[1][0][3]
	 - 0.119039f * x7[1][0][4] - 0.167987f * x7[1][0][5] - 0.132570f * x7[1][0][6] - 0.080588f * x7[1][0][7]
	 + 0.127423f * x7[1][1][0] + 0.088632f * x7[1][1][1] - 0.033735f * x7[1][1][2] - 0.065504f * x7[1][1][3]
	 - 0.118331f * x7[1][1][4] - 0.033833f * x7[1][1][5] - 0.143570f * x7[1][1][6] + 0.003369f * x7[1][1][7]
	 + 0.154731f * x7[1][2][0] - 0.199369f * x7[1][2][1] + 0.068232f * x7[1][2][2] - 0.189397f * x7[1][2][3]
	 + 0.102886f * x7[1][2][4] - 0.209580f * x7[1][2][5] - 0.106645f * x7[1][2][6] - 0.026025f * x7[1][2][7]
	 - 0.112725f * x7[1][3][0] - 0.007575f * x7[1][3][1] + 0.009648f * x7[1][3][2] + 0.073223f * x7[1][3][3]
	 + 0.016428f * x7[1][3][4] + 0.057178f * x7[1][3][5] - 0.057019f * x7[1][3][6] - 0.169445f * x7[1][3][7]
	 - 0.209476f * x7[2][0][0] - 0.206026f * x7[2][0][1] - 0.113718f * x7[2][0][2] - 0.057411f * x7[2][0][3]
	 + 0.020232f * x7[2][0][4] + 0.123359f * x7[2][0][5] + 0.001107f * x7[2][0][6] - 0.015461f * x7[2][0][7]
	 + 0.010764f * x7[2][1][0] - 0.039632f * x7[2][1][1] - 0.197149f * x7[2][1][2] - 0.119436f * x7[2][1][3]
	 + 0.159504f * x7[2][1][4] + 0.085360f * x7[2][1][5] + 0.109155f * x7[2][1][6] - 0.097827f * x7[2][1][7]
	 + 0.116954f * x7[2][2][0] + 0.071486f * x7[2][2][1] - 0.130585f * x7[2][2][2] + 0.146661f * x7[2][2][3]
	 - 0.182793f * x7[2][2][4] + 0.108084f * x7[2][2][5] - 0.084742f * x7[2][2][6] + 0.130900f * x7[2][2][7]
	 - 0.200018f * x7[2][3][0] + 0.071395f * x7[2][3][1] - 0.043535f * x7[2][3][2] + 0.053102f * x7[2][3][3]
	 + 0.146073f * x7[2][3][4] - 0.026174f * x7[2][3][5] + 0.085138f * x7[2][3][6] + 0.181985f * x7[2][3][7]
	 + 0.068893f * x7[3][0][0] - 0.143340f * x7[3][0][1] + 0.152897f * x7[3][0][2] + 0.003749f * x7[3][0][3]
	 - 0.149822f * x7[3][0][4] - 0.026248f * x7[3][0][5] + 0.047294f * x7[3][0][6] + 0.106137f * x7[3][0][7]
	 + 0.037330f * x7[3][1][0] - 0.082928f * x7[3][1][1] + 0.060697f * x7[3][1][2] + 0.118000f * x7[3][1][3]
	 - 0.136577f * x7[3][1][4] - 0.063281f * x7[3][1][5] - 0.184192f * x7[3][1][6] + 0.036351f * x7[3][1][7]
	 + 0.079038f * x7[3][2][0] + 0.197127f * x7[3][2][1] - 0.101882f * x7[3][2][2] - 0.171611f * x7[3][2][3]
	 - 0.023822f * x7[3][2][4] + 0.021151f * x7[3][2][5] - 0.089027f * x7[3][2][6] + 0.092949f * x7[3][2][7]
	 - 0.201191f * x7[3][3][0] - 0.045936f * x7[3][3][1] + 0.044410f * x7[3][3][2] - 0.045946f * x7[3][3][3]
	 + 0.141615f * x7[3][3][4] - 0.202303f * x7[3][3][5] - 0.062238f * x7[3][3][6] - 0.019486f * x7[3][3][7];

}

void dummy_model2::predict(const BallCandidates::PatchYUVClassified& patch, double meanBrightnessOffset)
{
	ASSERT(patch.size() == 16);

	for(size_t x=0; x < patch.size(); x++) {
		for(size_t y=0; y < patch.size(); y++) {
			// TODO: check
			// .pixel.y accesses the brightness channel of the pixel
			// subtract the mean brightness calculated on the dataset and the offset from the module parameters
			float value = (static_cast<float>((patch.data[patch.size() * x + y].pixel.y)) / 255.0f) - 0.000000f - static_cast<float>(meanBrightnessOffset);
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
