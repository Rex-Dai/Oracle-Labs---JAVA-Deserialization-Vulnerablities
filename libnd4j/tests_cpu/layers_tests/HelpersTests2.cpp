/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
#include <helpers/EigenValsAndVecs.h>
#include <helpers/FullPivLU.h>
#include <helpers/HessenbergAndSchur.h>
#include <helpers/Sqrtm.h>
#include <ops/declarable/helpers/triangular_solve.h>

#include "testlayers.h"

using namespace sd;

class HelpersTests2 : public testing::Test {
 public:
  HelpersTests2() { std::cout << std::endl << std::flush; }
};

// #ifndef __CUDABLAS__
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_1) {
  NDArray x1('c', {1, 4}, {14, 17, 3, 1}, sd::DataType::DOUBLE);
  NDArray x2('c', {1, 1}, {14}, sd::DataType::DOUBLE);
  NDArray expQ('c', {1, 1}, {1}, sd::DataType::DOUBLE);

  ops::helpers::Hessenberg<double> hess1(x1);
  ASSERT_TRUE(hess1._H.isSameShape(&x1));
  ASSERT_TRUE(hess1._H.equalsTo(&x1));
  ASSERT_TRUE(hess1._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess1._Q.equalsTo(&expQ));

  ops::helpers::Hessenberg<double> hess2(x2);
  ASSERT_TRUE(hess2._H.isSameShape(&x2));
  ASSERT_TRUE(hess2._H.equalsTo(&x2));
  ASSERT_TRUE(hess2._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess2._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_2) {
  NDArray x('c', {2, 2}, {1.5, -2, 17, 5}, sd::DataType::DOUBLE);
  NDArray expQ('c', {2, 2}, {1, 0, 0, 1}, sd::DataType::DOUBLE);
  ops::helpers::Hessenberg<double> hess(x);



  x.printIndexedBuffer("expected x");
  hess._H.printIndexedBuffer("output h");


  expQ.printIndexedBuffer("expected q");
  hess._Q.printIndexedBuffer("output q");


  ASSERT_TRUE(hess._H.isSameShape(&x));
  ASSERT_TRUE(hess._H.equalsTo(&x));

  ASSERT_TRUE(hess._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_3) {
  NDArray x('c', {3, 3}, {33, 24, -48, 57, 12.5, -3, 1.1, 10, -5.2}, sd::DataType::DOUBLE);
  NDArray expH('c', {3, 3}, {33, -23.06939, -48.45414, -57.01061, 12.62845, 3.344058, 0, -9.655942, -5.328448},
               sd::DataType::DOUBLE);
  NDArray expQ('c', {3, 3}, {1, 0, 0, 0, -0.99981, -0.019295, 0, -0.019295, 0.99981}, sd::DataType::DOUBLE);

  ops::helpers::Hessenberg<double> hess(x);

  ASSERT_TRUE(hess._H.isSameShape(&expH));
  ASSERT_TRUE(hess._H.equalsTo(&expH));

  ASSERT_TRUE(hess._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_4) {
  NDArray x('c', {4, 4},
            {0.33, -7.25, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 7.59, 3.44, 2.24, -6.82, -1.15, 4.80, -4.67, 2.14},
            sd::DataType::DOUBLE);
  NDArray expH('c', {4, 4},
               {0.33, 0.4961181, 3.51599, 9.017665, -7.792702, 4.190221, 6.500328, 5.438888, 0, 3.646734, 0.4641911,
                -7.635502, 0, 0, 5.873535, 5.105588},
               sd::DataType::DOUBLE);
  NDArray expQ(
      'c', {4, 4},
      {1, 0, 0, 0, 0, -0.171956, 0.336675, -0.925787, 0, -0.973988, 0.0826795, 0.210976, 0, 0.147574, 0.937984, 0.3137},
      sd::DataType::DOUBLE);

  ops::helpers::Hessenberg<double> hess(x);

  ASSERT_TRUE(hess._H.isSameShape(&expH));
  ASSERT_TRUE(hess._H.equalsTo(&expH));

  ASSERT_TRUE(hess._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_5) {
  NDArray x('c', {10, 10},
            {6.9,  4.8,  9.5,  3.1,  6.5,  5.8,  -0.9, -7.3, -8.1, 3.0,  0.1,  9.9,  -3.2, 6.4,  6.2,  -7.0, 5.5,
             -2.2, -4.0, 3.7,  -3.6, 9.0,  -1.4, -2.4, 1.7,  -6.1, -4.2, -2.5, -5.6, -0.4, 0.4,  9.1,  -2.1, -5.4,
             7.3,  3.6,  -1.7, -5.7, -8.0, 8.8,  -3.0, -0.5, 1.1,  10.0, 8.0,  0.8,  1.0,  7.5,  3.5,  -1.8, 0.3,
             -0.6, -6.3, -4.5, -1.1, 1.8,  0.6,  9.6,  9.2,  9.7,  -2.6, 4.3,  -3.4, 0.0,  -6.7, 5.0,  10.5, 1.5,
             -7.8, -4.1, -5.3, -5.0, 2.0,  -4.4, -8.4, 6.0,  -9.4, -4.8, 8.2,  7.8,  5.2,  -9.5, -3.9, 0.2,  6.8,
             5.7,  -8.5, -1.9, -0.3, 7.4,  -8.7, 7.2,  1.3,  6.3,  -3.7, 3.9,  3.3,  -6.0, -9.1, 5.9},
            sd::DataType::DOUBLE);
  NDArray expH(
      'c', {10, 10},
      {
          6.9,      6.125208,  -8.070945, 7.219828, -9.363308,  2.181236,  5.995414,  3.892612,  4.982657,   -2.088574,
          -12.6412, 1.212547,  -6.449684, 5.162879, 0.4341714,  -5.278079, -2.624011, -2.03615,  11.39619,   -3.034842,
          0,        -12.71931, 10.1146,   6.494434, -1.062934,  5.668906,  -4.672953, -9.319893, -2.023392,  6.090341,
          0,        0,         7.800521,  -1.46286, 1.484626,   -10.58252, -3.492978, 2.42187,   5.470045,   1.877265,
          0,        0,         0,         14.78259, -0.3147726, -5.74874,  -0.377823, 3.310056,  2.242614,   -5.111574,
          0,        0,         0,         0,        -9.709131,  3.885072,  6.762626,  4.509144,  2.390195,   -4.991013,
          0,        0,         0,         0,        0,          8.126269,  -12.32529, 9.030151,  1.390931,   0.8634045,
          0,        0,         0,         0,        0,          0,         -12.99477, 9.574299,  -0.3098022, 4.910835,
          0,        0,         0,         0,        0,          0,         0,         14.75256,  18.95723,   -5.054717,
          0,        0,         0,         0,        0,          0,         0,         0,         -4.577715,  -5.440827,
      },
      sd::DataType::DOUBLE);
  NDArray expQ('c', {10, 10},
               {1, 0,          0,         0,        0,        0,         0,          0,         0,         0,
                0, -0.0079106, -0.38175,  -0.39287, -0.26002, -0.44102,  -0.071516,  0.12118,   0.64392,   0.057562,
                0, 0.28478,    0.0058784, 0.3837,   -0.47888, 0.39477,   0.0036847,  -0.24678,  0.3229,    0.47042,
                0, -0.031643,  -0.61277,  0.087648, 0.12014,  0.47648,   -0.5288,    0.060599,  0.021434,  -0.30102,
                0, 0.23732,    -0.17801,  -0.31809, -0.31267, 0.27595,   0.30134,    0.64555,   -0.33392,  0.13363,
                0, -0.023732,  -0.40236,  0.43089,  -0.38692, -0.5178,   -0.03957,   -0.081667, -0.47515,  -0.0077949,
                0, 0.20568,    -0.0169,   0.36962,  0.49669,  -0.22475,  -0.22199,   0.50075,   0.10454,   0.46112,
                0, 0.41926,    0.30243,   -0.3714,  -0.16795, -0.12969,  -0.67572,   -0.1205,   -0.26047,  0.10407,
                0, -0.41135,   -0.28357,  -0.33858, 0.18836,  0.083822,  -0.0068213, -0.30161,  -0.24956,  0.66327,
                0, 0.68823,    -0.33616,  -0.12129, 0.36163,  -0.063256, 0.34198,    -0.37564,  -0.048196, -0.058948},
               sd::DataType::DOUBLE);

  ops::helpers::Hessenberg<double> hess(x);

  ASSERT_TRUE(hess._H.isSameShape(&expH));
  ASSERT_TRUE(hess._H.equalsTo(&expH));

  ASSERT_TRUE(hess._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_1) {
  NDArray x('c', {3, 3}, sd::DataType::DOUBLE);

  NDArray expT('c', {3, 3}, {-2.5, -2, 1, 0, 1.5, -2, 3, 4, 5}, sd::DataType::DOUBLE);
  NDArray expU('c', {3, 3}, {0.3, 0.2, -0.1, 0, -0.1, 0.2, -0.3, -0.4, 0.5}, sd::DataType::DOUBLE);

  ops::helpers::Schur<double> schur(x);
  schur.t.linspace(-3, 1);
  schur.u.linspace(-0.3, 0.1);

  schur.splitTwoRows(1, 0.5);

  ASSERT_TRUE(schur.t.isSameShape(&expT));
  ASSERT_TRUE(schur.t.equalsTo(&expT));

  ASSERT_TRUE(schur.u.isSameShape(&expU));
  ASSERT_TRUE(schur.u.equalsTo(&expU));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_2) {
  NDArray x('c', {3, 3}, sd::DataType::DOUBLE);

  NDArray shift('c', {3}, sd::DataType::DOUBLE);
  NDArray exp1('c', {3}, {1, -3, 0}, sd::DataType::DOUBLE);
  NDArray exp2('c', {3}, {3, 3, -7}, sd::DataType::DOUBLE);
  NDArray exp3('c', {3}, {0.964, 0.964, 0.964}, sd::DataType::DOUBLE);
  NDArray exp1T('c', {3, 3}, {-3, -2, -1, 0, 1, 2, 3, 4, 5}, sd::DataType::DOUBLE);
  NDArray exp2T('c', {3, 3}, {-8, -2, -1, 0, -4, 2, 3, 4, 0}, sd::DataType::DOUBLE);
  NDArray exp3T('c', {3, 3},
                {
                    -9.464102,
                    -2,
                    -1,
                    0,
                    -5.464102,
                    2,
                    3,
                    4,
                    -1.464102,
                },
                sd::DataType::DOUBLE);

  ops::helpers::Schur<double> schur(x);
  // schur._U.linspace(-0.3, 0.1);    // doesn't matter

  schur.t.linspace(-3, 1);
  double expShift = 0;
  schur.calcShift(1, 5, expShift, shift);
  ASSERT_TRUE(schur.t.equalsTo(&exp1T));
  ASSERT_TRUE(shift.isSameShape(&exp1));
  ASSERT_TRUE(shift.equalsTo(&exp1));
  ASSERT_TRUE(expShift == 0);

  schur.t.linspace(-3, 1);
  expShift = 0;
  schur.calcShift(2, 10, expShift, shift);
  ASSERT_TRUE(schur.t.equalsTo(&exp2T));
  ASSERT_TRUE(shift.isSameShape(&exp2));
  ASSERT_TRUE(shift.equalsTo(&exp2));
  ASSERT_TRUE(expShift == 5);

  schur.t.linspace(-3, 1);
  expShift = 0;
  schur.calcShift(2, 30, expShift, shift);
  ASSERT_TRUE(schur.t.equalsTo(&exp3T));
  ASSERT_TRUE(shift.isSameShape(&exp3));
  ASSERT_TRUE(shift.equalsTo(&exp3));
  ASSERT_TRUE((6.4641 - 0.00001) < expShift && expShift < (6.4641 + 0.00001));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_3) {
  NDArray x('c', {2, 2}, {1.5, -2, 17, 5}, sd::DataType::DOUBLE);
  NDArray expU('c', {2, 2}, {1, 0, 0, 1}, sd::DataType::DOUBLE);

  ops::helpers::Schur<double> schur(x);

  ASSERT_TRUE(schur.t.isSameShape(&x));
  ASSERT_TRUE(schur.t.equalsTo(&x));

  ASSERT_TRUE(schur.u.isSameShape(&expU));
  ASSERT_TRUE(schur.u.equalsTo(&expU));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_4) {
  NDArray x('c', {3, 3}, {33, 24, -48, 57, 12.5, -3, 1.1, 10, -5.2}, sd::DataType::DOUBLE);
  NDArray expT('c', {3, 3}, {53.73337, -20.21406, -50.44809, 0, -27.51557, 26.74307, 0, 0, 14.0822},
               sd::DataType::DOUBLE);
  NDArray expU(
      'c', {3, 3},
      {-0.5848506, 0.7185352, 0.3763734, -0.7978391, -0.5932709, -0.1071558, -0.1462962, 0.3629555, -0.9202504},
      sd::DataType::DOUBLE);

  ops::helpers::Schur<double> schur(x);

  ASSERT_TRUE(schur.t.isSameShape(&expT));
  ASSERT_TRUE(schur.t.equalsTo(&expT));

  ASSERT_TRUE(schur.u.isSameShape(&expU));
  ASSERT_TRUE(schur.u.equalsTo(&expU));
}

/*
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_5) {

    NDArray x('c', {4,4}, {0.33 ,-7.25 ,1.71 ,6.20 ,1.34 ,5.38 ,-2.76 ,-8.51 ,7.59 ,3.44 ,2.24 ,-6.82 ,-1.15 ,4.80
,-4.67 ,2.14}, sd::DataType::DOUBLE); NDArray expT('c', {4,4},
{6.940177,7.201107,2.523849,-8.534745,-3.109643,5.289615,-2.940507,9.330303, 0,0,-0.1740346,   7.19851,0,0, -2.870214,
-1.965758}, sd::DataType::DOUBLE); NDArray expU('c', {4,4}, {-0.2602141,
0.8077556,-0.3352316,-0.4091935,0.3285353,-0.4395489,-0.4714875,-0.6903338,0.7536921, 0.3005626,-0.3910435,
0.4343908,-0.5062621, -0.252962,-0.7158242, 0.4090287}, sd::DataType::DOUBLE);

    ops::helpers::Schur<double> schur(x);

    ASSERT_TRUE(schur._T.isSameShape(&expT));
    ASSERT_TRUE(schur._T.equalsTo(&expT));

    ASSERT_TRUE(schur._U.isSameShape(&expU));
    ASSERT_TRUE(schur._U.equalsTo(&expU));
}
*/
/*
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_6) {

    NDArray x('c', {10,10}, {6.9 ,4.8 ,9.5 ,3.1 ,6.5 ,5.8 ,-0.9 ,-7.3 ,-8.1 ,3.0 ,0.1 ,9.9 ,-3.2 ,6.4 ,6.2 ,-7.0 ,5.5
,-2.2 ,-4.0 ,3.7 ,-3.6 ,9.0 ,-1.4 ,-2.4 ,1.7 , -6.1 ,-4.2 ,-2.5 ,-5.6 ,-0.4 ,0.4 ,9.1 ,-2.1 ,-5.4 ,7.3 ,3.6 ,-1.7 ,-5.7
,-8.0 ,8.8 ,-3.0 ,-0.5 ,1.1 ,10.0 ,8.0 ,0.8 ,1.0 ,7.5 ,3.5 ,-1.8 , 0.3 ,-0.6 ,-6.3 ,-4.5 ,-1.1 ,1.8 ,0.6 ,9.6 ,9.2 ,9.7
,-2.6 ,4.3 ,-3.4 ,0.0 ,-6.7 ,5.0 ,10.5 ,1.5 ,-7.8 ,-4.1 ,-5.3 ,-5.0 ,2.0 ,-4.4 ,-8.4 , 6.0 ,-9.4 ,-4.8 ,8.2 ,7.8 ,5.2
,-9.5 ,-3.9 ,0.2 ,6.8 ,5.7 ,-8.5 ,-1.9 ,-0.3 ,7.4 ,-8.7 ,7.2 ,1.3 ,6.3 ,-3.7 ,3.9 ,3.3 ,-6.0 ,-9.1 ,5.9},
sd::DataType::DOUBLE); NDArray expT('c', {10,10}, {-13.78982,  6.072464, 0.3021194,
-8.455495,-0.3047058,  4.033153,  2.610364,   2.80607, -2.735616, 0.3040549,-2.188506, -12.38324, -1.167179, -4.539672,
-19.08546,  1.752401,-0.1354974,-0.2747422,-0.3270464, -5.070936,
                                0,0,0.5067366,  7.930223,-0.6465996,  8.659522,  1.283713,  4.551415,   12.7736,    3.4812,0,0,-9.858142,
-2.905068, -6.474159, -6.247967, 0.4720073, -10.49523,  3.617189, -4.941627, 0,0,0,0,9.461626,
-4.896166,  9.339704,  4.640336,   16.8626,  2.056027,0,0,0,0,6.479812,  8.462862,  7.386285, -4.123457, -5.817095,
-2.633641,0,0,0,0,0,0,13.46667, -4.907281,  4.602204,  5.198035,
                                0,0,0,0,0,0, 7.176822,  16.93311,  2.195036,  1.346086,0,0,0,0,0,0,0,0, 16.86979,
-3.052473,0,0,0,0,0,0,0,0,0, -5.52268}, sd::DataType::DOUBLE);

    // NDArray expT('c', {10,10}, {-13.78982,  6.072464, 0.1926198,
-8.458698,-0.3047363,  4.033151,  2.610336,  2.806096, -2.735616, 0.3040549,-2.188506, -12.38324, -1.225857,  -4.52418,
-19.08548,  1.752257,-0.1354946,-0.2747435,-0.3270464, -5.070936,
    //                             0,0,
0.4812058,  7.886377,-0.7304318,  8.577898,  1.289673,  4.415163,  12.81936,  3.416929,0,0, -9.901988, -2.879537,
-6.465196, -6.359608,  0.455452, -10.55328,  3.451505, -4.986284,
    //                             0,0,0,0,  9.461614,
-4.896159,  9.339602,   4.64046,  16.86265,  2.056047,0,0,0,0,   6.47982,  8.462874,  7.386396, -4.123349, -5.816967,
-2.633626,
    //                             0,0,0,0,0,0, 13.46665,
-4.907315,  4.602182,  5.198022,0,0,0,0,0,0, 7.176788,  16.93313,  2.195081,  1.346137,0,0,0,0,0,0,0,0,  16.86979,
-3.052473,0,0,0,0,0,0,0,0,0,  -5.52268}, sd::DataType::DOUBLE);

    NDArray expU('c', {10,10}, {0.1964177,  0.2165192, -0.2138164,  0.4083154, -0.1872303, -0.5087223,  0.5529025,
-0.2996174,-0.08772947, 0.07126534,-0.1906247,  -0.223588,  0.3574755,  0.4245914, -0.3885589,-0.07328949, -0.4176507,
-0.1885168, -0.4476957,  0.1971104, -0.2219015,  0.3084187,  0.1069209, -0.4905009, -0.3517786,  0.1446875,   0.121738,
-0.3772941,  0.1232591,  0.5353205,-0.4766346,  0.6158252, -0.1529085, 0.04780914,  0.1274182, -0.1219211, -0.3123289,
-0.2219282,-0.07613826,  -0.429201, 0.2577533, -0.3356205,  -0.225358, -0.1540796,  0.3155174, -0.1904664, -0.3567101,
-0.6831458,  0.1244646, 0.03383783,  -0.45597, -0.3350697, 0.06824276, -0.2861978,-0.06724917, -0.7046481, 0.01664764,
0.2270567,  0.2003283,-0.01544937, 0.122865,  0.1516775, -0.4446453, -0.2338583,  0.1633447,  -0.193498,  -0.198088,
0.3170272, -0.5869794,  0.4013553,  0.347383,  0.3666581,  0.6890763,-0.05797414,  0.3630058,  -0.319958, -0.1071812,
0.06162044, 0.03171228,  0.1275262, -0.2986812, 0.05382598, -0.1484276,  0.4936468,   0.362756, 0.05858297, -0.1055183,
0.1090384,  0.4217073,  0.5534347, 0.3864388,  0.2085926,  -0.204135, 0.05230855, -0.5290207, -0.1548485, -0.4670302,
0.2205726,  0.4380318,-0.01626632}, sd::DataType::DOUBLE);

    ops::helpers::Schur<double> schur(x);

    ASSERT_TRUE(schur._T.isSameShape(&expT));
    ASSERT_TRUE(schur._T.equalsTo(&expT, 1e-3));

    ASSERT_TRUE(schur._U.isSameShape(&expU));
    ASSERT_TRUE(schur._U.equalsTo(&expU));
}
*/

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, EigenValsAndVecs_1) {
  NDArray x('c', {2, 2}, {1.5, -2, 17, 5}, sd::DataType::DOUBLE);
  NDArray expVals('c', {2, 2}, {3.25, 5.562149, 3.25, -5.562149}, sd::DataType::DOUBLE);
  NDArray expVecs('c', {2, 2, 2}, {-0.3094862, -0.0973726, -0.3094862, 0.0973726, 0, 0.9459053, 0, -0.9459053},
                  sd::DataType::DOUBLE);

  ops::helpers::EigenValsAndVecs<double> eig(x);

  ASSERT_TRUE(eig._Vals.isSameShape(&expVals));
  ASSERT_TRUE(eig._Vals.equalsTo(&expVals));

  ASSERT_TRUE(eig._Vecs.isSameShape(&expVecs));
  ASSERT_TRUE(eig._Vecs.equalsTo(&expVecs));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, EigenValsAndVecs_2) {
  NDArray x('c', {3, 3}, {33, 24, -48, 57, 12.5, -3, 1.1, 10, -5.2}, sd::DataType::DOUBLE);
  NDArray expVals('c', {3, 2}, {53.73337, 0, -27.51557, 0, 14.0822, 0}, sd::DataType::DOUBLE);
  NDArray expVecs('c', {3, 3, 2},
                  {-0.5848506, 0, 0.5560778, 0, -0.04889745, 0, -0.7978391, 0, -0.7683444, 0, -0.8855156, 0, -0.1462962,
                   0, 0.3168979, 0, -0.4620293, 0},
                  sd::DataType::DOUBLE);

  ops::helpers::EigenValsAndVecs<double> eig(x);

  ASSERT_TRUE(eig._Vals.isSameShape(&expVals));
  ASSERT_TRUE(eig._Vals.equalsTo(&expVals));

  ASSERT_TRUE(eig._Vecs.isSameShape(&expVecs));
  ASSERT_TRUE(eig._Vecs.equalsTo(&expVecs));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, EigenValsAndVecs_3) {
  NDArray x('c', {4, 4},
            {0.33, -7.25, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 7.59, 3.44, 2.24, -6.82, -1.15, 4.80, -4.67, 2.14},
            sd::DataType::DOUBLE);
  NDArray expVals('c', {4, 2}, {6.114896, 4.659591, 6.114896, -4.659591, -1.069896, 4.45631, -1.069896, -4.45631},
                  sd::DataType::DOUBLE);
  NDArray expVecs('c', {4, 4, 2},
                  {-0.2141303, 0.4815241,  -0.2141303, -0.4815241,  0.1035092,  -0.4270603, 0.1035092,  0.4270603,
                   0.2703519,  -0.2892722, 0.2703519,  0.2892722,   -0.5256817, 0.044061,   -0.5256817, -0.044061,
                   0.6202137,  0.05521234, 0.6202137,  -0.05521234, -0.5756007, 0.3932209,  -0.5756007, -0.3932209,
                   -0.4166034, -0.0651337, -0.4166034, 0.0651337,   -0.1723716, 0.1138941,  -0.1723716, -0.1138941},
                  sd::DataType::DOUBLE);

  ops::helpers::EigenValsAndVecs<double> eig(x);

  ASSERT_TRUE(eig._Vals.isSameShape(&expVals));
  ASSERT_TRUE(eig._Vals.equalsTo(&expVals));

  ASSERT_TRUE(eig._Vecs.isSameShape(&expVecs));
  ASSERT_TRUE(eig._Vecs.equalsTo(&expVecs));
}

/*
///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, EigenValsAndVecs_4) {

    NDArray x('c', {10,10}, {6.9 ,4.8 ,9.5 ,3.1 ,6.5 ,5.8 ,-0.9 ,-7.3 ,-8.1 ,3.0 ,0.1 ,9.9 ,-3.2 ,6.4 ,6.2 ,-7.0 ,5.5
,-2.2 ,-4.0 ,3.7 ,-3.6 ,9.0 ,-1.4 ,-2.4 ,1.7 , -6.1 ,-4.2 ,-2.5 ,-5.6 ,-0.4 ,0.4 ,9.1 ,-2.1 ,-5.4 ,7.3 ,3.6 ,-1.7 ,-5.7
,-8.0 ,8.8 ,-3.0 ,-0.5 ,1.1 ,10.0 ,8.0 ,0.8 ,1.0 ,7.5 ,3.5 ,-1.8 , 0.3 ,-0.6 ,-6.3 ,-4.5 ,-1.1 ,1.8 ,0.6 ,9.6 ,9.2 ,9.7
,-2.6 ,4.3 ,-3.4 ,0.0 ,-6.7 ,5.0 ,10.5 ,1.5 ,-7.8 ,-4.1 ,-5.3 ,-5.0 ,2.0 ,-4.4 ,-8.4 , 6.0 ,-9.4 ,-4.8 ,8.2 ,7.8 ,5.2
,-9.5 ,-3.9 ,0.2 ,6.8 ,5.7 ,-8.5 ,-1.9 ,-0.3 ,7.4 ,-8.7 ,7.2 ,1.3 ,6.3 ,-3.7 ,3.9 ,3.3 ,-6.0 ,-9.1 ,5.9},
sd::DataType::DOUBLE); NDArray expVals('c', {10,2}, { -13.08653,3.577011,-13.08653,-3.577011,
-1.199166,8.675665,-1.199166,-8.675665,8.962244,
                                5.610424, 8.962244,-5.610424,  15.19989,5.675794, 15.19989,-5.675794,16.86979,0,-5.52268,0},
sd::DataType::DOUBLE); NDArray expVecs('c', {10,10,2}, {0.1652385,0.1439317,   0.1652385,-0.1439317, -0.198272,0.207306,
-0.198272,-0.207306,   0.1861466,-0.4599919,    0.1861466,0.4599919,  0.09384053,-0.4889922,   0.09384053,0.4889922,
-0.6153314,0, -0.2180209,0, -0.1603652,-0.1466119,   -0.1603652,0.1466119,    0.2817409,0.3301842, 0.2817409,-0.3301842,
0.09747303,-0.2218182,   0.09747303,0.2218182,   0.2318273,-0.3355113,    0.2318273,0.3355113, -0.4828878,0,
-0.1451126,0, -0.1866771,0.1220412,  -0.1866771,-0.1220412,  0.08937842,-0.3025104,   0.08937842,0.3025104,
0.2783766,0.2258364,   0.2783766,-0.2258364, -0.1413997,-0.09596012,  -0.1413997,0.09596012, -0.2286925,0,  0.3290011,0,
                        -0.4009741,0.238131,   -0.4009741,-0.238131,  -0.02772353,0.1338458, -0.02772353,-0.1338458,
0.09030543,-0.2222453,   0.09030543,0.2222453,   0.2565825,-0.2275446,    0.2565825,0.2275446, -0.2855937,0,
-0.3950544,0, 0.2168379,-0.1301121,    0.2168379,0.1301121,   -0.165433,-0.1220125,    -0.165433,0.1220125,
-0.2685605,0.008133055,-0.2685605,-0.008133055,   0.1929395,-0.1194659,    0.1929395,0.1194659,  0.2206467,0,
0.3289105,0, -0.3835898,-0.2478813,   -0.3835898,0.2478813,  0.1923005,-0.01036433,   0.1923005,0.01036433,
-0.1711637,-0.3548358,   -0.1711637,0.3548358,   0.2888441,0.09625169,  0.2888441,-0.09625169,  0.2595426,0,
-0.1288072,0, 0.1033616,0.09839151,  0.1033616,-0.09839151,  -0.3080167,-0.1624564,
-0.3080167,0.1624564,-0.03972293,-0.03967309, -0.03972293,0.03967309,    0.1965443,0.3025898,   0.1965443,-0.3025898,
0.04587166,0,   0.499261,0, 0.2922398,0.2461792,   0.2922398,-0.2461792,   0.2769633,-0.2745029,    0.2769633,0.2745029,
0.1034687,-0.002947149,  0.1034687,0.002947149,  -0.02611308,0.1658046, -0.02611308,-0.1658046,  0.2351063,0,
-0.3787892,0, -0.2512689,-0.02169855,  -0.2512689,0.02169855,  -0.01481625,0.4376404, -0.01481625,-0.4376404,
-0.2298635,-0.2360671,   -0.2298635,0.2360671,     0.11004,-0.1467444,      0.11004,0.1467444,  0.1501568,0, 0.340117,0,
                        0.325096,0.1712822,    0.325096,-0.1712822, -0.2412035,-0.09236849,  -0.2412035,0.09236849,
0.3894343,-0.08673087,   0.3894343,0.08673087,   0.3125305,0.07128152,  0.3125305,-0.07128152, -0.2415555,0,
0.1841298,0,}, sd::DataType::DOUBLE);

    ops::helpers::EigenValsAndVecs<double> eig(x);

    ASSERT_TRUE(eig._Vals.isSameShape(&expVals));
    ASSERT_TRUE(eig._Vals.equalsTo(&expVals));

    ASSERT_TRUE(eig._Vecs.isSameShape(&expVecs));
    ASSERT_TRUE(eig._Vecs.equalsTo(&expVecs));
}
*/

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, fullPivLU_1) {
  NDArray a('c', {4, 4},
            {0.33, -7.25, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 7.59, 3.44, 2.24, -6.82, -1.15, 4.80, -4.67, 2.14},
            sd::DataType::DOUBLE);
  NDArray b('c', {4, 1}, {-5., 10, 9, 1}, sd::DataType::DOUBLE);

  NDArray x = b.ulike();

  NDArray expX('c', {4, 1}, {0.8527251, -0.2545784, -1.076495, -0.8526268}, sd::DataType::DOUBLE);

  ops::helpers::FullPivLU<double>::solve(a, b, x);

  ASSERT_TRUE(x.equalsTo(&expX));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, fullPivLU_2) {
  NDArray a('c', {4, 4},
            {0.33, -7.25, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 7.59, 3.44, 2.24, -6.82, -1.15, 4.80, -4.67, 2.14},
            sd::DataType::DOUBLE);
  NDArray b('c', {4, 2}, {-5., 10, 9, 1, 1.5, -2, 17, 5}, sd::DataType::DOUBLE);

  NDArray x = b.ulike();

  NDArray expX('c', {4, 2}, {1.462913, 1.835338, 0.4083664, -2.163816, -3.344481, -3.739225, 0.5156383, 0.01624954},
               sd::DataType::DOUBLE);

  ops::helpers::FullPivLU<double>::solve(a, b, x);

  ASSERT_TRUE(x.equalsTo(&expX));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, fullPivLU_3) {
  NDArray a1('c', {4, 3}, {0.33, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 2.24, -6.82, 4.80, -4.67, 2.14},
             sd::DataType::DOUBLE);
  NDArray a2('c', {3, 4}, {0.33, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 2.24, -6.82, 4.80, -4.67, 2.14},
             sd::DataType::DOUBLE);
  NDArray b1('c', {4, 2}, {-5., 10, 9, 1, 1.5, -2, 17, 5}, sd::DataType::DOUBLE);
  NDArray b2('c', {3, 2}, {-5., 10, 9, 1, 1.5, -2}, sd::DataType::DOUBLE);

  NDArray expX1('c', {3, 2}, {0.9344955, -0.5841325, 0.8768102, 1.029137, -1.098021, 1.360152}, sd::DataType::DOUBLE);
  NDArray expX2('c', {4, 2}, {0.3536033, 0.5270184, 0, 0, -0.8292221, 0.967515, 0.01827441, 2.856337},
                sd::DataType::DOUBLE);

  NDArray x1 = expX1.ulike();
  ops::helpers::FullPivLU<double>::solve(a1, b1, x1);
  ASSERT_TRUE(x1.equalsTo(&expX1));

  NDArray x2 = expX2.ulike();
  ops::helpers::FullPivLU<double>::solve(a2, b2, x2);
  ASSERT_TRUE(x2.equalsTo(&expX2));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, fullPivLU_4) {
  NDArray a('c', {10, 10},
            {6.9,  4.8,  9.5,  3.1,  6.5,  5.8,  -0.9, -7.3, -8.1, 3.0,  0.1,  9.9,  -3.2, 6.4,  6.2,  -7.0, 5.5,
             -2.2, -4.0, 3.7,  -3.6, 9.0,  -1.4, -2.4, 1.7,  -6.1, -4.2, -2.5, -5.6, -0.4, 0.4,  9.1,  -2.1, -5.4,
             7.3,  3.6,  -1.7, -5.7, -8.0, 8.8,  -3.0, -0.5, 1.1,  10.0, 8.0,  0.8,  1.0,  7.5,  3.5,  -1.8, 0.3,
             -0.6, -6.3, -4.5, -1.1, 1.8,  0.6,  9.6,  9.2,  9.7,  -2.6, 4.3,  -3.4, 0.0,  -6.7, 5.0,  10.5, 1.5,
             -7.8, -4.1, -5.3, -5.0, 2.0,  -4.4, -8.4, 6.0,  -9.4, -4.8, 8.2,  7.8,  5.2,  -9.5, -3.9, 0.2,  6.8,
             5.7,  -8.5, -1.9, -0.3, 7.4,  -8.7, 7.2,  1.3,  6.3,  -3.7, 3.9,  3.3,  -6.0, -9.1, 5.9},
            sd::DataType::DOUBLE);
  NDArray b('c', {10, 2}, {-5., 10, 9, 1, 1.5, -2, 17, 5, 3.6, 0.12, -3.1, 2.27, -0.5, 27.3, 8.9, 5, -7, 8, -9, 10},
            sd::DataType::DOUBLE);

  NDArray x = b.ulike();

  NDArray expX('c', {10, 2}, {-0.697127, 2.58257,    2.109721,  3.160622,  -2.217796, -3.275736, -0.5752479,
                              2.475356,  1.996841,   -1.928947, 2.213154,  3.541014,  0.7104885, -1.981451,
                              -3.297972, -0.4720612, 3.672657,  0.9161028, -2.322383, -1.784493},
               sd::DataType::DOUBLE);

  ops::helpers::FullPivLU<double>::solve(a, b, x);

  ASSERT_TRUE(x.equalsTo(&expX));
}
