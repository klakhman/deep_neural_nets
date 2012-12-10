#include "armadillo"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "TFFNet.h"

using namespace arma;
using namespace std;

int main(){
  TFFNet test_net;
  ifstream ifs;
  ifs.open("C:/Coding/test_net.txt");
  test_net.loadNetwork(ifs);
  ifs.close();
  vector<double> input;
  input.push_back(1);
  input.push_back(-1);
  vector<double> output = test_net.calculate(input);
  cout << output.size() << endl << output[0] << endl;
  //Mat<double> A;
  //vector<double> vec;
  //vec.push_back(1);
  //vec.push_back(2);
  //A = conv_to< rowvec >::from(vec);
  //A.print();
  //cout << 1/(1+exp(-2*(1+0.99752 - 0.98201)));
  return 0;
}