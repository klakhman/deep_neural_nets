#include "armadillo"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "TFFNet.h"

using namespace arma;
using namespace std;

int main(){
  ofstream ofs;
  ofs.open("C:/Coding/rand_net.txt");
  TFFNet rand_net;
  vector<int> neuronsDistribution;
  neuronsDistribution.push_back(2);
  neuronsDistribution.push_back(2);
  neuronsDistribution.push_back(1);
  rand_net.generateRandomNet(2, neuronsDistribution);
  rand_net.printNetwork(ofs);
  ofs.close();
  return 0;
}