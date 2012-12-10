#include "TFFNet.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "armadillo"

using namespace std;
using namespace arma;

// Загрузка сети из файла
void TFFNet::loadNetwork(istream& inputSource){
  string tmpStr;
  inputSource >> tmpStr;
  layersQuantity = atoi(tmpStr.c_str());
  weightsMatrices.resize(layersQuantity);
  neuronsDistribution.resize(layersQuantity + 1);
  // Сначала заполняем распределение нейронов по слоям
  for (int currentLayer = 0; currentLayer <= layersQuantity; ++currentLayer){
    inputSource >> tmpStr;
    neuronsDistribution[currentLayer] = atoi(tmpStr.c_str());
  }
  // Заполняем матрицы весов
  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    weightsMatrices[currentLayer].set_size(neuronsDistribution[currentLayer + 1], neuronsDistribution[currentLayer] + 1);
    for (int currentPostNeuron = 1; currentPostNeuron <= neuronsDistribution[currentLayer + 1]; ++currentPostNeuron)
      for (int currentPreNeuron = 0; currentPreNeuron <= neuronsDistribution[currentLayer]; ++currentPreNeuron){
        inputSource >> tmpStr;
        weightsMatrices[currentLayer](currentPostNeuron - 1, currentPreNeuron) = atof(tmpStr.c_str());
      }
  }
  // Создаем структуру выходов нейронов
  neuronsOutputs.resize(layersQuantity + 1);
  for (int currentLayer = 0; currentLayer <= layersQuantity; ++currentLayer)
    neuronsOutputs[currentLayer].resize(neuronsDistribution[currentLayer], 0);
}

// Выгрузка сети в файл
void TFFNet::printNetwork(std::ostream& outputSource){
  outputSource << layersQuantity << endl;
  for (int currentLayer = 0; currentLayer <= layersQuantity; ++currentLayer)
    outputSource << neuronsDistribution[currentLayer] << "\t";
  outputSource << endl;
  // Записываем матрицы весов
  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    for (int currentPostNeuron = 1; currentPostNeuron <= neuronsDistribution[currentLayer + 1]; ++currentPostNeuron){
      for (int currentPreNeuron = 0; currentPreNeuron <= neuronsDistribution[currentLayer]; ++currentPreNeuron)
        outputSource << weightsMatrices[currentLayer](currentPostNeuron - 1, currentPreNeuron);
      outputSource << endl;
    }
    outputSource << endl;
  }
}

// Обсчет сети (возвращает выход сети)
vector<double> TFFNet::calculate(const vector<double>& inputVector){
  // Заполняем выходы нулевого слоя и подготавливаем нулевой вектор
  neuronsOutputs[0] = inputVector;
  // Вектор со входом bias
  vector<double> extraInputVector;
  extraInputVector.push_back(1);
  extraInputVector.insert(extraInputVector.end(), inputVector.begin(), inputVector.end());
  Col<double> inputToLayer = conv_to< Col<double> >::from(extraInputVector);

  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    Col<double> layerOutput = logisticFunc(weightsMatrices[currentLayer] * inputToLayer); 
    neuronsOutputs[currentLayer + 1] = conv_to< vector<double> >::from(layerOutput);
    // Заполняем вход для следующего слоя
    inputToLayer.resize(neuronsDistribution[currentLayer + 1] + 1);
    inputToLayer(0) = 1;
    inputToLayer.subvec(1, neuronsDistribution[currentLayer + 1]) = layerOutput;
  }
  return neuronsOutputs[layersQuantity];
}

