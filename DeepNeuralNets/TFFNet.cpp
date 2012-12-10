#include "TFFNet.h"
#include "TTrainingData.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "armadillo"

using namespace std;
//using namespace arma;

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
        outputSource << weightsMatrices[currentLayer](currentPostNeuron - 1, currentPreNeuron)  << "\t";
      outputSource << endl;
    }
    outputSource << endl;
  }
}

// Генерация сети со случайными весами
// layersQuantity - кол-во слоев, не считая входного; weightsInitValue - дипапазон рандомизации весов связей (-weightsInitValue; weightsInitValue)
void TFFNet::generateRandomNet(int _layersQuantity, const std::vector<int>& _neuronsDistribution, double weightsInitValue /*=0.1*/){
  layersQuantity = _layersQuantity;
  weightsMatrices.resize(layersQuantity);
  neuronsDistribution = _neuronsDistribution;

  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    weightsMatrices[currentLayer] = arma::randu< arma::Mat<double> >(neuronsDistribution[currentLayer + 1], neuronsDistribution[currentLayer] + 1);
    // Armagillo генерирует случаные матрицы в диапазоне [0, 1] - приводим к нужному нам диапазону
    weightsMatrices[currentLayer] = weightsMatrices[currentLayer] * (2 * weightsInitValue) -  weightsInitValue; 
  }
  // Создаем структуру выходов нейронов
  neuronsOutputs.resize(layersQuantity + 1);
  for (int currentLayer = 0; currentLayer <= layersQuantity; ++currentLayer)
    neuronsOutputs[currentLayer].resize(neuronsDistribution[currentLayer], 0);
}

// Обсчет сети (возвращает выход сети)
vector<double> TFFNet::calculate(const vector<double>& inputVector){
  // Заполняем выходы нулевого слоя и подготавливаем нулевой вектор
  neuronsOutputs[0] = inputVector;
  // Вектор со входом bias
  vector<double> extraInputVector;
  extraInputVector.push_back(1);
  extraInputVector.insert(extraInputVector.end(), inputVector.begin(), inputVector.end());
  arma::Col<double> inputToLayer = arma::conv_to< arma::Col<double> >::from(extraInputVector);

  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    arma::Col<double> layerOutput = logisticFunc(weightsMatrices[currentLayer] * inputToLayer); 
    neuronsOutputs[currentLayer + 1] = arma::conv_to< vector<double> >::from(layerOutput);
    // Заполняем вход для следующего слоя
    inputToLayer.resize(neuronsDistribution[currentLayer + 1] + 1);
    inputToLayer(0) = 1;
    inputToLayer.subvec(1, neuronsDistribution[currentLayer + 1]) = layerOutput;
  }
  return neuronsOutputs[layersQuantity];
}

// Подсчет локальных градиентов для всех нейронов сети (при алгоритме обратного распространения)
vector< vector<double> >  TFFNet::calculateLocalGradients(const vector<double>& desiredOutput) const{
  vector< vector<double> > localGradients(layersQuantity);
  //for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer)
  //  localGradients[currentLayer].resize(neuronsDistribution[currentLayer])
  // Вычисляем для выходного слоя
  for (int currentOutput = 0; currentOutput < neuronsDistribution[layersQuantity]; ++currentOutput)
    localGradients[layersQuantity - 1].push_back((desiredOutput[currentOutput] - neuronsOutputs[layersQuantity][currentOutput]) * 
      derLogisticFunc(neuronsOutputs[layersQuantity][currentOutput]));
  // Теперь вычисляем последовательно для всех слоев
  for (int currentLayer = layersQuantity - 1; currentLayer > 0; --currentLayer)
    // Получаем локальные градиенты транспонируя матрицу весов и перемножая с локальными градиентами следующего слоя, а потом каждый элемент с производной по выходу
    localGradients[currentLayer - 1] = arma::conv_to< vector<double> >::from(
      (strans(weightsMatrices[currentLayer]) * arma::conv_to< arma::Col<double> >::from(localGradients[currentLayer])) % 
      derLogisticFunc(arma::conv_to< arma::Col<double> >::from(neuronsOutputs[currentLayer])));
  return localGradients;
}

// Модификация весов сети - передается "матрица" локальных градиентов нейронов сети
void TFFNet::modifyWeights(const vector< vector<double> >& localGradients, double learningRate /*=0.1*/){
  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    arma::Row<double> outputs(neuronsDistribution[currentLayer + 1] + 1);
    outputs(0) = 1;
    outputs.subvec(1, neuronsDistribution[currentLayer + 1]) = arma::conv_to< arma::Row<double> >::from(neuronsOutputs[currentLayer + 1]);
    weightsMatrices[currentLayer] -= learningRate * (arma::conv_to< arma::Col<double> >::from(localGradients[currentLayer]) * outputs);
  }
}

