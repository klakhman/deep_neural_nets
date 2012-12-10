#ifndef TFFNET_H
#define TFFNET_H

#include <iostream>
#include <vector>
#include "armadillo"

// Класс нейронной сети прямого распространения
class TFFNet{
  // Матрицы весов связей между слоями
  // Кол-во матриц layersQuantity. Первая матрица - это веса от входов к первому слою.
  // Веса для одного нейрона расположены в строках матриц, где нулевой элемент строки - это bias.
  std::vector< arma::Mat<double> > weightsMatrices;
  // Распределение нейронов по слоям - layersQuantity + 1, не включая bias (нулевой элемент указывает на кол-во входов в сеть)
  std::vector<int> neuronsDistribution;
  // Кол-во слоев в сети (не считая входной)
  int layersQuantity;
  // Текущие выходы нейронов во всех слоях
  std::vector< std::vector<double> > neuronsOutputs;

  // Функция выхода логистической функции (на векторе - то есть сразу для всего слоя нейронов)
  arma::Col<double> logisticFunc(const arma::Col<double>& inputCol, int slope = 2) const{
    return 1 / (exp(inputCol * -slope) + 1);
  }
  // Функция производной логистической функции (сразу для всего слоя нейронов - передаются текущие выходы нейронов)
  arma::Col<double> derLogisticFunc(const arma::Col<double>& inputCol, int slope = 2) const{
    return slope * ((1 - inputCol) % inputCol);
  }

  // Функция производной логистической функции на одном числе (передается текущий выход нейрона)
  double derLogisticFunc(double x, int slope = 2) const{
    return slope * (1 - x) * x;
  }

  // Подсчет локальных градиентов для всех нейронов сети (при алгоритме обратного распространения)
  std::vector< std::vector<double> > calculateLocalGradients(const std::vector<double>& desiredOutput) const;
  // Модификация весов сети - передается "матрица" локальных градиентов нейронов сети
  void modifyWeights(const std::vector< std::vector<double> >& localGradients, double learningRate = 0.1);


public:
  TFFNet(){
    layersQuantity = 0;
  }
  ~TFFNet(){}
  
  // Возвращает кол-во слоев (не учитывая входной)
  int getLayersQuantity() const{
    return layersQuantity;
  }

  //Возвращает кол-во нейронов в слое (нулевой слой - входной)
  int getNeuronsQuantity(int layer) const{
    if ((layer > layersQuantity) || (!layersQuantity)) {
      std::cout << "Error: Trying to access layer beyond the existing..." << std::endl;
      exit(1);
    }
    return neuronsDistribution[layer];
  }

  // Загрузка сети из файла
  void loadNetwork(std::istream& inputSource);
  // Выгрузка сети в файл
  void printNetwork(std::ostream& outputSource);
  // Генерация сети со случайными весами
  // layersQuantity - кол-во слоев, не считая входного; weightsInitValue - дипапазон рандомизации весов связей (-weightsInitValue; weightsInitValue)
  void generateRandomNet(int _layersQuantity, const std::vector<int>& _neuronsDistribution, double weightsInitValue = 0.1);

  // Обсчет сети (возвращает выход сети)
  std::vector<double> calculate(const std::vector<double>& inputVector);

  // Возвращает текущий выходной вектор (полученный после последнего обсчета)
  std::vector<double> getOutputVector() const{
    if (!layersQuantity){
      std::cout << "Error: Trying to access layer beyond the existing..." << std::endl;
      exit(1);
    }
    return neuronsOutputs[layersQuantity];
  }

};

#endif // TFFNET_H