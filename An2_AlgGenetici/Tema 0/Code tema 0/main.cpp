#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#define PMAX 100
#define INF 2000
using namespace std;

double DeJong(double[], int);
double SixHump(double[], int);
double Schwefel(double[], int);
double Rastrigin(double[], int);

double GetFunctionMin(double(*testFunction)(double[], int), int);
bool pValueIsGood(double(*testFunction)(double[], int), int, int);
void Initialize(double(*testFunction)(double[], int), double[], int);
double fRand(double fMin, double fMax);
void Ruleaza(double(*testFunction)(double[], int), int iteratii, int nrVals);

int main()
{
    srand(time(NULL));
    Ruleaza(Rastrigin, 10, 3);
    return 0;
}

double DeJong(double v[], int nrVals) {
	double functionValue = 0;
	for (int i = 0; i < nrVals; i++) functionValue += v[i] * v[i];
	return functionValue;
}

double Schwefel(double v[], int nrVals) {
    double functionValue = 0;
	for (int i = 0; i < nrVals; i++) functionValue += -v[i] * sin(sqrt(abs(v[i])));
	return functionValue;
}

double Rastrigin(double v[], int nrVals) {
    double functionValue = 10*nrVals;
    for(int i=0; i<nrVals; i++) functionValue += v[i]*v[i] - 10*cos(2 * 3.1415 * v[i]);
    return functionValue;
}

double SixHump(double v[], int nrVals) {
    double functionValue = 10*nrVals;
    if(nrVals != 2) {cout<<"SixHump has 2 parameters!\n"; return 0;}
    for(int i=0; i<nrVals; i++)
        functionValue = (4-2.1*pow(v[0],2)+pow(v[0],4)/3)*pow(v[0],2)+v[0]*v[1]+(-4+4*pow(v[1],2))*pow(v[1],2);
    return functionValue;
}

double GetFunctionMin(double(*testFunction)(double[], int), int nrVals) {
    double epsilon, p[PMAX], fVal1, fVal2, fMin = 999999, valPrec, valCurrent, fEpsilon = 0.00001, pAux;
    Initialize(testFunction, p, nrVals);
    for(int i=0; i<nrVals; i++) {
        epsilon = 0.00001;
        pAux = p[i];
        p[i]+=epsilon;
        fVal1 = testFunction(p, nrVals);
        p[i]-=2*epsilon;
        fVal2 = testFunction(p, nrVals);
        if(fVal1 > fVal2) epsilon = -epsilon;
        fMin = fMin > min(fVal1, fVal2) ? min(fVal1, fVal2) : fMin;
        p[i] = pAux;
        do {
            valPrec = testFunction(p, nrVals);
            p[i]+=epsilon;
            if(pValueIsGood(testFunction, p[i], i) == false) break;
            valCurrent = testFunction(p, nrVals);
            fMin = min(valCurrent, fMin);
        } while(valPrec - valCurrent > fEpsilon);
    }
    cout<<fMin<<" ";
    return fMin;
}

bool pValueIsGood(double(*testFunction)(double[], int), int value, int i) {
    if(testFunction == DeJong || testFunction == Rastrigin)
        if(value < -5.12 || value > 5.12) return false;
    if(testFunction == Schwefel)
        if(value < -500 || value > 500) return false;
    if(testFunction == SixHump) {
        if(i==0 && (value < -3 || value > 3)) return false;
        if(i==1 && (value < -2 || value > 2)) return false;
    }
    return true;
}

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void Initialize(double(*testFunction)(double[], int), double v[], int nrVals) {
    int i;
    if(testFunction == DeJong || testFunction == Rastrigin)
        for(i=0; i<nrVals; i++) v[i] = fRand(-5.12, 5.12);
    else if(testFunction == Schwefel)
        for(i=0; i<nrVals; i++) v[i] = fRand(-500, 500);
    else {
        v[0] = fRand(-3, 3);
        v[1] = fRand(-2, 2);
    }
}

void Ruleaza(double(*testFunction)(double[], int), int iteratii, int nrVals) {
    double mediaSol=0, bestSol = INF, worseSol = -INF, solution, timpMediu=0, timpMax = -INF, timpMin = INF, timp;
    for(int i=0; i<iteratii; i++) {
        auto start = std::chrono::system_clock::now();

        solution = GetFunctionMin(testFunction, nrVals);

        auto ending = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = ending-start;
        timp = elapsed_seconds.count();
        timpMediu += timp;
        if(timp < timpMin) timpMin = timp;
        if(timp > timpMax) timpMax = timp;

        mediaSol += solution;
        if(solution > worseSol) worseSol = solution;
        if(solution < bestSol) bestSol = solution;
    }
    mediaSol /= iteratii;
    timpMediu /= iteratii;
    cout<<"\nMedia solutiilor: "<<mediaSol<<endl;
    cout<<"Cea mai proasta solutie: "<<worseSol<<endl;
    cout<<"Cea mai buna solutie: "<<bestSol<<endl;
    cout<<"Timp mediu: "<<timpMediu<<endl;
    cout<<"Timp minim: "<<timpMin<<endl;
    cout<<"Timp maxim: "<<timpMax<<endl;
}
