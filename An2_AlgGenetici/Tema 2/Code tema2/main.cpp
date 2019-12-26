#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <fstream>

#define INF 999999999
#define DMAX 500
#define NMAX 100
#define EPSILON 0.1
#define C 2.0
#define P_MUTATION 0.5
#define P_CROSS 0.3

using namespace std;
ofstream fout("date_experimentale.txt");
ofstream f("rezultate.txt");

double prob[DMAX], probCumulated[DMAX];

double Rastrigin(double[], int);
double DeJong(double[], int);
double Griewank(double[], int);
double Rosenbrock(double[], int);

double GeneticAlgorithm(double(*)(double[], int), int);
void GenerateRandomSolution(bool[][NMAX], int, int, int);
void CreateRandomChromosome(bool[], int nrVals, int);

void getInterval(double(*)(double[], int), double&, double&);
double RandomDouble(const double&, const double&);
void calculeazaParametrii(double(*)(double[], int), int, int, int&, int&);
void convertToDouble(double(*)(double[], int), bool[], double[], int, int, double, double);
bool PopulationIsEvoluating(int, int);
double EvaluatePopulation(double(*)(double[], int), bool[][NMAX], int, int, int, int, int);
void BuildPartialProbabilities(double, long double[], int);
void SelectNextPopulation(bool[][NMAX], int&, int);
double Fitness(double(*)(double[], int), double[], int);
void MutateChromosomes(bool[][NMAX], int, int, int);
void CrossChromosomes(bool[][NMAX], int, int, int);
void CrossIndividualChromosomes(bool[][NMAX], int, int, int, int);
void runFunction(double(*)(double[], int), int, int);

int main()
{
	srand(time(NULL));
	runFunction(DeJong, 1, 30);
	return 0;
}

double Rastrigin(double v[], int nrVals) { // -5.12 <= x <= 5.12
	double functionValue = 10 * nrVals;
	for (int i = 0; i<nrVals; i++) functionValue += v[i] * v[i] - 10 * cos(2 * 3.1415 * v[i]);
	return functionValue;
}

double DeJong(double v[], int nrVals) { // -5.12 <= x <= 5.12
	double functionValue = 0;
	for (int i = 0; i < nrVals; i++) functionValue += v[i] * v[i];
	return functionValue;
}

double Griewank(double v[], int nrVals) { // -600 <= x <= 600
	double a = 0, b = 1, functionValue;
	for (int i = 1; i <= nrVals; i++) {
		a += (v[i] * v[i]) / 4000;
		b *= cos(v[i] / sqrt(i));
	}
	functionValue = a - b + 1;
	return functionValue;
}

double Rosenbrock(double v[], int nrVals) { // -2.048 <= x <= 2.048
	double functionValue = 0;
	for (int i = 0; i<nrVals; i++) functionValue += 100 * pow(v[i + 1] - v[i] * v[i], 2) + pow(1 - v[i], 2);
	return functionValue;
}

double GeneticAlgorithm(double(*testFunction)(double[], int), int nrVals) {
	double v[DMAX], bestSol = INF, generationResult, a = 0, b = 0;
	int n, N, nrIterations = 100, populationSize = 100, currentGeneration, lastBestGeneration, precizie = 2;
	bool chromosomes[DMAX][NMAX];

    getInterval(testFunction, a, b);
    calculeazaParametrii(testFunction, precizie, nrVals, n, N);
	for (int i = 0; i < nrIterations; i++) {
		GenerateRandomSolution(chromosomes, populationSize, nrVals, n);
		currentGeneration = lastBestGeneration = 0;

		while (PopulationIsEvoluating(lastBestGeneration, currentGeneration)) {
			generationResult = EvaluatePopulation(testFunction, chromosomes, populationSize, nrVals, n, a, b);

			if (generationResult < bestSol) {
				bestSol = generationResult;
				lastBestGeneration = currentGeneration;
			}

			SelectNextPopulation(chromosomes, populationSize, nrVals);
			CrossChromosomes(chromosomes, populationSize, nrVals, n);
			MutateChromosomes(chromosomes, populationSize, nrVals, n);

			++currentGeneration;
		}
		fout << bestSol << endl;
	}
	return bestSol;
}

void GenerateRandomSolution(bool chromosomes[DMAX][NMAX], int populationSize, int nrVals, int n) {
	for (int i = 0; i < populationSize; i++) CreateRandomChromosome(chromosomes[i], nrVals, n);
}

void CreateRandomChromosome(bool chromosome[], int nrVals, int n) {
	for (int i = 0; i < n*nrVals; i++) chromosome[i] = rand() % 2;
}

double RandomDouble(const double &minVal, const double &maxVal) {
	double f;
	f = (double)rand() / RAND_MAX;
	return minVal + f * (maxVal - minVal);
}

void calculeazaParametrii(double(*testFunction)(double[], int), int d, int nrVals, int& n, int& N) {
	double a, b;
	getInterval(testFunction, a, b);
	N = (b - a)*pow(10, d); //subintervale egale
	n = ceil(log2(N)); // n biti pt fiecare nr
}

void getInterval(double(*testFunction)(double[], int), double &a, double &b) {
	if (testFunction == Rastrigin || DeJong) { a = -5.12; b = 5.12; }
	else if (testFunction == Griewank) { a = -600; b = 600; }
	else if (testFunction == Rosenbrock) { a = -2.048; b = 2.048; }
}

void convertToDouble(double(*testFunction)(double[], int), bool x[], double v[], int nrVals, int n, double a, double b) {
	int i, j, nr;
	for (i = 0; i<n*nrVals; i += n) {
		nr = 0;
		for (j = 0; j<n; j++) nr += pow(2, j)*x[i + j]; // convert to int
        v[i / n] = a + nr * (b - a) / (pow(2, n) - 1);
	}
}

bool PopulationIsEvoluating(int lastBestGeneration, int currentGeneration) {
	if (currentGeneration - lastBestGeneration > 1000) return false;
    // daca dupa 1000 generatii nu s-a imbunatatit rezultatul, inseamna ca populatia nu mai evolueaza
	return true;
}

double EvaluatePopulation(double(*testFunction)(double[], int), bool chromosomes[DMAX][NMAX], int populationSize, int nrVals, int n, int a, int b) {
	double doubleVals[DMAX];
	long double chromosomeFitness[DMAX], bestSol = INF, sumFitness = 0, functionValue;

	for (int i = 0; i < populationSize; ++i) {
		convertToDouble(testFunction, chromosomes[i], doubleVals, nrVals, n, a, b);

		chromosomeFitness[i] = Fitness(testFunction, doubleVals, nrVals); // fitness pt fiecare cromozom
		functionValue = testFunction(doubleVals, nrVals);
		bestSol = min(bestSol, functionValue);
		sumFitness += chromosomeFitness[i]; // fitness total
	}

	BuildPartialProbabilities(sumFitness, chromosomeFitness, populationSize);

	return bestSol;
}

double Fitness(double(*testFunction)(double[], int), double doubleVals[], int nrVals) {
	return 1 / (testFunction(doubleVals, nrVals) + EPSILON); // adun epsilon ca sa nu existe posibilitatea sa impart 1 la 0
}

void BuildPartialProbabilities(double sumFitness, long double chromosomeFitness[DMAX], int populationSize) {
	for (int i = 0; i < populationSize; ++i) prob[i] = chromosomeFitness[i] / sumFitness; // probabilitatea selectiei individuale
	probCumulated[0] = prob[0];
	for (int i = 1; i < populationSize; ++i) probCumulated[i] = probCumulated[i-1] + prob[i-1]; // probabilitatea selectiei cumulate
}

void SelectNextPopulation(bool chromosomes[][NMAX], int &populationSize, int nrVals) {
	double randomNr;
	int i, j = 0;
	bool chosen[DMAX], chosenChromosomes[DMAX][NMAX];
	bool auxChromosomes[DMAX][NMAX];
	memset(chosen, 0, sizeof(chosen));

	for (i = 1; i < populationSize; i++) {
		randomNr = RandomDouble(0, 1); // genereaza uniform r in (0,1]
		for(j = 0; j < populationSize; j++) // selecteza pentru supravieture individul j
			if (probCumulated[j] < randomNr && randomNr <= probCumulated[j+1]) chosen[i] = j; // al i-lea cromozom din noua populatie este al j-lea cromozom din populatia trecuta
	}

	for (int i = 0; i < populationSize; i++) memcpy(auxChromosomes[i], chromosomes[i], sizeof(chromosomes[i]));
	int aux = populationSize;
	populationSize = 0;
	for (i = 0; i < aux; i++) memcpy(chromosomes[populationSize++], auxChromosomes[chosen[i]], sizeof(chromosomes[i]));
}

bool chromosomeWillBeCrossed[DMAX];
void MutateChromosomes(bool chromosomes[DMAX][NMAX], int populationSize, int nrVals, int n) {
	double randomNr;

	for (int i = 0; i < populationSize; i++) {
		if (chromosomeWillBeCrossed[i]) continue; // daca a fost incrucisat, nu ii mai facem mutatie
		for (int j = 0; j < n*nrVals; j++) {
			randomNr = RandomDouble(0, 1);
			if (randomNr < P_MUTATION) // fiecare cromozom are o sansa de P_MUTATION % de a intra in mutatie
                chromosomes[i][j] = 1 - chromosomes[i][j];
		}
	}
}

void CrossChromosomes(bool chromosomes[DMAX][NMAX], int populationSize, int nrVals, int n) {
	double randomNr;
	int chosenForCross[DMAX], k = 0, i;
	memset(chromosomeWillBeCrossed, 0, sizeof(chromosomeWillBeCrossed));

	for (i = 0; i < populationSize; ++i) {
		randomNr = RandomDouble(0, 1);
		if (randomNr < P_CROSS) { // fiecare cromozom are o sansa de P_CROSS % de a intra in incrucisare
			chosenForCross[k++] = i;
			chromosomeWillBeCrossed[i] = true;
		}
	}

	if (k % 2 == 1) { // daca nr de cromozomi care vor fi incrucisati e impar, scot un cromozom
		chromosomeWillBeCrossed[chosenForCross[k-1]] = false;
		k--;
	}

	for (i = 0; i < k; i += 2) CrossIndividualChromosomes(chromosomes, nrVals, n, chosenForCross[i], chosenForCross[i+1]);
}

void CrossIndividualChromosomes(bool chromosomes[DMAX][NMAX], int nrVals, int n, int pos1, int pos2) {
	int pos = rand() % (nrVals*n) + 1; // punctul de taiere
	for (int j = pos; j < nrVals*n; ++j) swap(chromosomes[pos1][j], chromosomes[pos2][j]);
}

void runFunction(double(*testFunction)(double[], int), int nrRulari, int nrParametri) {
    int i;
    double sol, solMin = INF, solMax = 0, medie = 0, valori[DMAX];
    if(testFunction == Griewank) f<<"Griewank\n";
    else if(testFunction == DeJong) f<<"DeJong\n";
    else if(testFunction == Rastrigin) f<<"Rastrigin\n";
    else if(testFunction == Rosenbrock) f<<"Rosenbrock\n";
    f<<nrRulari<<" rulari\n";
    f<<nrParametri<<" parametri\n";
    for(i=0; i<nrRulari; i++) {
        sol = GeneticAlgorithm(testFunction, nrParametri);
        if(sol < solMin) solMin = sol;
        if(sol > solMax) solMax = sol;
        medie += sol;
    }
    medie /= nrRulari;
    f<<"Solutia minima: "<<solMin<<endl;
    f<<"Solutia maxima: "<<solMax<<endl;
    f<<"Media solutiilor: "<<medie<<endl;
}
