#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <fstream>
#include <chrono>
#define DMAX 1000
#define NMAX 10000
#define INF 999999999

using namespace std;
ofstream fout("date_experimentale.txt");
ofstream f("date_experimentaleSA.txt");

double DeJong(double[], int);
double SixHump(double[], int);
double Schwefel(double[], int);
double Rastrigin(double[], int);

double HillClimb(double(*testFunction)(double[], int), int, bool);
double SimulatedAnnealing(double(*testFunction)(double[], int), int);

void Initialize(double(*testFunction)(double[], int), char[], int, int);
void generateNeighbours(double(*testFunction)(double[], int), char[][DMAX], char[], int, int);
void getInterval(double(*testFunction)(double[], int), double&, double&);
void convertToDouble(double(*testFunction)(double[], int), char[], double[], int, int, double, double);
void calculeazaParametrii(double(*testFunction)(double[], int), int, int, int&, int&);
double randomDouble(const double&, const double&);
bool badValueIsAccepted(double, double);

void runFunction(double(*testFunction)(double[], int), int, int);

int main()
{
    srand(time(NULL));
    runFunction(DeJong, 1, 30);
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

double SimulatedAnnealing(double(*testFunction)(double[], int), int nrVals) {
    double v[DMAX], bestSol = INF, bestIterationSol, temperature, neighbourValue;
	double coolingRate = 3, absoluteTemperature = 0.0000001, a, b;
	int nrIterations = 100, randomPoz, i, precizie = 2, n, N;
	bool local;
	char neighbour[DMAX], x[DMAX];

	//double timp;
	//auto start = std::chrono::system_clock::now();

	if(testFunction != SixHump) {
        getInterval(testFunction, a, b);
        calculeazaParametrii(testFunction, precizie, nrVals, n, N);
	}
	else n = 10;
	for (i = 0; i < nrIterations; ++i) {
        Initialize(testFunction, x, nrVals, n);
        convertToDouble(testFunction, x, v, nrVals, n, a, b);
        bestIterationSol = testFunction(v, nrVals);
        temperature = 100;
        local = true;
        while (local && temperature > absoluteTemperature) {
            randomPoz = rand() % (n*nrVals);
            strcpy(neighbour, x);
            neighbour[randomPoz] = !neighbour[randomPoz];
			local = false;

            convertToDouble(testFunction, neighbour, v, nrVals, n, a, b);
            neighbourValue = testFunction(v, nrVals);
			if(neighbourValue < bestIterationSol) {
                bestIterationSol = neighbourValue;
                local = true;
            }
            else if(badValueIsAccepted(neighbourValue - bestIterationSol, temperature)) {
                bestIterationSol = neighbourValue;
                local = true;
            }

			temperature /= coolingRate;

			//auto ending = chrono::system_clock::now();
            //chrono::duration<double> elapsed_seconds = ending-start;
            //timp = elapsed_seconds.count();

            //f<<timp<<endl;
			//fout<<bestIterationSol<<endl;
        }
        bestSol = min(bestIterationSol, bestSol);
        //fout<<bestSol<<endl;
	}
	return bestSol;
}

double HillClimb(double(*testFunction)(double[], int), int nrVals, bool improvement) { // improvement = 0 daca FirstImprovement, altfel 1
    double v[DMAX], bestSol = INF, bestIterationSol, a=0, b=0, bestNeighbour, neighbourValue;
    int n, N, indice, nrIterations = 100, i, j, precizie=2;
    char neighbours[DMAX][DMAX], x[DMAX];
	bool local;

	//double timp;
	//auto start = std::chrono::system_clock::now();

	if(testFunction != SixHump) {
        getInterval(testFunction, a, b);
        calculeazaParametrii(testFunction, precizie, nrVals, n, N);
	}
	else n = 10;
	for (i = 0; i < nrIterations; ++i) {
        Initialize(testFunction, x, nrVals, n);
        convertToDouble(testFunction, x, v, nrVals, n, a, b);
        bestIterationSol = testFunction(v, nrVals);
        local = true;
		while (local) {
            generateNeighbours(testFunction, neighbours, x, n, nrVals);
			local = false;
			bestNeighbour = bestIterationSol+1;
            for(j=0; j<n*nrVals; j++) {
                convertToDouble(testFunction, neighbours[j], v, nrVals, n, a, b);
                neighbourValue = testFunction(v, nrVals);
                if(neighbourValue < bestNeighbour) {
                    bestNeighbour = neighbourValue;
                    indice = j;
                }
                if(improvement == 0 && neighbourValue < bestIterationSol) break;
            }
            if(bestNeighbour < bestIterationSol) {
                bestIterationSol = bestNeighbour;
                local = true;
                strcpy(x, neighbours[indice]);
            }
            //auto ending = chrono::system_clock::now();
            //chrono::duration<double> elapsed_seconds = ending-start;
            //timp = elapsed_seconds.count();

            //fout<<bestIterationSol<<endl;
            //f<<timp<<endl;
		}
		bestSol = min(bestIterationSol, bestSol);
		fout<<bestSol<<endl;
	}
	return bestSol;
}

bool badValueIsAccepted(double delta, double temperature) {
	return (randomDouble(0, 0.9999999999999) < exp(-delta / temperature));
}

double randomDouble(const double &minVal, const double &maxVal) {
	double f;
	f = (double)rand() / RAND_MAX;
	return minVal + f * (maxVal - minVal);
}

void calculeazaParametrii(double(*testFunction)(double[], int), int d, int nrVals, int& n, int& N) {
    double a, b;
    getInterval(testFunction, a, b);
    N = (b-a)*pow(10, d); //subintervale egale
    n = ceil(log2(N)); // n biti pt fiecare nr
}

void generateNeighbours(double(*testFunction)(double[], int), char neighbours[][DMAX], char x[], int n, int nrVals) {
    int i;
    for(i=0; i<n*nrVals; i++) {
        strcpy (neighbours[i], x);
        neighbours[i][i] = !x[i];
    }
    if(testFunction == SixHump) neighbours[19][19] = 0;
}

void convertToDouble(double(*testFunction)(double[], int), char x[], double v[], int nrVals, int n, double a, double b) {
    int i, j, nr;
    for(i=0; i<n*nrVals; i+=n) {
        nr=0;
        for(j=0; j<n; j++) nr += pow(2,j)*x[i+j];
        if(testFunction == SixHump && i==0) v[0] = -3 + nr*6/(pow(2,n)-1);
        else if(testFunction == SixHump && i==10) v[1] = -2 + nr*4/(pow(2,n)-1);
        else v[i/n] = a + nr*(b-a)/(pow(2,n)-1);
    }
}

void getInterval(double(*testFunction)(double[], int), double &a, double &b) {
    if(testFunction == DeJong || testFunction == Rastrigin) {a=-5.12; b=5.12;}
    else if(testFunction == Schwefel) {a=-500; b=500;}
}

void Initialize(double(*testFunction)(double[], int), char x[], int nrVals, int n) {
    int i;
    for(i=0; i<n*nrVals; i++) x[i] = rand()%2;
    if(testFunction == SixHump) x[19] = 0;
}

void runFunction(double(*testFunction)(double[], int), int nrRulari, int nrParametri) {
    int i;
    double sol, solMin = INF, solMax = 0, medie = 0, dispersie = 0, deviatie, valori[DMAX];
    cout<<"Hill Climbing pentru functia: ";
    if(testFunction == DeJong) cout<<"De Jong\n";
    else if(testFunction == SixHump) cout<<"Six-hump camel back\n";
    else if(testFunction == Rastrigin) cout<<"Rastrigin\n";
    else if(testFunction == Schwefel) cout<<"Schwefel\n";
    for(i=0; i<nrRulari; i++) {
        sol = HillClimb(testFunction, nrParametri, 0); //1 = BestImprovement, 0 = FirstImprovement
        if(sol < solMin) solMin = sol;
        if(sol > solMax) solMax = sol;
        medie += sol;
        valori[i] = sol;
    }
    medie /= nrRulari;
    for(i=0; i<nrRulari; i++) dispersie += (valori[i] - medie) * (valori[i] - medie);
    dispersie /= (nrParametri-1);
    deviatie = sqrt(dispersie);
    cout<<"Solutia minima: "<<solMin<<endl;
    cout<<"Solutia maxima: "<<solMax<<endl;
    cout<<"Media solutiilor: "<<medie<<endl;
    cout<<"Deviatia standard: "<<deviatie<<endl;

    cout<<endl;
    double solSA, solMinSA = INF, solMaxSA = 0, medieSA = 0, dispersieSA = 0, deviatieSA, valoriSA[DMAX];
    cout<<"Simulated Annealing pentru functia: ";
    if(testFunction == DeJong) cout<<"De Jong\n";
    else if(testFunction == SixHump) cout<<"Six-hump camel back\n";
    else if(testFunction == Rastrigin) cout<<"Rastrigin\n";
    else if(testFunction == Schwefel) cout<<"Schwefel\n";
    for(i=0; i<nrRulari; i++) {
        solSA = SimulatedAnnealing(testFunction, nrParametri);
        if(solSA < solMinSA) solMinSA = solSA;
        if(solSA > solMaxSA) solMaxSA = solSA;
        medieSA += solSA;
        valoriSA[i] = solSA;
    }
    medieSA /= nrRulari;
    for(i=0; i<nrRulari; i++) dispersieSA += (valoriSA[i] - medieSA) * (valoriSA[i] - medieSA);
    dispersieSA /= (nrParametri-1);
    deviatieSA = sqrt(dispersieSA);
    cout<<"Solutia minima: "<<solMinSA<<endl;
    cout<<"Solutia maxima: "<<solMaxSA<<endl;
    cout<<"Media solutiilor: "<<medieSA<<endl;
    cout<<"Deviatia standard: "<<deviatieSA<<endl;
}
