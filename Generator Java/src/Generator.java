import java.util.ArrayList;

public class Generator {

    static void printG(ArrayList<Integer> numbers){
        System.out.println("Liczby wygenerowane przez generator liczb calkowitych G:");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.print(numbers.get(i) + " ");
        }
        System.out.println();
    }

    static void printJ(ArrayList<Double> numbers){
        System.out.println("Liczby wygenerowane przez generator liczb J (0,1):");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.printf("%.2f", numbers.get(i));
            System.out.print(" ");
        }
        System.out.println();
    }

    static void printB(ArrayList<Integer> numbers){
        System.out.println("Liczby wygenerowane przez generator liczb B (rozklad Bernoulliego):");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.print(numbers.get(i) + " ");
        }
        System.out.println();
    }

    static void printD(ArrayList<Integer> numbers){
        System.out.println("Liczby wygenerowane przez generator liczb D (rozklad dwumianowy):");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.print(numbers.get(i) + " ");
        }
        System.out.println();
    }


    static void printP(ArrayList<Integer> numbers){
        System.out.println("Liczby wygenerowane przez generator liczb P (rozklad Poissona):");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.print(numbers.get(i) + " ");
        }
        System.out.println();
    }

    static void printW(ArrayList<Double> numbers){
        System.out.println("Liczby wygenerowane przez generator liczb W (rozklad wykladniczy):");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.printf("%.2f", numbers.get(i));
            System.out.print(" ");
        }
        System.out.println();
    }

    static void printN(ArrayList<Double> numbers){
        System.out.println("Liczby wygenerowane przez generator liczb N (rozklad normalny):");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.printf("%.2f", numbers.get(i));
            System.out.print(" ");
        }
        System.out.println();
    }

    static ArrayList<Integer> G(int a, int mod, int seed, int amount){

        ArrayList<Integer> numbersG = new ArrayList<>(amount+1);
        numbersG.add(0, seed);

        for (int i = 1; i < amount+1; i++) {
            numbersG.add(i, a*numbersG.get(i-1)%mod);
        }

        return numbersG;
    }

    static ArrayList<Double> J(int a, int mod, int seed, int amount){

        ArrayList<Integer> numbersG;
        numbersG = G(a, mod, seed, amount);

        ArrayList<Double> numbersJ = new ArrayList<>(numbersG.size());

        for (int i = 0; i < numbersG.size(); i++) {
            numbersJ.add(i, (double) numbersG.get(i) / (mod));
        }

        return numbersJ;
    }

    static ArrayList<Integer> B(int a, int mod, int seed, int amount, double p){

        ArrayList<Double> numbersJ;
        numbersJ = J(a, mod, seed, amount);

        ArrayList<Integer> numbersB = new ArrayList<>(numbersJ.size());

        for (int i = 0; i < numbersJ.size(); i++) {
            if(numbersJ.get(i) <= p)
                numbersB.add(1);
            else
                numbersB.add(0);
        }

        return numbersB;
    }

    static ArrayList<Integer> D(int a, int mod, int seed, double p, int n){

        ArrayList<Double> numbersJ;
        ArrayList<Integer> numbersD = new ArrayList<>(n + 1);

        int successCounter = 0;

        for (int i = 0; i < n+1; i++) {
            numbersJ = J(a, mod, seed + i, n);

            for (int j = 0; j < numbersJ.size(); j++) {
                if(numbersJ.get(j) <= p)
                    successCounter++;
            }

            numbersD.add(successCounter);
            successCounter = 0;
        }

        return numbersD;
    }

    static ArrayList<Integer> P(int a, int mod, int seed, int lambda, int amount){

        ArrayList<Double> numbersJ;
        numbersJ = J(a, mod, seed, amount);

        ArrayList<Integer> numbersP = new ArrayList<>(amount + 1);

        double L = Math.exp(-lambda);
        int k = 0;
        double p = 1;
        int j = 0;

        for (int i = 0; i < amount + 1; i++) {
            while(p > L){
                k++;
                p = p * numbersJ.get(j);
                j++;

                if(j == numbersJ.size()-1)
                    j = 0;
            }
            numbersP.add(k-1);
            k = 0;
            p = 1;
        }

        return numbersP;
    }

    static ArrayList<Double> W(int a, int mod, int seed, int amount){

        ArrayList<Double> numbersJ;
        numbersJ = J(a, mod, seed, amount + 1);

        ArrayList<Double> numbersW = new ArrayList<>(amount + 1);

        for (int i = 0; i < amount + 1; i++) {
            numbersW.add(-Math.log(1 - numbersJ.get(i)));
        }

        return numbersW;
    }

    static ArrayList<Double> N(int a, int mod, int seed, int amount){

        ArrayList<Double> numbersJ;
        numbersJ = J(a, mod, seed, amount);

        ArrayList<Double> numbersN = new ArrayList<>(amount + 1);

        for (int i = 0; i < numbersJ.size() - 1; i = i + 2) {
            numbersN.add(Math.sqrt(-2 * Math.log(1 - numbersJ.get(i)))
                            * Math.cos(2 * Math.PI*numbersJ.get(i+1)));
            numbersN.add(Math.sqrt(-2 * Math.log(1 - numbersJ.get(i)))
                            * Math.sin(2 * Math.PI*numbersJ.get(i+1)));
        }

        return numbersN;
    }

    public static void main(String[] args){

        //GENERATOR G INIT
        ArrayList<Integer> randomNumbersG;
        int aG = 7;             // pierwszy parametr generatora
        int modG = 101;         // drugi parametr modulo generatora
        int seedG = 19;         // ziarno generatora
        int amountG = 10;       // ilosc docelowo wygenerowanych liczb w G + ziarno

        //GENERATOR J INIT
        ArrayList<Double> randomNumbersJ;
        int aJ = 7;             // pierwszy parametr generatora
        int modJ = 113;         // drugi parametr modulo generatora
        int seedJ = 73;         // ziarno generatora
        int amountJ = 10;       // ilosc docelowo wygenerowanych liczb w J + ziarno

        //GENERATOR B INIT
        ArrayList<Integer> randomNumbersB;
        int aB = 7;             // pierwszy parametr generatora
        int modB = 101;         // drugi parametr modulo generatora
        int seedB = 19;         // ziarno generatora
        int amountB = 10;       // ilosc docelowo wygenerowanych liczb w B + ziarno
        double pB = 0.4;        // parametr prawdopodobienstwa sukcesu w B

        //GENERATOR D INIT
        ArrayList<Integer> randomNumbersD;
        int aD = 7;             // pierwszy parametr generatora
        int modD = 101;         // drugi parametr modulo generatora
        int seedD = 19;         // ziarno generatora
        int nD = 10;            // ilosc prob szukania sukcesu rozkladu dwumianowego
        double pD = 0.4;        // parametr prawdopodobienstwa sukcesu w D

        //GENERATOR P INIT
        ArrayList<Integer> randomNumbersP;
        int aP = 7;             // pierwszy parametr generatora
        int modP = 113;         // drugi parametr modulo generatora
        int seedP = 73;         // ziarno generatora
        int amountP = 10;       // ilosc docelowo wygenerowanych liczb w P + ziarno
        int lambda = 3;         // parametr Poisonna

        //GENERATOR W INIT
        ArrayList<Double> randomNumbersW;
        int aW = 7;             // pierwszy parametr generatora
        int modW = 101;         // drugi parametr modulo generatora
        int seedW = 19;         // ziarno generatora
        int amountW = 10;       // ilosc docelowo wygenerowanych liczb w W + ziarno

        //GENERATOR N INIT
        ArrayList<Double> randomNumbersN;
        int aN = 7;             // pierwszy parametr generatora
        int modN = 101;         // drugi parametr modulo generatora
        int seedN = 19;         // ziarno generatora
        int amountN = 10;       // ilosc docelowo wygenerowanych liczb w N + ziarno


        // Test Generatora G
        randomNumbersG = G(aG, modG, seedG, amountG);
        printG(randomNumbersG);

        // Test Generatora J
        randomNumbersJ = J(aJ, modJ, seedJ, amountJ);
        printJ(randomNumbersJ);

        //Test Generatora B
        randomNumbersB = B(aB, modB, seedB, amountB, pB);
        printB(randomNumbersB);

        //Test Generatora D
        randomNumbersD = D(aD, modD, seedD, pD, nD);
        printD(randomNumbersD);

        //Test Generatora P
        randomNumbersP = P(aP, modP, seedP, lambda, amountP);
        printP(randomNumbersP);

        //Test Generatora W
        randomNumbersW = W(aW, modW, seedW, amountW);
        printW(randomNumbersW);

        //Test Generatora N
        randomNumbersN = N(aN, modN, seedN, amountN);
        printN(randomNumbersN);

    }
}

/*
3. Projekt “generator liczb losowych”

W ramach tego projektu należy zaimplementować generator G całkowitych liczb pseudo-losowych o rozkładzie równomiernym,
oczywiście bez wykorzystania dostępnych funkcji czy bibliotek dla generatorów liczb losowych.
Nie wolno również wykorzystywać dostępu do takich źródeł „pseudolosowych” danych jak zegar systemowy.
Może to być jeden z prostych generatorów opartych na arytmetyce modularnej.
Można zaimplementować więcej niż jeden z takich dostępnych w literaturze generatorów.

Na podstawie generatora G należy następnie stworzyć generator J liczb losowych z rozkładu jednostajnego
a przedziale (0, 1), a następnie – na jego podstawie – generatory liczb losowych z rozkładów:

- Bernoulliego[dwupunktowego](B),
- dwumianowego(D),
- Poissona(P),
- wykładniczego(W),
- normalnego (N).

Następnie należy znaleźć w literaturze metody testowania jakości generatorów liczb losowych i wykonać
odpowiednie testy dla generatorów G, J, B, D, P, W, N.

Przydatne materiały
Test chi-kwadrat zgodności rozkładu
Testy serii
Generator Mersenne Twister

http://home.agh.edu.pl/~chwiej/mn/generatory_16.pdf
https://webhome.phy.duke.edu/~rgb/General/dieharder.php
http://simul.iro.umontreal.ca/testu01/tu01.html

Student może spróbować zaprojektować własny, oryginalny generator G i –przy użyciu testów losowości
z powyższych linków –sprawdzić, czy jest on lepszy/gorszy od innych, znanych z literatury prostych generatorów
z rozkładu równomiernego wykorzystujących np. arytmetykę modularną.
 */