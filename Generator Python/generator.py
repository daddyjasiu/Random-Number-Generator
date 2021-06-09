import math
import matplotlib.pyplot as plotter
import numpy as np
from scipy.stats import chisquare


rozkladSerii1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 0, 2, 2, 3, 3, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 0, 2, 2, 3, 3, 3, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 0, 2, 3, 3, 3, 4, 4, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 0, 2, 3, 3, 4, 4, 5, 5, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 0, 2, 3, 3, 4, 5, 5, 5, 6, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 0, 2, 3, 4, 4, 5, 5, 6, 6, 7,  0,  0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7,  7,  0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7,  8,  8,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8,  8,  9,  9,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8,  8,  9,  9, 10,  0,  0,  0,  0,  0], 
                 [0, 0, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8,  9,  9, 10, 10, 11,  0,  0,  0,  0], 
                 [0, 0, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9,  9, 10, 10, 11, 11, 11,  0,  0,  0], 
                 [0, 0, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9,  9, 10, 10, 11, 11, 12, 12,  0,  0], 
                 [0, 0, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13,  0],
                 [0, 0, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 12, 12, 13, 13, 13, 13, 14]]

rozkladSerii2 = [[0, 0, 0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 4, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 5, 6, 0,  0, 0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 5, 7, 8,  0, 0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 5, 7, 8,  9, 0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 5, 7, 8,  9, 10, 0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 10, 11, 12, 0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 10, 11, 12, 13, 0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 12, 13, 13, 14, 0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 12, 13, 14, 15, 15, 0,  0,   0,  0,  0,  0,  0,  0,  0,  0],
                 [0, 0, 5, 7, 9, 11, 12, 13, 14, 15, 16, 16, 0,   0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 12, 13, 15, 15, 16, 17, 18,  0,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 13, 14, 15, 16, 17, 18, 18, 19,  0,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 19, 20,  0,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 13, 14, 15, 17, 17, 18, 19, 20, 21, 21,  0,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 20, 21, 22, 22,  0,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 22, 23, 24,  0,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 24, 25,  0,  0], 
                 [0, 0, 5, 7, 9, 11, 13, 15, 16, 17, 19, 20, 21, 22, 22, 23, 24, 25, 25, 26,  0],
                 [0, 0, 5, 7, 9, 11, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 24, 25, 26, 26, 27]]


def printG(ciag):
    print("Liczby wygenerowane przez generator liczb calkowitych G:")
    for i in ciag:
        print(i, end=" ")
    
    print("\n")


def printJ(ciag):
    print("Liczby wygenerowane przez generator liczb J (0,1):")
    for i in ciag:
        print(round(i, 2), end=" ")

    print("\n")


def printB(ciag):
    print("Liczby wygenerowane przez generator liczb B (rozklad Bernoulliego):")
    for i in ciag:
        print(i, end=" ")
    
    print("\n")


def printD(ciag):
    print("Liczby wygenerowane przez generator liczb D (rozklad dwumianowy):")
    for i in ciag:
        print(i, end=" ")
    
    print("\n")



def printP(ciag):
    print("Liczby wygenerowane przez generator liczb P (rozklad Poissona):")
    for i in ciag:
        print(i, end=" ")
    
    print("\n")


def printW(ciag):
    print("Liczby wygenerowane przez generator liczb W (rozklad wykladniczy):")
    for i in ciag:
        print(round(i, 2), end=" ")
    
    print("\n")


def printN(ciag):
    print("Liczby wygenerowane przez generator liczb N (rozklad normalny):")
    for i in ciag:
        print(round(i, 2), end=" ")
    
    print("\n")


def G(a, mod, seed, amount):

    numbersG = [0] * (amount + 1)
    numbersG[0] = seed

    for i in range(1, amount + 1):
        numbersG[i] = (a*numbersG[i-1] % mod)
    
    return numbersG


def J(a, mod, seed, amount):

    numbersG = []
    numbersG = G(a, mod, seed, amount)

    numbersJ = [0] * (amount + 1)

    for i in range(len(numbersG)):
        numbersJ[i] = (numbersG[i] / (mod))
    
    return numbersJ


def B(a, mod, seed, amount, p):

    numbersJ = []
    numbersJ = J(a, mod, seed, amount)

    numbersB = []

    for i in range(len(numbersJ)):
        if(numbersJ[i] <= p):
            numbersB.append(1)
        else:
            numbersB.append(0)
    
    return numbersB


def D(a, mod, seed, p, n):

    numbersJ = []
    numbersD = []

    successCounter = 0

    for i in range(0, n+1):
        numbersJ = J(a, mod, seed + i, n)

        for j in range(len(numbersJ)):
            if(numbersJ[j] <= p):
                successCounter += 1
        
        numbersD.append(successCounter)
        successCounter = 0
    
    return numbersD


def P(a, mod, seed, lambdaP, amount):

    numbersJ = []
    numbersJ = J(a, mod, seed, amount)

    numbersP = []

    L = math.exp(-lambdaP)
    k = 0
    p = 1

    for i in range(0, amount+1):
        while(p > L):
            k += 1
            p = p * numbersJ[i]
        
        numbersP.append(k-1)
        k = 0
        p = 1
    
    return numbersP


def W(a, mod, seed, amount):

    numbersJ = []
    numbersJ = J(a, mod, seed, amount+1)

    numbersW = []

    for i in range(0, amount+1):
        numbersW.append(-math.log(1 - numbersJ[i]))
    
    return numbersW


def N(a, mod, seed, amount):

    numbersJ = []
    numbersJ = J(a, mod, seed, amount)

    numbersN = []

    for i in range(0, len(numbersJ)-1, 2):
        numbersN.append(math.sqrt(-2 * math.log(1 - numbersJ[i])) * math.cos(2 * math.pi*numbersJ[i+1]))
        numbersN.append(math.sqrt(-2 * math.log(1 - numbersJ[i])) * math.sin(2 * math.pi*numbersJ[i+1]))
    
    return numbersN

def rysujWykres(ciag, name, num_bins, range=[-5, 5]):
    fig, ax = plotter.subplots()
    ax.hist(ciag, num_bins, range=range, edgecolor='black', density=True)
    ax.set_title(name)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    fig.set_size_inches(10, 5)
    fig.tight_layout()
    plotter.show()

def testSeriiLong(ciag):
    
    ciagSorted = ciag.copy()
    ciagSorted.sort()

    mediana = ciagSorted[int((len(ciagSorted))/2)] if len(ciagSorted)%2 != 0 else (1/2) * (ciagSorted[int(len(ciagSorted)/2)] + ciagSorted[(int((len(ciagSorted) - 1)/2))])

    ciagAiB = []

    for i in range(len(ciag)):
        if ciag[i] > mediana:
            ciagAiB.append('a')
        elif ciag[i] < mediana:
            ciagAiB.append('b')
        else:
            ciagAiB.append('-')

    liczbaSerii = 1
    aCounter = 0
    bCounter = 0

    for i in range(len(ciagAiB)-1):
        if ciagAiB[i] != '-':
            if ciagAiB[i] == 'a' and (ciagAiB[i+1] != 'a'):
                if i != len(ciagAiB)-2:
                    liczbaSerii += 1
            elif ciagAiB[i] == 'b' and (ciagAiB[i+1] != 'b'):
                if i != len(ciagAiB)-2:
                    liczbaSerii += 1

    for i in range(len(ciagAiB)):
        if ciagAiB[i] == 'a':
            aCounter += 1
        elif ciagAiB[i] == 'b':
            bCounter += 1

    k1 = rozkladSerii1[aCounter][bCounter]
    k2 = rozkladSerii2[aCounter][bCounter]

    print("----------------------------------------------------------------------------------------------------------------------")
    print("TEST SERII: DŁUGI")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("CIAG WEJSCIOWY:                           ", ciag)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("CIAG WEJSCIOWY POSORTOWANY:               ", ciagSorted)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("MEDIANA CIAGU WEJSCIOWEGO:                ", mediana)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("CIAG SERIOWY a & b:                       ", ciagAiB)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("LICZBA SERII a & b:                       ", liczbaSerii)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("LICZBA a:                                 ", aCounter)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("LICZBA b:                                 ", bCounter)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("ZBIOR KRYTYCZNY:                           (-inf ;", k1, "]","∪", "[", k2, "; +inf]")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("CZY", liczbaSerii, "∈ (-inf ;", k1, "]","∪", "[", k2, "; +inf]?       ", "NIE" if k1 < liczbaSerii < k2 else "TAK")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("FINALNY WERDYKT:                          ", "CIĄG DOBRZE LOSOWY" if k1 < liczbaSerii < k2 else "CIAG SŁABO LOSOWY")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("######################################################################################################################")

def testSeriiShort(ciag):
    
    ciagSorted = ciag.copy()
    ciagSorted.sort()

    mediana = ciagSorted[int((len(ciagSorted))/2)] if len(ciagSorted)%2 != 0 else (1/2) * (ciagSorted[int(len(ciagSorted)/2)] + ciagSorted[(int((len(ciagSorted) - 1)/2))])

    ciagAiB = []

    for i in range(len(ciag)):
        if ciag[i] > mediana:
            ciagAiB.append('a')
        elif ciag[i] < mediana:
            ciagAiB.append('b')
        else:
            ciagAiB.append('-')

    liczbaSerii = 1
    aCounter = 0
    bCounter = 0

    for i in range(len(ciagAiB)-1):
        if ciagAiB[i] != '-':
            if ciagAiB[i] == 'a' and (ciagAiB[i+1] != 'a'):
                if i != len(ciagAiB)-2:
                    liczbaSerii += 1
            elif ciagAiB[i] == 'b' and (ciagAiB[i+1] != 'b'):
                if i != len(ciagAiB)-2:
                    liczbaSerii += 1

    for i in range(len(ciagAiB)):
        if ciagAiB[i] == 'a':
            aCounter += 1
        elif ciagAiB[i] == 'b':
            bCounter += 1

    k1 = rozkladSerii1[aCounter][bCounter]
    k2 = rozkladSerii2[aCounter][bCounter]

    print("----------------------------------------------------------------------------------------------------------------------")
    print("TEST SERII: KRÓTKI")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("CZY", liczbaSerii, "∈ (-inf ;", k1, "]","∪", "[", k2, "; +inf]?       ", "NIE" if k1 < liczbaSerii < k2 else "TAK")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("FINALNY WERDYKT:                          ", "CIĄG DOBRZE LOSOWY" if k1 < liczbaSerii < k2 else "CIAG SŁABO LOSOWY")
    print("----------------------------------------------------------------------------------------------------------------------")
    print("######################################################################################################################")


def chiKwadrat(ciag):
    pass

# Zasady dobierania parametrów a, b, p, X0:
# •"p" powinno być bardzo duże, aby jak najbardziej ograniczyć okresowość (powtarzanie się tych samych liczb)
# •"a" powinno być bardzo duże i względnie pierwsze z "p", również, aby ograniczyć okresowość
# •"b" ma mniejsze znaczenie, często przyjmuje się zero
# •X0można przyjąć jako 1 lub stempel czasu (czas od 1.01.1970)

# GENERATOR G INIT
randomNumbersG = []
aG = 2                  #  pierwszy parametr generatora
modG = 11               #  drugi parametr modulo generatora
seedG = 1               #  ziarno generatora
amountG = 14           #  ilosc docelowo wygenerowanych liczb w G + ziarno
        
# GENERATOR J INIT      
randomNumbersJ = []     
aJ = 7              #  pierwszy parametr generatora
modJ = 11       #  drugi parametr modulo generatora
seedJ = 1               #  ziarno generatora
amountJ = 14            #  ilosc docelowo wygenerowanych liczb w J + ziarno
        
# GENERATOR B INIT      
randomNumbersB = []     
aB = 16807              #  pierwszy parametr generatora
modB = 2147483647       #  drugi parametr modulo generatora
seedB = 1               #  ziarno generatora
amountB = 100           #  ilosc docelowo wygenerowanych liczb w B + ziarno
pB = 0.4                #  parametr prawdopodobienstwa sukcesu w B
        
# GENERATOR D INIT      
randomNumbersD = []     
aD = 16807              #  pierwszy parametr generatora
modD = 2147483647       #  drugi parametr modulo generatora
seedD = 1               #  ziarno generatora
nD = 100                #  ilosc prob szukania sukcesu rozkladu dwumianowego
pD = 0.4                #  parametr prawdopodobienstwa sukcesu w D
        
# GENERATOR P INIT      
randomNumbersP = []     
aP = 16807              #  pierwszy parametr generatora
modP = 2147483647       #  drugi parametr modulo generatora
seedP = 1               #  ziarno generatora
amountP = 100           #  ilosc docelowo wygenerowanych liczb w P + ziarno
lambdaP = 4             #  parametr Poisonna
        
# GENERATOR W INIT      
randomNumbersW = []     
aW = 16807              #  pierwszy parametr generatora
modW = 2147483647       #  drugi parametr modulo generatora
seedW = 1               #  ziarno generatora
amountW = 100000        #  ilosc docelowo wygenerowanych liczb w W + ziarno
        
# GENERATOR N INIT      
randomNumbersN = []     
aN = 16807              #  pierwszy parametr generatora
modN = 2147483647       #  drugi parametr modulo generatora
seedN = 1               #  ziarno generatora
amountN = 100000        #  ilosc docelowo wygenerowanych liczb w N + ziarno


# Test Generatora G
randomNumbersG = G(aG, modG, seedG, amountG)

# Test Generatora J
randomNumbersJ = J(aJ, modJ, seedJ, amountJ)

# Test Generatora B
randomNumbersB = B(aB, modB, seedB, amountB, pB)

# Test Generatora D
randomNumbersD = D(aD, modD, seedD, pD, nD)

# Test Generatora P
randomNumbersP = P(aP, modP, seedP, lambdaP, amountP)

# Test Generatora W
randomNumbersW = W(aW, modW, seedW, amountW)

# Test Generatora N
randomNumbersN = N(aN, modN, seedN, amountN)


rysujWykres(randomNumbersN, "ROZKŁAD NORMALNY", 100)
rysujWykres(randomNumbersW, "ROZKŁAD WYKŁADNICZY", 100)

# testSeriiLong([16, 20, 25, 34, 22, 33, 47, 30, 28, 19, 22, 40, 36, 31, 38])
# testSeriiShort(randomNumbersG)

# arr = [1,2,3,4,5,6,7,8,9]
# chisquare(arr)

# print(chisquare(arr))

printG(randomNumbersG)

# 3. Projekt “generator liczb losowych”

# W ramach tego projektu należy zaimplementować generator G całkowitych liczb pseudo-losowych o rozkładzie równomiernym,
# oczywiście bez wykorzystania dostępnych funkcji czy bibliotek dla generatorów liczb losowych.
# Nie wolno również wykorzystywać dostępu do takich źródeł „pseudolosowych” danych jak zegar systemowy.
# Może to być jeden z prostych generatorów opartych na arytmetyce modularnej.
# Można zaimplementować więcej niż jeden z takich dostępnych w literaturze generatorów.

# Na podstawie generatora G należy następnie stworzyć generator J liczb losowych z rozkładu jednostajnego
# a przedziale (0, 1), a następnie – na jego podstawie – generatory liczb losowych z rozkładów:

# - Bernoulliego[dwupunktowego](B),
# - dwumianowego(D),
# - Poissona(P),
# - wykładniczego(W),
# - normalnego (N).

# Następnie należy znaleźć w literaturze metody testowania jakości generatorów liczb losowych i wykonać
# odpowiednie testy dla generatorów G, J, B, D, P, W, N.

# Przydatne materiały
# Test chi-kwadrat zgodności rozkładu
# Testy serii
# Generator Mersenne Twister

# http://home.agh.edu.pl/~chwiej/mn/generatory_16.pdf
# https://webhome.phy.duke.edu/~rgb/General/dieharder.php
# http://simul.iro.umontreal.ca/testu01/tu01.html

# Student może spróbować zaprojektować własny, oryginalny generator G i – przy użyciu testów losowości
# z powyższych linków –sprawdzić, czy jest on lepszy/gorszy od innych, znanych z literatury prostych generatorów
# z rozkładu równomiernego wykorzystujących np. arytmetykę modularną.