import math
import matplotlib.pyplot as plotter
import numpy as np
import numpy.random as random
from scipy.stats import chisquare
import statistics

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


rozkladChi = [1, 3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307]

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
    j = 0

    for i in range(0, amount+1):
        while(p > L):
            k += 1
            p = p * numbersJ[j]
            j += 1

            if j == (len(numbersJ)-1):
                j = 0
        
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


def rysujWykres(ciag, name, num_bins, range=[-5, 10]):
    fig, ax = plotter.subplots()
    ax.hist(ciag, num_bins, range=range, edgecolor='black', density=True)
    ax.set_title(name)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    fig.set_size_inches(10, 5)
    fig.tight_layout()
    plotter.show()


def median(ciag):
    ciagSorted = ciag.copy()
    ciagSorted.sort()

    mediana = ciagSorted[int((len(ciagSorted))/2)] if len(ciagSorted)%2 != 0 else (1/2) * (ciagSorted[int(len(ciagSorted)/2)] + ciagSorted[(int((len(ciagSorted) - 1)/2))])
    return mediana


def testSeriiLong(ciag):
    
    mediana = median(ciag)

    ciagSorted = ciag.copy()
    ciagSorted.sort()

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

    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("TEST SERII: DŁUGI")
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("CIAG WEJSCIOWY:                           ", end='') 
    print(["{0:0.2f}".format(i) for i in ciag])
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("CIAG WEJSCIOWY POSORTOWANY:               ", end='')
    print(["{0:0.2f}".format(i) for i in ciagSorted])
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("MEDIANA CIAGU WEJSCIOWEGO:                ", mediana)
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("CIAG SERIOWY a & b:                       ", ciagAiB)
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("LICZBA SERII a & b:                       ", liczbaSerii)
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("LICZBA a:                                 ", aCounter)
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("LICZBA b:                                 ", bCounter)
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("ZBIOR KRYTYCZNY:                           (-inf ;", k1, "]","∪", "[", k2, "; +inf]")
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("CZY", liczbaSerii, "∈ (-inf ;", k1, "]","∪", "[", k2, "; +inf]?       ", "NIE" if k1 < liczbaSerii < k2 else "TAK")
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("FINALNY WERDYKT:                          ", "CIĄG DOBRZE LOSOWY" if k1 < liczbaSerii < k2 else "CIAG SŁABO LOSOWY")
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("###############################################################################################################################################################")


def testSeriiShort(ciag):

    mediana = median(ciag)

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


def srednia(ciag):
    ciagSorted = ciag.copy()
    ciagSorted.sort()
    return statistics.mean(ciagSorted)


def odchylenieStandardowe(ciag):
    ciagSorted = ciag.copy()
    ciagSorted.sort()
    return statistics.stdev(ciagSorted)


def oczekiwanaP(lambdaP, ciagNoDups, ciag):
    
    result = [0, 0, 0, 0, 0, 0]
    amount = len(ciag)

    result[0] = (amount*pow(lambdaP, ciagNoDups[0])*math.exp(-lambdaP)/math.factorial(ciagNoDups[0]))
    result[0] += (amount*pow(lambdaP, ciagNoDups[1])*math.exp(-lambdaP)/math.factorial(ciagNoDups[1]))
    result[0] += (amount*pow(lambdaP, ciagNoDups[2])*math.exp(-lambdaP)/math.factorial(ciagNoDups[2]))
    result[1] = (amount*pow(lambdaP, ciagNoDups[3])*math.exp(-lambdaP)/math.factorial(ciagNoDups[3]))
    result[1] += (amount*pow(lambdaP, ciagNoDups[4])*math.exp(-lambdaP)/math.factorial(ciagNoDups[4]))
    result[2] = (amount*pow(lambdaP, ciagNoDups[5])*math.exp(-lambdaP)/math.factorial(ciagNoDups[5]))
    result[2] += (amount*pow(lambdaP, ciagNoDups[6])*math.exp(-lambdaP)/math.factorial(ciagNoDups[6]))
    result[3] = (amount*pow(lambdaP, ciagNoDups[7])*math.exp(-lambdaP)/math.factorial(ciagNoDups[7]))
    result[3] += (amount*pow(lambdaP, ciagNoDups[8])*math.exp(-lambdaP)/math.factorial(ciagNoDups[8]))
    result[4] = (amount*pow(lambdaP, ciagNoDups[9])*math.exp(-lambdaP)/math.factorial(ciagNoDups[9]))
    result[4] += (amount*pow(lambdaP, ciagNoDups[10])*math.exp(-lambdaP)/math.factorial(ciagNoDups[10]))
    result[5] = (amount*pow(lambdaP, 11)*math.exp(-lambdaP)/math.factorial(11))
    result[5] += (amount*pow(lambdaP, ciagNoDups[11])*math.exp(-lambdaP)/math.factorial(ciagNoDups[11]))

    return result


def chiKwadratB(ciag, p):
    
    alpha = 0.05
    degFreedom = 1 # dzielimy na dwa kubełki

    observedZero = 0
    observedOne = 0

    for i in range(len(ciag)):
        if ciag[i] == 0:
            observedZero += 1
        else:
            observedOne += 1

    estimatedZero = (1-p)*len(ciag)
    estimatedOne = p*len(ciag)

    chi0 = pow(observedZero - estimatedZero, 2) / estimatedZero
    chi1 = pow(observedOne - estimatedOne, 2) / estimatedOne

    if chi0 + chi1 < rozkladChi[degFreedom]:
        print("CIAG POSIADA DOBRE LICZBY Z ROZKŁADU BERNOULLIEGO")
        print("#######################################################", end='\n\n')
    else:
        print("CIAG POSIADA ZŁE LICZBY Z ROZKŁADU BERNOULLIEGO")
        print("#######################################################", end='\n\n')


def chiKwadratP(ciag, lambdaP):
        
    ciagSorted = ciag.copy()
    ciagSorted.sort()

    ciagNoDups = list(dict.fromkeys(ciagSorted.copy()))

    alpha = 0.05
    degFreedom = 5 # dzielimy na 6 kubełków

    observed = [0] * 6

    estimated = oczekiwanaP(lambdaP, ciagNoDups, ciag)

    j = 0

    for i in range(len(ciagSorted)):
        if 0 <= ciag[i] <= 2:
            observed[0] += 1
        elif 2 < ciag[i] <= 4:
            observed[1] += 1
        elif 4 < ciag[i] <= 6:
            observed[2] += 1
        elif 6 < ciag[i] <= 8:
            observed[3] += 1
        elif 8 < ciag[i] <= 10:
            observed[4] += 1
        else:
            observed[5] += 1

    chi = 0

    for i in range(len(observed)):
        chi += pow(observed[i] - estimated[i], 2) / estimated[i]

    if chi < rozkladChi[degFreedom]:
        print("CIAG POSIADA DOBRE LICZBY Z ROZKŁADU POISSONA")
        print("#######################################################", end='\n\n')
    else:
        print("CIAG POSIADA ZŁE LICZBY Z ROZKŁADU POISSONA")
        print("#######################################################", end='\n\n')


# GENERATOR G INIT
randomNumbersG = []
aG = 16807              #  pierwszy parametr generatora
modG = 2147             #  drugi parametr modulo generatora
seedG = 1               #  ziarno generatora
amountG = 10            #  ilosc docelowo wygenerowanych liczb w G + ziarno
        
# GENERATOR J INIT      
randomNumbersJ = []     
aJ = 16807              #  pierwszy parametr generatora
modJ = 2147483647       #  drugi parametr modulo generatora
seedJ = 1               #  ziarno generatora
amountJ = 10            #  ilosc docelowo wygenerowanych liczb w J + ziarno
        
# GENERATOR B INIT      
randomNumbersB = []     
aB = 16807              #  pierwszy parametr generatora
modB = 2147483647       #  drugi parametr modulo generatora
seedB = 1               #  ziarno generatora
amountB = 10            #  ilosc docelowo wygenerowanych liczb w B + ziarno
pB = 0.4                #  parametr prawdopodobienstwa sukcesu w B
        
# GENERATOR D INIT      
randomNumbersD = []     
aD = 16807              #  pierwszy parametr generatora
modD = 2147483647       #  drugi parametr modulo generatora
seedD = 1               #  ziarno generatora
nD = 10                 #  ilosc prob szukania sukcesu rozkladu dwumianowego
pD = 0.4                #  parametr prawdopodobienstwa sukcesu w D
        
# GENERATOR P INIT      
randomNumbersP = []     
aP = 16807              #  pierwszy parametr generatora
modP = 2147483647       #  drugi parametr modulo generatora
seedP = 1               #  ziarno generatora
amountP = 10            #  ilosc docelowo wygenerowanych liczb w P + ziarno
lambdaP = 3             #  parametr Poisonna
        
# GENERATOR W INIT      
randomNumbersW = []     
aW = 16807              #  pierwszy parametr generatora
modW = 2147483647       #  drugi parametr modulo generatora
seedW = 1               #  ziarno generatora
amountW = 10            #  ilosc docelowo wygenerowanych liczb w W + ziarno
        
# GENERATOR N INIT      
randomNumbersN = []     
aN = 16807              #  pierwszy parametr generatora
modN = 2147483647       #  drugi parametr modulo generatora
seedN = 1               #  ziarno generatora
amountN = 10            #  ilosc docelowo wygenerowanych liczb w N + ziarno


# Generowanie liczb z G
randomNumbersG = G(aG, modG, seedG, amountG)

# Generowanie liczb z J
randomNumbersJ = J(aJ, modJ, seedJ, amountJ)

# Generowanie liczb z B
randomNumbersB = B(aB, modB, seedB, amountB, pB)

# Generowanie liczb z D
randomNumbersD = D(aD, modD, seedD, pD, nD)

# Generowanie liczb z P
randomNumbersP = P(aP, modP, seedP, lambdaP, amountP)

# Generowanie liczb z W
randomNumbersW = W(aW, modW, seedW, amountW)

# Generowanie liczb z N
randomNumbersN = N(aN, modN, seedN, amountN)


rysujWykres(randomNumbersN, "ROZKŁAD NORMALNY", 100)
rysujWykres(randomNumbersW, "ROZKŁAD WYKŁADNICZY", 100)



print("GENERATOR G TEST SERII")
testSeriiLong(randomNumbersG)
print("GENERATOR J TEST SERII")
testSeriiLong(randomNumbersJ)
print("GENERATOR B TEST SERII")
testSeriiLong(randomNumbersB)
print("GENERATOR D TEST SERII")
testSeriiLong(randomNumbersD)
print("GENERATOR P TEST SERII")
testSeriiLong(randomNumbersP)
print("GENERATOR W TEST SERII")
testSeriiLong(randomNumbersW)
print("GENERATOR N TEST SERII")
testSeriiLong(randomNumbersN)

randomChiB = B(aB, modB, seedB, 10000, pB)

randomChiP = P(aP, modP, seedP, lambdaP, 10000)

print()
print("BERNOULLI CHI-KWADRAT TEST 10000", end='\n-------------------------------------------------------\n')
chiKwadratB(randomChiB, pB)
print("POISSON CHI-KWADRAT TEST 10000", end='\n-------------------------------------------------------\n')
chiKwadratP(randomChiP, lambdaP)

randomChiB = B(aB, modB, seedB, 50000, pB)

randomChiP = P(aP, modP, seedP, lambdaP, 50000)

print()
print("BERNOULLI CHI-KWADRAT TEST 50000", end='\n-------------------------------------------------------\n')
chiKwadratB(randomChiB, pB)
print("POISSON CHI-KWADRAT TEST 50000", end='\n-------------------------------------------------------\n')
chiKwadratP(randomChiP, lambdaP)


randomChiB = B(aB, modB, seedB, 110000, pB)

randomChiP = P(aP, modP, seedP, lambdaP, 110000)

print()
print("BERNOULLI CHI-KWADRAT TEST 110000", end='\n-------------------------------------------------------\n')
chiKwadratB(randomChiB, pB)
print("POISSON CHI-KWADRAT TEST 110000", end='\n-------------------------------------------------------\n')
chiKwadratP(randomChiP, lambdaP)