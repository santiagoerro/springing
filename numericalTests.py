import numpy as np
import matplotlib.pyplot as plt



def Mw0w0General(a):
    expa = np.exp(a)
    exp2a = np.exp(2 * a)
    exp3a = np.exp(3 * a)
    exp4a = np.exp(4 * a)
    return (2*a**3*exp4a + 4*a**3*exp3a + 24*a**3*exp2a + 4*a**3*expa + 2*a**3 - 15*a**2*exp4a - 24*a**2*exp3a + 24*a**2*expa + 15*a**2 + 36*a*exp4a - 36*a*exp3a - 36*a*expa + 36*a - 18*exp4a + 36*exp3a - 36*expa + 18)/(6*a**3*(a**2*exp4a - 2*a**2*exp2a + a**2 - 4*a*exp4a + 8*a*exp3a - 8*a*expa + 4*a + 4*exp4a - 16*exp3a + 24*exp2a - 16*expa + 4))

def Mw0w0Small(a):
    return 1/105 - a**2/3150 + 149*a**4/14553000 - 361*a**6/1135134000 + 45691*a**8/4767562800000

def Mw0w0Accurate(a):
    return 1/105 - a**2/3150 + 149*a**4/14553000 - 361*a**6/1135134000 + 45691*a**8/4767562800000 - 13997*a**10/49621572000000

def Mw0w0Big(a):
    return (2*a**3 - 15*a**2 + 36*a - 18)/(6*a**3*(a**2- 4*a+ 4))



def Mw0w1General(a):
    expa = np.exp(a)
    exp2a = np.exp(2 * a)
    exp3a = np.exp(3 * a)
    exp4a = np.exp(4 * a)
    return (-a**3*exp4a - 14*a**3*exp3a - 6*a**3*exp2a - 14*a**3*expa - a**3 + 6*a**2*exp4a + 42*a**2*exp3a - 42*a**2*expa - 6*a**2 - 12*a*exp4a - 60*a*exp3a + 144*a*exp2a - 60*a*expa - 12*a + 18*exp4a - 36*exp3a + 36*expa - 18)/(6*a**3*(a**2*exp4a - 2*a**2*exp2a + a**2 - 4*a*exp4a + 8*a*exp3a - 8*a*expa + 4*a + 4*exp4a - 16*exp3a + 24*exp2a - 16*expa + 4))

def Mw0w1Small(a):
    return -1/140 + a**2/3600 - 559*a**4/58212000 + 509*a**6/1651104000 - 13847*a**8/1466942400000

def Mw0w1Accurate(a):
    return -1/140 + a**2/3600 - 559*a**4/58212000 + 509*a**6/1651104000 - 13847*a**8/1466942400000 + 111149*a**10/396972576000000

def Mw0w1Big(a):
    return (-a**3 + 6*a**2 - 12*a + 18)/(6*a**3*(a**2 - 4*a + 4))



def Mr0w0General(a):
    expa = np.exp(a)
    exp2a = np.exp(2 * a)
    return (2*a**2*exp2a + 2*a**2*expa + 2*a**2 - 9*a*exp2a + 9*a + 12*exp2a - 24*expa + 12)/(6*a**2*(a*exp2a - a - 2*exp2a + 4*expa - 2))

def Mr0w0Small(a):
    return 1/20 - 19*a**2/25200 + 13*a**4/756000 - 109*a**6/258720000 + 28703*a**8/2724321600000

def Mr0w0Accurate(a):
    return 1/20 - 19*a**2/25200 + 13*a**4/756000 - 109*a**6/258720000 + 28703*a**8/2724321600000 - 303689*a**10/1144215072000000

def Mr0w0Big(a):
    return (2*a**2 - 9*a + 12)/(6*a**2*(a - 2))



def Mr0w1General(a):
    expa = np.exp(a)
    exp2a = np.exp(2 * a)
    return (-a*exp2a - 4*a*expa - a + 3*exp2a - 3)/(6*a*(a*exp2a - a - 2*exp2a + 4*expa - 2))

def Mr0w1Small(a):
    return -1/30 + a**2/1575 - a**4/63000 + 59*a**6/145530000 - 7043*a**8/681080400000

def Mr0w1Accurate(a):
    return -1/30 + a**2/1575 - a**4/63000 + 59*a**6/145530000 - 7043*a**8/681080400000 + 12539*a**10/47675628000000

def Mr0w1Big(a):
    return (-a + 3)/(6*a*(a - 2))



sh = 1

def My0w0General(a):
    expa = np.exp(a)
    exp2a = np.exp(2 * a)
    return (-20*a**4*sh*exp2a - 20*a**4*sh*expa - 20*a**4*sh - 21*a**4*exp2a - 18*a**4*expa - 21*a**4 + 90*a**3*sh*exp2a - 90*a**3*sh + 90*a**3*exp2a - 90*a**3 - 120*a**2*sh*exp2a + 240*a**2*sh*expa - 120*a**2*sh - 60*a**2*exp2a + 120*a**2*expa - 60*a**2 - 360*a*exp2a + 360*a + 720*exp2a - 1440*expa + 720)/(60*a**4*(a*exp2a - a - 2*exp2a + 4*expa - 2))

def My0w0Small(a):
    return -11/210 - sh/20 + a**2*(19*sh/25200 + 13/16800) + a**4*(-13*sh/756000 + -4057/232848000) + a**6*(109*sh/258720000 + 25673/60540480000) + a**8*(-28703*sh/2724321600000 + -6299/595945350000)

def My0w0Accurate(a):
    return -11/210 - sh/20 + a**2*(19*sh/25200 + 13/16800) + a**4*(-13*sh/756000 + -4057/232848000) + a**6*(109*sh/258720000 + 25673/60540480000) + a**8*(-28703*sh/2724321600000 + -6299/595945350000) + a**10*(303689*sh/1144215072000000 + 10341733/38903312448000000)

def My0w0Big(a):
    return (-20*a**4*sh- 21*a**4 + 90*a**3*sh + 90*a**3 - 120*a**2*sh - 60*a**2 - 360*a + 720)/(60*a**4*(a - 2))



def Mpsi0w0General(a):
    expa = np.exp(a)
    exp2a = np.exp(2 * a)
    return (-a**4*sh*exp2a/24 - a**4*sh*expa/12 - a**4*sh/24 - a**4*exp2a/20 - a**4*expa/15 - a**4/20 + a**3*sh*exp2a/12 - a**3*sh/12 + a**3*exp2a/12 - a**3/12 + a**2*sh*exp2a/2 + a**2*sh*expa + a**2*sh/2 + a**2*exp2a + a**2 - 2*a*sh*exp2a + 2*a*sh - 5*a*exp2a + 5*a + 2*sh*exp2a - 4*sh*expa + 2*sh + 8*exp2a - 16*expa + 8)/(a**4*(a*exp2a - a - 2*exp2a + 4*expa - 2))

def Mpsi0w0Small(a):
    return -1/105 - sh/120 + a**2*(sh/6720 + 1/6300) + a**4*(-13*sh/3628800 + -1291/349272000) + a**6*(43*sh/479001600 + 97/1064188125) + a**8*(-1483*sh/653837184000 + -130733/57210753600000)

def Mpsi0w0Accurate(a):
    return -1/105 - sh/120 + a**2*(sh/6720 + 1/6300) + a**4*(-13*sh/3628800 + -1291/349272000) + a**6*(43*sh/479001600 + 97/1064188125) + a**8*(-1483*sh/653837184000 + -130733/57210753600000) + a**10*(901*sh/15692092416000 + 420353/7294371084000000)

def Mpsi0w0Big(a):
    return (-a**4*sh/24 - a**4/20 + a**3*sh/12 + a**3/12 + a**2*sh/2 + a**2 - 2*a*sh - 5*a + 2*sh + 8)/(a**4*(a - 2))



def My1w0General(a):
    expa = np.exp(a)
    exp2a = np.exp(2 * a)
    return (-10*a**4*sh*exp2a - 40*a**4*sh*expa - 10*a**4*sh - 9*a**4*exp2a - 42*a**4*expa - 9*a**4 + 30*a**3*sh*exp2a - 30*a**3*sh + 30*a**3*exp2a - 30*a**3 - 60*a**2*exp2a + 120*a**2*expa - 60*a**2 + 360*a*exp2a - 360*a - 720*exp2a + 1440*expa - 720)/(60*a**4*(a*exp2a - a - 2*exp2a + 4*expa - 2))

def My1w0Small(a):
    return -13/420 - sh/30 + a**2*(sh/1575 + 31/50400) + a**4*(-sh/63000 + -3643/232848000) + a**6*(59*sh/145530000 + 24377/60540480000) + a**8*(-7043*sh/681080400000 + -65519/6356750400000)

def My1w0Accurate(a):
    return -13/420 - sh/30 + a**2*(sh/1575 + 31/50400) + a**4*(-sh/63000 + -3643/232848000) + a**6*(59*sh/145530000 + 24377/60540480000) + a**8*(-7043*sh/681080400000 + -65519/6356750400000) + a**10*(12539*sh/47675628000000 + 785809/2992562496000000)

def My1w0Big(a):
    return (-10*a**4*sh - 9*a**4 + 30*a**3*sh + 30*a**3 - 60*a**2 + 360*a - 720)/(60*a**4*(a - 2))



def Mpsi1w0General(a):
    expa = np.exp(a)
    exp2a = np.exp(2 * a)
    return (a**4*sh*exp2a/24 + a**4*sh*expa/12 + a**4*sh/24 + a**4*exp2a/30 + a**4*expa/10 + a**4/30 - a**3*sh*exp2a/12 + a**3*sh/12 - a**3*exp2a/12 + a**3/12 - a**2*sh*exp2a/2 - a**2*sh*expa - a**2*sh/2 - 2*a**2*expa + 2*a*sh*exp2a - 2*a*sh - a*exp2a + a - 2*sh*exp2a + 4*sh*expa - 2*sh + 4*exp2a - 8*expa + 4)/(a**4*(a*exp2a - a - 2*exp2a + 4*expa - 2))

def Mpsi1w0Small(a):
    return 1/140 + sh/120 + a**2*(-sh/6720 + -1/7200) + a**4*(13*sh/3628800 + 2423/698544000) + a**6*(-43*sh/479001600 + -48161/544864320000) + a**8*(1483*sh/653837184000 + 16099/7151344200000)

def Mpsi1w0Accurate(a):
    return 1/140 + sh/120 + a**2*(-sh/6720 + -1/7200) + a**4*(13*sh/3628800 + 2423/698544000) + a**6*(-43*sh/479001600 + -48161/544864320000) + a**8*(1483*sh/653837184000 + 16099/7151344200000) + a**10*(-901*sh/15692092416000 + -6676727/116709937344000000)

def Mpsi1w0Big(a):
    return (a**4*sh/24 + a**4/30 - a**3*sh/12 - a**3/12 - a**2*sh/2 + 2*a*sh - a - 2*sh + 4)/(a**4*(a - 2))



def AnalizeSmallCrossover(GeneralExpression: callable, SmallExpression: callable, AccurateExpression: callable, xMin: float, xMax: float, name: str):
    numberEvals = 1000
    logXMin = np.log(xMin)
    logXMax = np.log(xMax)
    x = np.exp(np.linspace(logXMin, logXMax, numberEvals))

    general = GeneralExpression(x)
    small = SmallExpression(x)
    accurate = AccurateExpression(x)

    generalRelativeError = np.abs((general - accurate)/accurate)
    smallRelativeError = np.abs((small - accurate)/accurate)

    for i in range(numberEvals):
        index = numberEvals - i - 1
        if generalRelativeError[index] > smallRelativeError[index]:
            crossoverIndex = index + 1
            break

    crossoverX = x[crossoverIndex]
    maxRelativeError = smallRelativeError[crossoverIndex]

    print()
    print(name)
    print('Crossover value:    %.4f'%crossoverX)
    print('Max relative error: %.4e'%maxRelativeError)
    print()

    plt.figure()
    plt.title('Relative errors, %s'%name)
    plt.loglog(x, generalRelativeError, label = 'General expression error')
    plt.loglog(x, smallRelativeError, label = 'Small expression error')
    plt.legend()


def AnalizeBigCrossover(GeneralExpression: callable, BigExpression: callable, xMin: float, xMax: float, name: str):
    numberEvals = 1000
    x = np.linspace(xMin, xMax, numberEvals)

    general = GeneralExpression(x)
    big = BigExpression(x)

    difference = np.abs((general - big)/big)

    plt.figure()
    plt.title('Relative difference between general and big expresssions, %s'%name)
    plt.yscale('log')
    plt.plot(x, difference)


AnalizeSmallCrossover(Mw0w0General, Mw0w0Small, Mw0w0Accurate, 1e-10, 2, 'w0 w0')
AnalizeSmallCrossover(Mw0w1General, Mw0w1Small, Mw0w1Accurate, 1e-10, 2, 'w0 w1')
AnalizeSmallCrossover(Mr0w0General, Mr0w0Small, Mr0w0Accurate, 1e-10, 2, 'r0 w0')
AnalizeSmallCrossover(Mr0w1General, Mr0w1Small, Mr0w1Accurate, 1e-10, 2, 'r0 w1')
AnalizeBigCrossover(Mw0w0General, Mw0w0Big, 20, 200, 'w0 w0')
AnalizeBigCrossover(Mw0w1General, Mw0w1Big, 20, 200, 'w0 w1')
AnalizeBigCrossover(Mr0w0General, Mr0w0Big, 20, 200, 'r0 w0')
AnalizeBigCrossover(Mr0w1General, Mr0w1Big, 20, 200, 'r0 w1')

sh = 0.001
AnalizeSmallCrossover(My0w0General, My0w0Small, My0w0Accurate, 1e-10, 2, 'y0 w0, sh = 0.001')
AnalizeBigCrossover(My0w0General, My0w0Big, 20, 400, 'y0 w0, sh = 0.001')

AnalizeSmallCrossover(Mpsi0w0General, Mpsi0w0Small, Mpsi0w0Accurate, 1e-10, 2, 'psi0 w0, sh = 0.001')
AnalizeBigCrossover(Mpsi0w0General, Mpsi0w0Big, 20, 400, 'psi0 w0, sh = 0.001')

AnalizeSmallCrossover(My1w0General, My1w0Small, My1w0Accurate, 1e-10, 2, 'y1 w0, sh = 0.001')
AnalizeBigCrossover(My1w0General, My1w0Big, 20, 400, 'y1 w0, sh = 0.001')

AnalizeSmallCrossover(Mpsi1w0General, Mpsi1w0Small, Mpsi1w0Accurate, 1e-10, 2, 'psi1 w0, sh = 0.001')
AnalizeBigCrossover(Mpsi1w0General, Mpsi1w0Big, 20, 400, 'psi1 w0, sh = 0.001')

sh = 1000
AnalizeSmallCrossover(My0w0General, My0w0Small, My0w0Accurate, 1e-10, 2, 'y0 w0, sh = 1000')
AnalizeBigCrossover(My0w0General, My0w0Big, 20, 400, 'y0 w0, sh = 1000')

AnalizeSmallCrossover(Mpsi0w0General, Mpsi0w0Small, Mpsi0w0Accurate, 1e-10, 2, 'psi0 w0, sh = 1000')
AnalizeBigCrossover(Mpsi0w0General, Mpsi0w0Big, 20, 400, 'psi0 w0, sh = 1000')

AnalizeSmallCrossover(My1w0General, My1w0Small, My1w0Accurate, 1e-10, 2, 'y1 w0, sh = 1000')
AnalizeBigCrossover(My1w0General, My1w0Big, 20, 400, 'y1 w0, sh = 1000')

AnalizeSmallCrossover(Mpsi1w0General, Mpsi1w0Small, Mpsi1w0Accurate, 1e-10, 2, 'psi1 w0, sh = 1000')
AnalizeBigCrossover(Mpsi1w0General, Mpsi1w0Big, 20, 400, 'psi1 w0, sh = 1000')

plt.show()
