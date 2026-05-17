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



def AnalizeSmallCrossover(GeneralExpression: callable, SmallExpression: callable, AccurateExpression: callable, xMin: float, xMax: float):
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

    print('Crossover value:    %.4f'%crossoverX)
    print('Max relative error: %.4e'%maxRelativeError)

    plt.figure()
    plt.title('Relative errors')
    plt.loglog(x, generalRelativeError, label = 'General expression error')
    plt.loglog(x, smallRelativeError, label = 'Small expression error')
    plt.legend()


def AnalizeBigCrossover(GeneralExpression: callable, BigExpression: callable, xMin: float, xMax: float):
    numberEvals = 1000
    x = np.linspace(xMin, xMax, numberEvals)

    general = GeneralExpression(x)
    big = BigExpression(x)

    difference = np.abs((general - big)/big)

    plt.figure()
    plt.title('Relative difference between general and big expresssions')
    plt.yscale('log')
    plt.plot(x, difference)


AnalizeSmallCrossover(Mw0w0General, Mw0w0Small, Mw0w0Accurate, 1e-10, 2)
AnalizeSmallCrossover(Mw0w1General, Mw0w1Small, Mw0w1Accurate, 1e-10, 2)
AnalizeSmallCrossover(Mr0w0General, Mr0w0Small, Mr0w0Accurate, 1e-10, 2)
AnalizeSmallCrossover(Mr0w1General, Mr0w1Small, Mr0w1Accurate, 1e-10, 2)
AnalizeBigCrossover(Mw0w0General, Mw0w0Big, 20, 200)
AnalizeBigCrossover(Mw0w1General, Mw0w1Big, 20, 200)
AnalizeBigCrossover(Mr0w0General, Mr0w0Big, 20, 200)
AnalizeBigCrossover(Mr0w1General, Mr0w1Big, 20, 200)

plt.show()
