# Solution of 1D Poisson's Equation with FDM
# Joao Pedro Colaco Romana 2022

import numpy as np
import matplotlib.pyplot as plt

n = 5

def finitedifferences(n):
    xgrid = np.linspace(0, 3, n + 1)  # Divides [0,3] into n sub-intervals
    x_0 = 0
    u_0 = 1
    x_n = 3
    u_n = 1
    h = (x_n - x_0) / n

    def func1(x):
        f1 = []
        for i in range(n + 1):
            f1.append(3 * x[i] - 2)
        return f1

    def func2(x):
        f2 = []
        for i in range(n + 1):
            f2.append(x[i] ** 2 + 3 * x[i] - 2)
        return f2

    def exact1(x):
        u1ex = []
        for i in range(n + 1):
            u1ex.append(-1 / 2 * x[i] ** 3 + x[i] ** 2 + 3 / 2 * x[i] + 1)
        return u1ex

    def exact2(x):
        u2ex = []
        for i in range(n + 1):
            u2ex.append(-1 / 12 * x[i] ** 4 - 1 / 2 * x[i] ** 3 + x[i] ** 2 + 15 / 4 * x[i] + 1)
        return u2ex

    f1 = func1(xgrid)
    f2 = func2(xgrid)
    u1ex = exact1(xgrid)
    u2ex = exact2(xgrid)

    A = np.diag((n - 2) * [-1 / (h ** 2)], -1) + np.diag((n - 1) * [2 / (h ** 2)], 0) + np.diag(
        (n - 2) * [-1 / (h ** 2)], 1)

    f1rhs = func1(xgrid)
    f1rhs[1] += u_0 / (h * h)
    f1rhs[n - 1] += u_n / (h ** 2)
    f1rhs.pop(n)
    f1rhs.pop(0)

    f2rhs = func2(xgrid)
    f2rhs[1] += u_0 / (h * h)
    f2rhs[n - 1] += u_n / (h ** 2)
    f2rhs.pop(n)
    f2rhs.pop(0)

    u1 = np.linalg.solve(A, f1rhs)
    u2 = np.linalg.solve(A, f2rhs)
    u1 = np.concatenate((u_0, u1, u_n), axis=None)
    u2 = np.concatenate((u_0, u2, u_n), axis=None)

    global_error_1 = 0
    for i in range(1, n):
        global_error_1 += abs(u1ex[i] - u1[i]) ** 2
    global_error_1 = np.sqrt(global_error_1 / (n - 1))

    global_error_2 = 0
    for i in range(1, n):
        global_error_2 += abs(u2ex[i] - u2[i]) ** 2
    global_error_2 = np.sqrt(global_error_2 / (n - 1))

    return global_error_1, global_error_2, xgrid, f1, f2, u1ex, u2ex, A, u1, u2


def global_error(n, N):
    global_error_1_all = []
    global_error_2_all = []
    n_all = range(n, N)

    for i in range(n, N):
        global_error_1_all.append(finitedifferences(i)[0])
        global_error_2_all.append(finitedifferences(i)[1])

    figure5 = plt.figure('Figure 5')
    plt.loglog(n_all, global_error_1_all, '-r', label=r"$u_1(x)$")
    plt.loglog(n_all, global_error_2_all, '-b', label=r"$u_2(x)$")
    plt.legend()
    plt.grid()
    plt.xlabel("Number of Spatial Elements")
    plt.ylabel("Mean Squared Error")
    plt.savefig('MSE_N%02d-%02d.png' % (n, N), bbox_inches='tight', dpi = 300)

    plt.show(block='True')


print('1 - FD Method')
print('2 - Rate of Convergence of the FD Method')
option = int(input('Choose an option: '))

if option == 1:
    n = int(input('Number of intervals: '))

    global_error_1, global_error_2, xgrid, f1, f2, u1ex, u2ex, A, u1, u2 = finitedifferences(n)

    # Plot of the Source Functions
    figure1 = plt.figure('Figure 1')
    plt.plot(xgrid, f1, '-or', label=r"$f_1(x)$")
    plt.plot(xgrid, f2, '-ob', label=r"$f_2(x)$")
    plt.legend()
    plt.grid()
    plt.title(r"Source Functions for $N=$" + str(n))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.savefig('source_functions_N%02d.png' % n, bbox_inches='tight', dpi = 300)

    # Plot of the Exact Solutions
    figure2 = plt.figure('Figure 2')
    plt.plot(xgrid, u1ex, '-or', label=r"$u_1^{ex}(x)$")
    plt.plot(xgrid, u2ex, '-ob', label=r"$u_2^{ex}$(x)")
    plt.legend()
    plt.grid()
    plt.title(r"Exact Solutions for $N=$" + str(n))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.savefig('exact_solution_N%02d.png' % n, bbox_inches='tight', dpi = 300)

    # Plot of the Structure of A
    figure3 = plt.figure('Figure 3')
    plt.spy(A, marker='o', color='g')
    plt.grid('on')
    plt.title(r'Structure of Matrix $A$')
    plt.savefig('matrix_structure_N%02d.png' % n, bbox_inches='tight', dpi = 300)

    eigenA = np.linalg.eig(A)
    print('The eigenvalues of matrix A are: '+str(eigenA[0]))

    # Plot of the Numerical and Exact Solutions
    figure4 = plt.figure('Figure 4')
    plt.plot(xgrid, u1ex, '-or', label=r"$u_1^{ex}(x)$")
    plt.plot(xgrid, u2ex, '-ob', label=r"$u_2^{ex}(x)$")
    plt.plot(xgrid, u1, '--or', label=r"$u_1(x)$")
    plt.plot(xgrid, u2, '--ob', label=r"$u_2(x)$")
    plt.legend()
    plt.grid()
    plt.title(r"FDM Solutions for $N=$" + str(n))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.savefig('FDM_solution_N%02d.png' % n, bbox_inches='tight', dpi = 300)

    print('The global error for case 1 is: ' + str(global_error_1))
    print('The global error for case 2 is: ' + str(global_error_2))

    plt.show(block='True')
elif option == 2:
    n = int(input('Minimum number of intervals: '))
    # Maximum number of intervals
    N = int(input('Maximum number of intervals: '))
    global_error(n, N)
else:
    print('Incorrect option, try again')
