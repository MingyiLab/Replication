import numpy as np
import scipy


#### Recover CRS model locatoin fundamentals
def solveab(params,x0, L, w, H,dist,LL, maxiter = 1e5, error = 1e-6):
    '''
    Recover CRS location fundamentals using fixed point iteration given equilibrium outcomes (wage, labor)

    Parameters
    -------------------
    params: model parameters
    x0 : initial guess, np.array([a, b])
    L : observed labor distribution
    w : observed wage distribution
    H : land area, exogenous
    dist : trade cost
    LL : total labor

    Returns
    -----------------
    locational fundamentals: tuple(nd.array, nd.array)

    '''
    alpha, theta, epsilon = params
    a, b = x0[:,0].reshape(-1,1), x0[:,1].reshape(-1,1)
    
    
    def trade_share():
        nume = a * (dist * w)**(-theta)
        pi = (nume)/(nume.sum(axis=0))
        return pi
    
    def labor_share():
        pinn = np.diag(pi).reshape(-1,1)
        numerator = b * (a/pinn)**(alpha * epsilon/theta)* (L/H)**(-epsilon*(1 - alpha))
        deno = numerator.sum(axis=0)
        L_e = numerator/deno
        L_e *= LL
        return L_e
        
    # define a counter
    count = 0
    while True:
        count += 1
        if count%100 == 0:
            print('Current iteration:',count)
        pi = trade_share()
        inc = w * L 
        expense = pi.dot(inc)
        L_e = labor_share()
        ae = a * (inc/expense)
        a = ae * 0.25 + a * 0.75
        a = a/scipy.stats.gmean(a)
        
        if np.max(np.abs(inc - expense) + np.max(np.abs(L_e - L))) < error:
            print('Unique location fundamentals found......')
            break
        if count > maxiter:
            print('Iteration limit hit......')
            return -1
        
        be = b * L/L_e 
    #         print(np.abs(bi-be).max())
        b = 0.25 * be + 0.75 * b
        b = b/scipy.stats.gmean(b)

    return a, b



#### Recover IRS model locatoin fundamentals
    
def solveHab(params,x0, L, w, H,dist,LL, maxiter = 1e5, error = 1e-6):
    ''' 
    Find the equilibrium labor and wage given the IRS production function.

    Parameters
    ---------------
    params : IRS(Helpman) model parameters, list(alpha, theta, epsilon)
    x0 : initial guess for location fundamentals, np.array([a,b])
    H : fixed land area
    dist : trade cost
    LL : total labor
    
    Returns
    ------------
    tuple(np.array, np.array):
        (exogenous productivity, exogenous amenity)

    '''

    def trade_share():
        nume = L * (a/(dist * w))**(theta)
        pi = (nume)/(nume.sum(axis=0))
        return pi
  
    def labor_share():
        pinn = np.diag(pi).reshape(-1,1)
        numerator = b * a**(alpha * epsilon) * H**(epsilon * (1 - alpha)) * pinn**(-alpha* epsilon/theta) * L**(-((epsilon*(1-alpha))-(alpha*epsilon/theta)))
        deno = numerator.sum(axis=0)
        L_e = numerator/deno
        L_e = L_e * LL
        return L_e
        
    # initialize
    alpha, theta, epsilon = params
    a, b = x0[:,0].reshape(-1,1), x0[:,1].reshape(-1,1)
    
    count = 0
    while True:
        count += 1
        if count%100 == 0:
            print('Current iteration:', count)

        pi = trade_share()
        inc = w * L 
        expense = pi.dot(inc)

        L_e = labor_share()
        if np.max(np.abs(inc - expense) + np.max(np.abs(L_e - L))) < error:
            print('Unique solution found......')
            break
        if count > maxiter:
            print('Iteration limit hit......')
            return -1
        
        ae = a * (inc/expense)**(1/theta)
        a = ae * 0.25 + a * 0.75
        a = a/scipy.stats.gmean(a)
        
        be = b * L/L_e 
        b = 0.25 * be + 0.75 * b
        b = b/scipy.stats.gmean(b)

    return a, b
