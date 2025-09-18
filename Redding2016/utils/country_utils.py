import numpy as np
import scipy



def solvewl_country(params, Iwest, Ieast,funds, dist, dw, LLwest, LLeast, nobs, maxiter = 1e5, error = 1e-7):
    '''
    Find the equilibrium outcome when the two economies are closed based on the CRS model

    Parameters
    -----------------
    params : model parameters, list(alpha, theta, epsilon)
    Iwest : Indicator, to identify regions in the west, nd.array
    Ieast : Indicator, to identify regions in the east, nd.array
    funds : fundamentals, [a, b, H]
    dist : trade cost
    dw : distance weight, weight = 0 if two regions are in different countries
    LLwest : fixed labor supply in the west
    LLeast : fixed labor supply in the east
    nobs : number of total regions
    
    Returns
    -------------------
    tuple(labor, wage)
        equilibrium labor and equilibrium wage in each location
    '''


    alpha, theta, epsilon = params
    a, b, H = funds
    # Initialize outcome
    L = (np.ones(nobs) * (LLeast+LLwest)/nobs).reshape(-1,1)
    w = np.random.uniform(size = nobs).reshape(-1,1)
    Iwest = Iwest.reshape(-1,1)
    Ieast = Ieast.reshape(-1,1)
    dd=dist**(-theta)
    # weighted trade frictions
    dd=dw * dd
    
    def trade_share():
        # Trade is NOT allowed between countries, that is, dd_{nk} = 0 if n and k are in different countries
        nume = a * w**(-theta)* dd
        pi = (nume/(nume.sum(axis=0)))
        return pi

    def labor_share():
        # Labor can ONLY choose locations within their own country.
        pinn = np.diag(pi).reshape(-1,1)
        # With or without the restriction of labor mobility, we can always derive the choosing probability equation.
        numerator = b * (a/pinn)**(alpha * epsilon/theta) * (L/H)**(-epsilon*(1-alpha))

        L_e = np.zeros_like(L)

        L_e[Iwest==1]=numerator[Iwest==1]/np.sum(numerator[Iwest==1])
        L_e[Ieast==1]=numerator[Ieast==1]/np.sum(numerator[Ieast==1])
        L_e[Iwest==1]=L_e[Iwest==1] * LLwest
        L_e[Ieast==1]=L_e[Ieast==1] * LLeast
        return L_e

    count = 0 
    while True:
        if count % 200 == 0:
            print('Current iteration = ',count)
        count += 1
        pi = trade_share()
        inc = w * L 
        expense = pi.dot(inc)
        Le = labor_share()
        est_error = np.max(np.abs(inc - expense) + np.max(np.abs(Le - L)))

        if est_error < error:
            print('Unique solution found......')
            break
        if count >=maxiter:
            print('Iteration is out of limit......')
            return -1
        

        we = w * (expense/inc)**(1/theta)
        w = we * 0.25 + w * 0.75
        
        w[Iwest==1] = w[Iwest==1]/scipy.stats.gmean(w[Iwest==1])
        if np.min(dw) == 0:
            w[Ieast==1] = w[Ieast==1]/scipy.stats.gmean(w[Ieast==1])
        Le = L *(Le/L)**(1/(epsilon* (1 - alpha)))
        L = 0.25 * Le + 0.75 * L

    return L, w, pi


