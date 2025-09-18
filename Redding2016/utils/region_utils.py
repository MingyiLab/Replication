
import numpy as np
import scipy


def solve_lw(params,funds, nobs,dist,L, maxiter = 1e4, error = 1e-5):
    '''
    Given the simulated exogenous amenity and productivity, solve the unique equilibrium labor and wage.
    
    Parameters
    ------------
    params : calibrated or self-defined parameters  list(alpha, theta, epsilon)
    funds: fundamentals (a,b,H)
    nobs: number of regions
    dist: trading cost
    L : total labor

    Returns
    ----------
    (Labor, Wage, pi)

    '''
    def trade_share():
        nume = a * (dist * wi)**(-theta)
        pi = (nume)/(nume.sum(axis=0))
        return pi
    
    def labor_share():
        pinn = np.diag(pi).reshape(-1,1)
        numerator = b * (a/pinn)**(alpha * epsilon/theta)* (Li/H)**(-epsilon*(1 - alpha))
        deno = numerator.sum(axis=0)
        L_e = numerator/deno
        L_e *= L
        return L_e
        
    
    
    alpha, theta, epsilon = params
    a, b, H = funds
    # initialize, using fixed initialization
    Li = (np.ones(nobs) * L/nobs).reshape(-1,1)
    wi = np.ones(nobs).reshape(-1,1)
    # define a counter
    count = 0
    while True:
        count += 1
        if count%100 == 0:
            print('Current iteration:',count)
        pi = trade_share()
        inc = wi*Li 
        expense = pi.dot(inc)
        L_e = labor_share()
        
        if np.max(np.abs(inc - expense) + np.max(np.abs(L_e - Li))) < error:
            print('Unique solution found......')
            break
        if count > maxiter:
            print('Iteration limit hit......')
            return -1
        we = wi * (expense/inc)**(1/theta)
        wi = we * 0.25 + wi * 0.75
        wi = wi/scipy.stats.gmean(wi.ravel())
        L_e=Li*(L_e/Li)**(1/(epsilon*(1-alpha)))
        Li=(0.25*L_e)+(0.75*Li)
    return Li,wi,pi


# aggregate price index

def price_index(params,funds, pi,w):
    '''
    Calculate aggregate price in each location.
    
    Parameters
    -------- 
    params: (alpha, theta, sigma)
    funds: location fundamentals, list(a, b, H)
    pi: trading share
    w: equilibrium wage

    Returns:
    ------
    price index
    '''
    alpha, theta, sigma = params
    a, b, H = funds
    gammaf = (scipy.special.gamma((theta+1-sigma)/theta))**(-theta)
    pinn = np.diag(pi).reshape(-1,1)
    P = (gammaf * a * w**(-theta)/pinn)**(-1/theta)
    return P

def land_rent(alpha, H, w, L):
    '''
    Return to local land rent.

    Parameters
    ------------
    alpha, H: params
    w: wage
    L:labor

    Returns
    ------------
    land rents
    '''
    return (1 - alpha)/alpha * w * L/H


def real_wage(alpha,w, P, r):
    """
    Calculate the real income at each location

    Parameters
    ---------------
    alpha : model parameter alpha
    w : wage
    P : aggregate price index
    r : land rent

    Returns
    ----------------
    nd.array
        real income
    """
    
    v = w/alpha
    rwage = v/(P**alpha*r**(1 - alpha))
    
    return rwage





def eu(params, funds, L, w, pi, dist, r, P):
    '''
    Calculate the expected utility across the country

    Parameters
    ----------
    params : model parameters, [alpha, theta, epsilon]
    funds : location fundamentals, [a, b, H]
    L : labor
    w : wage
    pi : trade share
    dist : trade cost
    P : aggregate price index
    r : land rent
    '''
   
    alpha, theta, epsilon = params
    a,b,H = funds
    pinn = np.diag(pi)
    v = w/alpha
    u = (v/(P**alpha *r**(1 - alpha)))**epsilon
    delta = scipy.special.gamma((epsilon - 1)/epsilon)
    Ubar = delta * (b.T.dot(u))**(1/epsilon)
    return Ubar



def pseudo_pwelfare(params, pi, Cpi):
    '''
    Calculate the welfare change before and after the transport infrasturcture given labor is completely immobile.

    Parameters
    -------------
    params : model parameters, [alpha, theta]
    pi : observed trade share
    Cpi : counterfactual trade share

    Returns
    -----------
    Pseudo-welfare change
    '''

    alpha, theta = params
    pinn = np.diag(pi)
    Cpinn = np.diag(Cpi)
    return (pinn/Cpinn)**(alpha/theta)



def grid_search_equil(params, funds, nobs, dist,Cdist,LL):
    '''
    Compare the difference between the observed equilbrium and counterfactual equilbrium
    under different model parameters

    Parameters
    -------------------
    params : model parameters, [alpha, theta, epsilon, sigma]
    funds : location fundamentals
    nobs : number of regions
    dist : observed trade cost
    Cdist : counterfactual trade cost
    LL : total labor

    Returns
    ----------------
    tuple(l0, w0, r0, P0, rwage0, l1, w1, r1, P1, rwage1, pseu_welfare)
        l0 means the observed outcome labor, l1 means the counterfactual outcome labor.
    '''
    # extract model parameters and location fundamentals
    alpha, theta, epsilon, sigma = params
    param = [alpha, theta,epsilon]
    a, b, H = funds
    # solve observed equilibrium.
    l0,w0,pi0 = solve_lw(param,funds, nobs, dist, LL)
    r0 = land_rent(alpha, H, w0,l0)
    P0 = price_index([alpha, theta, sigma], funds, pi0, w0)
    rwage0 = real_wage(alpha, w0, P0, r0)

    # solve counterfactual equilibrium.
    l1,w1,pi1 = solve_lw(param,funds, nobs, Cdist, LL)
    r1 = land_rent(alpha, H, w1,l1)
    P1 = price_index([alpha, theta, sigma], funds, pi1, w1)
    rwage1 = real_wage(alpha, w1, P1, r1)

    pseu_welfare = pseudo_pwelfare([alpha, theta], pi0,pi1)
    return l0, w0, r0, P0, rwage0, l1, w1, r1, P1, rwage1, pseu_welfare

    


#####################################
######## Helpman IRS model ############
####################################
def solve_Hlw(params, Hfund, nobs, dist, LL, maxiter = 1e6,error = 1e-5):
    ''' 
    Find the equilibrium labor and wage given the IRS production function.

    Parameters
    ---------------
    params : IRS(Helpman) model parameters, [alpha, theta, epsilon]
    Hfunds : location fundamentals in the helpman setting, [a,b,H]
    nobs : number of regions
    dist : trade cost
    LL : total labor
    
    Returns
    ------------
    tuple(np.array, np.array, np.array):
        (labor, wage, trade share)

    '''
    def trade_share():
        nume = L * (a/(dist * w))**(theta)
        pi = (nume)/(nume.sum(axis=0))
        return pi
  
    def labor_share():
        pinn = np.diag(pi).reshape(-1,1)
        numerator = b * a **(alpha * epsilon) * H**(epsilon * (1 - alpha)) * pinn**(-alpha* epsilon/theta) * L**(-((epsilon*(1-alpha))-(alpha*epsilon/theta)))
        deno = numerator.sum(axis=0)
        L_e = numerator/deno
        L_e = L_e * LL
        return L_e
        
    # initialize
    alpha, theta, epsilon = params
    a, b, H = Hfund
    L = (np.ones(nobs) * LL/nobs).reshape(-1,1)
    w = np.ones(nobs).reshape(-1,1)
    
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
        
        we = w * (expense/inc)**(1/theta)
        w = we * 0.25 + w * 0.75
        w = w/scipy.stats.gmean(w)
        L_e = L*(L_e/L)**(1/(epsilon*(1-alpha)))
        L = (0.25*L_e)+(0.75*L)
        
    return L,w,pi








def Hprice_index(params,a, pi,w, L):
    '''
    Calculate price index base on Helpman IRS setting
    
    Parameters
    -------------
    params : model parameters, [theta, Hsigma, F]
    a : productivity
    pi : trade share
    w : wage
    L : labor
    '''
    theta, Hsigma, F = params
    pinn = np.diag(pi).reshape(-1,1)
    P=(Hsigma/(Hsigma-1))*(w/a)*((L/(Hsigma*F*pinn))**(-1/theta))

    return P



def Hprice_index(params,a, pi,w, L):
    '''
    Calculate price index base on Helpman IRS setting
    
    Parameters
    -------------
    params : model parameters, [theta, Hsigma, F]
    a : productivity
    pi : trade share
    w : wage
    L : labor
    '''
    theta, Hsigma, F = params
    pinn = np.diag(pi).reshape(-1,1)
    P=(Hsigma/(Hsigma-1))* w/a * (L/(Hsigma*F*pinn))**(-1/theta)

    return P



