#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import minimize
from scipy.stats import rv_continuous
from scipy.integrate import quad
import warnings

def column(matrix, attr):
    if isinstance(matrix[0], a):
        if isinstance(attr, str):
            if attr == 'v':
                attr = 'val'
            if attr == 'm':
                attr = 'minus'
            if attr == 'p':
                attr = 'plus'
            return [getattr(row,attr) for row in matrix]

    elif isinstance(matrix[0], list):
        if isinstance(attr, (int, float)):
            return [row[attr] for row in matrix]
    else:
        return matrix

class a:
    r"""
    This class provide basic arithmetic operations with the values
    in case of asymmetric uncertainties.
    
    Written by Serj Balashev
    email: s.balashev@gmail.com
    
    To import class put a_unc folder in python path folder and just use:
        >>> from a_unc import a
        
    Uncertainties are estimated by minimization of joint likelihood function
    for details see e.g. Barlow 2003 - arXiv:physics/0406120
    
    Supported operations are: +, -, *, /, product by some factor
    
    You can initialize asymmetric value by several ways:
    1. set three values and additional parameters:
        x = a(19, 0.3, 0.2, form)
        x = a(19, 0.2, form)  -- will be symmetric value
             
                 form is optional and set representation format:
                        1. 'd' or 'dec' - for decimal
                        2. 'l' or 'log' - for log10
                        3. if not set - set to be default
                            
    2. using latex style 
        x = a('$19.0^{+0.3}_{-0.2}$', form)
        x = a('$19.0\pm0.2$', form)
        
    3. using list 
        x = a([19.0, 0.3, 0.2], form)
    
    Upper and lower limits also can be specified:
    1. manually:
        x = a(19, form, t='')
                
                where <t> specified limit:
                    t='u'   : for upper limits
                    t='l'   : for lower limits
    2. using latex style:
        x = a('$<19.0$', form)
        
    Example of usage: 
    
    x = a(19,0.3,0.2)
    y = a(2,0.5,0.5,'d')
    z = x * y
    z.log()
    print(z)
    z.plot_lnL() 
    
    
    Some useful methods:
    a.log() - convert to log10 format
    a.dec() - convert to decimal format
    a.default() - convert to default format set by <default_format> variable
    a.latex() - convert to latex format output
    a.lnL(x) - return the ln from Likelihood function at x position
    a.plot_lnL(x) - plot ln of Likelihood function
    
    Parameters:
    default_format - default format for input
    likelihood_approx - type of approximation of asymmetric likelyhood function
                        see lnL method and Barlow 2003 (arXiv:physics/0406120)
    """
    
    def __init__(self, *args, **kwargs):

        self.likelihood_approx = 2
        self.method = 'Nelder-Mead'
        #self.method = 'L-BFGS-B'
        #self.method = 'SLSQP'
        #self.method = 'TNC'
        self.tol = 1e-5

        # if the input is empty       
        if len(args) in [0]:
            self.val = 0
            self.plus = 0
            self.minus = 0
            self.type = 'f'

        if len(args) in [1, 2]:
            # if the input like latex style a($19.00^{+0.10}_{-0.10}$, 'l')        
            if isinstance(args[0], str):
                self.fromtex(args[0])
            # if the input like latex style a([19.00, 0.01, 0.01], 'l')
            elif isinstance(args[0], (list, tuple)):
                self.val = float(args[0][0])
                self.plus = float(args[0][1])
                self.minus = float(args[0][2])
                if self.plus == 0 and self.minus == 0:
                    self.type = 'f'
                else:
                    self.type = 'm'
            elif isinstance(args[0], (float, int, np.floating)):
                self.val = float(args[0])
                if len(args) > 1 and isinstance(args[1], (float, int)):
                    self.plus = float(args[1])
                    self.minus = float(args[1])
                    self.type = 'm'
                else:
                    self.plus = 0
                    self.minus = 0
                    self.type = 'f'
            
                
        # if the input like data style a(19.00, 0.10, 0.10, 'l')
        elif len(args) in [3, 4]:
            self.val = float(args[0])
            self.plus = abs(float(args[1]))
            if isinstance(args[2], (int, float, np.floating)):
                self.minus = abs(float(args[2]))
            elif isinstance(args[2], (str)):
                self.minus = abs(float(args[1]))
            if self.plus == 0 and self.minus == 0:
                self.type = 'f'
            else:
                self.type = 'm'

        if 't' in kwargs.keys():
            self.type = kwargs['t']
        if 'type' in kwargs.keys():
            self.type = kwargs['type']

        if any(x in args for x in ['d', 'dec']):
            self.repr = 'dec'
            self.default_format = self.repr
        elif any(x in args for x in ['l', 'log']):
            self.repr = 'log'
        else:
            if self.plus < 1. and self.minus < 1.:
                self.repr = 'log'
            else:
                self.repr = 'dec'
        self.default_format = self.repr

    def __str__(self):
        if self.repr == 'log':
            if self.type in ['m', 'f']:
                return "{0:.2f} +{1:.2f} -{2:.2f} in {3} format".format(self.val, self.plus, self.minus, self.repr)
            elif self.type == 'u':
                return "<{0:.2f} in {1} format".format(self.val, self.repr)
            elif self.type == 'l':
                return ">{0:.2f} in {1} format".format(self.val, self.repr)
                
        if self.repr == 'dec':
            if self.type in ['m', 'f']:
                return "{0:.2e} +{1:.2e} -{2:.2e} in {3} format".format(self.val, self.plus, self.minus, self.repr)
            elif self.type == 'u':
                return "<{0:.2e} in {1} format".format(self.val, self.repr)
            elif self.type == 'l':
                return ">{0:.2e} in {1} format".format(self.val, self.repr)
    
    def __repr__(self):
        return self.__str__()
    
    def latex(self, eqs=1, f=None, base=None):
        """
        show representation of <a> values in tex
        parameters:
                eqs      : if ==1 add $...$
                f        : format - number of numbers after floating point, if None, than it will be set automatically
                base     : base for dec values                
        """

        #print(self.val, self.plus, self.minus)
        if base == None and self.repr == 'dec':
            base = int(np.log10(abs(self.val)))

        # >>> set automatical number of significant digits, called <f>
        if f == None:
            d = np.asarray([self.plus, self.minus])
            if len(np.nonzero(d)[0]) > 0:
                if self.repr == 'log':
                    f = int(np.round(np.abs(np.log10(np.min(d[np.nonzero(d)]))) + 1.97)) - 1
                if self.repr == 'dec':
                    #f = 1 if np.abs(np.min(d) / self.val) > 0.5 else 2
                    f = 1 if int(np.floor(np.min(d) / (10 ** np.floor(np.log10(np.min(d)))))) > 2 else 2
            else:
                f = 2
            #print(self.repr, f, int(np.round(np.abs(np.log10(np.min(d[np.nonzero(d)]))) + 1.97)), base)

        # >>> recalc values, taking into account significant digits
        if self.repr == 'log':
            val = round(self.val, f)
            plus = round(round(self.val + self.plus, f) - round(self.val, f), f)
            minus = round(round(self.val, f) - round(self.val - self.minus, f), f)

        elif self.repr == 'dec':

            #r = int(f - (np.floor(np.log10(abs(self.val))) - base) - np.floor(np.log10(min(abs(self.plus), abs(self.minus)))))
            if min(abs(self.plus), abs(self.minus)) > 0:
                r = int(f - np.floor(np.log10(min(abs(self.plus), abs(self.minus))))) - 1 + base
            else:
                r = f - 1 + base
            val = round(self.val/10**base, r)
            plus = round(round(self.val/10**base + self.plus/10**base, r) - round(self.val/10**base, r), r)
            minus = round(round(self.val/10**base, r) - round(self.val/10**base - self.minus/10**base, r), r)

        s = 'none'
        if np.isfinite(self.plus) and np.isfinite(self.minus) and self.type == 'm':
            if self.repr == 'log':
                #print(val, plus, minus, f)
                s = "{0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}}".format(val, plus, minus, n=f)
            if self.repr == 'dec':
                if r > 0:
                    s = "{0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}}".format(val, plus, minus, n=r) + base * " \\times 10^{{{b}}}".format(b=base)
                else:
                    s = "{0:{n}d}^{{+{1:{n}d}}}_{{-{2:{n}d}}}".format(int(val), int(plus), int(minus), b=base, n=r) + base * " \\times 10^{{{b}}}".format(b=base)
        else:
            pre = None
            if (not np.isfinite(self.plus) and np.isfinite(self.minus)) or self.type == 'l':
                pre, val = '>', round(self.val - self.minus, f)
            elif (np.isfinite(self.plus) and not np.isfinite(self.minus)) or self.type == 'u':
                pre, val = '<', round(self.val + self.plus, f)
            if pre is not None:
                if self.repr == 'log':
                    #print(self.val, self.plus, self.minus, f)
                    s = "{0:s}{1:.{n}f}".format(pre, val, n=f)
                if self.repr == 'dec':
                    if base == None:
                        base = int(np.log10(abs(val)))
                    if base == 0:
                        s = "{0:s}{1:.{n}f}".format(pre, val/10**base, n=f)
                    else:
                        s = "{0:s}{1:.{n}f} \\times 10^{{{b}}}".format(pre, val/10**base, b=base, n=f)

        la = '$' if eqs == 1 else ''
        return la + s + la
        
    def fromtex(self, text):
        """
        get <a> object from latex expression as 
        "$15.75^{+0.06}_{-0.07}$"
        "$15.75\pm0.07$"
        "$<15.75$"
        parameter:
            - text     : latex expression
                         type: str
        """
        text = text.replace('$', '')
        if text.find('pm') > -1:
            self.val = float(text[:text.find(r'\pm')])
            self.plus = float(text[text.find(r'\pm')+3:])
            self.minus = self.plus
            self.type = 'm'
        elif text.find('<') > -1:
            self.val = float(text.split('<')[1])
            self.plus = 0
            self.minus = 0
            self.type = 'u'
        elif text.find('>') > -1:
            self.val = float(text.split('>')[1])
            self.plus = 0
            self.minus = 0
            self.type = 'l'
        elif text.find('(') > -1 and text.find(')') > -1:
            text = text.replace('(', '').replace(')', '')
            self.val = float(text.split('^')[0])
            self.plus = abs(float(re.findall(r'\^\{([^}]*)\}', text)[0]))/10**(len(text.split('^')[0])-2)
            self.minus = abs(float(re.findall(r'_\{([^}]*)\}', text)[0]))/10**(len(text.split('^')[0])-2)
            self.type = 'm'
        else:
            self.val = float(text.split('^')[0])
            self.plus = abs(float(re.findall(r'\^\{([^}]*)\}', text)[0]))
            self.minus = abs(float(re.findall(r'_\{([^}]*)\}', text)[0]))
            self.type = 'm'
        #print(self.val, self.plus, self.minus)

    def check_type(self):
        if self.type == 'm' and self.plus == 0 and self.minus == 0:
            self.type = 'f'
        if self.type == 'f' and (self.plus != 0 or self.minus != 0):
            self.type = 'm'

    def dec(self):
        if self.repr == 'log':
            if self.type in ['m', 'f']:
                self.plus = 10**(self.val+self.plus) - 10**self.val
                self.minus = 10**self.val - 10**(self.val-self.minus)
            self.val = 10**self.val
            self.repr = 'dec'
        return self
                
    def log(self):
        if self.repr == 'dec':
            #assert self.val > 0 and self.val-self.minus > 0, ('Argument of log is <= 0')
            if self.type in ['m', 'f']:
                self.plus = np.log10(self.val+self.plus) - np.log10(self.val)
                if self.val < 0:
                    assert self.val > 0, ('Argument of log is <= 0')
                elif self.val-self.minus < 0:
                    self.minus = np.inf
                    warnings.warn('Argument of minus boundary is <= 0, set to -np.inf')
                else:
                    self.minus = np.log10(self.val) - np.log10(self.val - self.minus)
            self.val = np.log10(self.val)
            self.repr = 'log'
        return self

    def default(self, fmt=None):
        """
        Print (if fmt in not specified) or set (if fmt is specified) default format
        """
        self.check_type()
        if fmt is None:
            if self.default_format == 'log':
                return self.log()
            if self.default_format == 'dec':
                return self.dec()
        else:
            if fmt in ['log', 'l']:
                self.default_format == 'log'
            elif fmt in ['dec', 'd']:
                self.default_format == 'dec'

    def copy(self):
        return copy.deepcopy(self)

    def __add__(self, other):
        
        self.dec()
        #res = A_unc(0,0,0,typ='d')
        res = copy.deepcopy(self)

        if isinstance(other, a):
            other.dec()
            res.val, res.plus, res.minus = self.val + other.val, np.sqrt(self.plus ** 2 + other.plus ** 2), np.sqrt(self.minus ** 2 + other.minus ** 2)
            if self.type == 'f' or other.type == 'f':
                pass
            if self.type == 'm' and other.type == 'm':
                res = self.mini(self.sum_lnL, other, res)
            other.default()

        elif isinstance(other, (int, float)):
            res.val = self.val + other
            res.plus = self.plus
            res.minus = self.minus
        return res.default()
    
    def __radd__(self, other):
        
        res = self.__add__(other)
        
        return res.default()
        
    def __sub__(self, other):
        
        self.dec()
        #res = A_unc(0,0,0,typ='d')
        res = copy.deepcopy(self)

        if isinstance(other, a):
            other.dec()
            #print(self.plus, other.plus, self.minus, other.minus, np.sqrt(self.plus ** 2 + other.plus ** 2), np.sqrt(self.minus ** 2 + other.minus ** 2))
            res.val, res.plus, res_minus = self.val - other.val, np.sqrt(self.plus ** 2 + other.plus ** 2), np.sqrt(self.minus ** 2 + other.minus ** 2)
            #print(res.val, res.plus, res_minus)
            res = self.mini(self.sub_lnL, other, res)
            other.default()

        elif isinstance(other, (int, float)):
            res.val = self.val - other
            res.plus = self.plus
            res.minus = self.minus
        return res.dec()
        
    def __truediv__(self, other):
        
        self.dec()
        res = copy.deepcopy(self)
        #res = A_unc(0,0,0,typ='d')

        if isinstance(other, a):
            other.dec()
            if self.type == 'm' and other.type == 'm':
                res.val = self.val / other.val
                res.val, res.plus, res.minus = self.val / other.val, min(res.val * 0.9, np.sqrt((self.plus / other.val)**2 + (self.val / other.val ** 2 * other.plus) ** 2)), min(res.val * 0.9, np.sqrt((self.minus / other.val)**2 + (self.val / other.val**2 * other.minus)**2))
                #print('truediv:', res.val, res.plus, res.minus)
                res = self.mini(self.div_lnL, other, res)
            else: 
                res.val = self.val/other.val
                
                if other.type == 'f':
                    res.plus, res.minus = self.plus/other.val, self.minus/other.val 
                if other.type == 'u':
                    if self.type == 'u':
                        raise TypeError('Value is unconstrained: upper limit is divided by upper limit')
                    else:
                        res.type = 'l'
                        
                elif other.type == 'l':
                    if self.type == 'l':
                        raise TypeError('Value is unconstrained: lower limit is divided by lower limit')
                    else:
                        res.type = 'u'
                if res.type in ['u', 'l']:
                    res.plus, res.minus = 0, 0
            other.default()
                
        elif isinstance(other, (int, float)):
            res.val = self.val / other
            res.plus = self.plus / other
            res.minus = self.minus / other
        
        return res.default()
    
    def __rtruediv__(self, other):
        self.dec()
        res = copy.deepcopy(self)
        #res = A_unc(0,0,0,typ='d')
        if isinstance(other, (int, float)):
            res.val = other / self.val
            res.plus = abs(other / (self.val - self.minus) - res.val)
            res.minus = abs(res.val - other / (self.val + self.plus))
            if self.val < 0:
                res.plus, res.minus = res.minus, res.plus
        return res.default()
        
    def __mul__(self, other):

        self.dec()
        res = copy.deepcopy(self)
        #res = A_unc(0,0,0,typ='d')

        if isinstance(other, a):

            other.dec()
            if self.type == 'm' and other.type == 'm':
                res.val, res.plus, res.minus = self.val * other.val,  np.sqrt(self.val**2 * other.plus**2 + self.plus**2 * other.val**2), np.sqrt(self.val**2 * other.minus**2 + self.minus**2 * other.val**2)
                #print(res.val, res.plus, res.minus)
                res = self.mini(self.mul_lnL, other, res)
            else: 
                res.val = self.val * other.val
                if (self.type == 'l' and other.type == 'u') or (self.type == 'u' and other.type == 'l'):
                    raise TypeError('Value is unconstrained: lower limit is multiplied by upper limit')
                else:
                    if other.type != 'm':
                       res.type = other.type
                if res.type in ['u', 'l']:
                    res.plus, res.minus = 0, 0
            other.default()

        elif isinstance(other, (int, float)):
            res.val = self.val * other
            res.plus = self.plus * other
            res.minus = self.minus * other
        
        return res.default()

    def minim(self, delta, func, other):
        m = minimize(func, 1, args=(other, delta), method=self.method, bounds=[(self.tol, None)], options={'ftol':self.tol, 'eps':0.001})
        #print(delta, m.fun, m.x)
        return np.abs(m.fun - 0.5)

    def mini(self, func, other, res):
        res.plus = res.val * (minimize(self.minim, 1 + res.plus / 2 / res.val, args=(func, other), method=self.method, bounds=[(0.999, None)], options={'ftol': self.tol, 'eps': 0.001}).x[0] - 1)
        res.minus = res.val * (1 - minimize(self.minim, 1 - res.minus / 2 / res.val, args=(func, other), method=self.method, bounds=[(None, 0.999)], options={'ftol': self.tol, 'eps': 0.001}).x[0])

        return res

    def sum_lnL(self, x, other, delta):
        return (-1) * self.lnL(delta * (self.val + other.val) - x * other.val) - other.lnL(x * other.val)
    
    def sub_lnL(self, x, other, delta):
        return (-1) * self.lnL(delta * (self.val - other.val) + x * other.val) - other.lnL(x * other.val)
    
    def mul_lnL(self, x, other, delta):
        return (-1) * self.lnL(delta * self.val / x) - other.lnL(x * other.val)
    
    def div_lnL(self, x, other, delta):
        return (-1) * self.lnL(delta * self.val * x) - other.lnL(x * other.val)
        
    def lnL(self, x, ind=2):
        alpha = x - self.val
        if self.type == 'm':
            if self.plus != self.minus:
                # Barlow 2003
                if ind == 1:
                    beta = self.plus / self.minus
                    gamma = self.plus * self.minus / (self.plus + self.minus)
                    return -0.5 * (np.log(1 + alpha / gamma) / np.log(beta)) ** 2
                # two parabola
                if ind == 2:
                    if isinstance(alpha, (int, float)):
                        if alpha > 0:
                            return -0.5 * (alpha / self.plus) ** 2
                        else:
                            return -0.5 * (alpha / self.minus) ** 2
                    else:
                        ret = np.zeros_like(alpha)
                        ret[alpha>=0] = -0.5 * (alpha[alpha>=0] / self.plus) ** 2
                        ret[alpha<0] = -0.5 * (alpha[alpha<0] / self.minus) ** 2
                        return ret
                # Another from Barlow arXiv:physics/0406120
                if ind == 3:
                    return -0.5 * (alpha * (self.plus + self.minus) / (2 * self.plus * self.minus + (self.plus - self.minus) * alpha)) ** 2
                # Another from Barlow arXiv:physics/0406120
                if ind == 4:
                    return -0.5 * alpha**2 / (self.plus * self.minus + (self.plus - self.minus) * alpha)
            else:
                #print (((x-n[0])/n[1])**2)
                return -0.5 * (alpha / self.plus) ** 2

        elif self.type in ['l', 'u']:
            d = {'l': -1, 'u': 1}
            return np.log(np.abs((np.sign(alpha) - d[self.type]) / 2))


    def sigma(self, x):
        print(x, self.val, self.plus, self.minus)
        return (x - self.val) / self.plus if x - self.val > 0 else (self.val - x) / self.minus

    def L(self, x, ind=2):
        return np.exp(self.lnL(x, ind=ind))

    def plot_lnL(self):
        self.default()
        z = np.linspace(self.val-3*self.minus,self.val+3*self.plus,100)
        y = [self.lnL(z) for z in z]
        plt.plot(z,y)
        plt.axhline(-0.5, c='k', ls='--')

    def rvs(self, n=1, repr='log'):
        """
        Generates the sample drawn from the likelihood related to <a> object
        parameters:
            - n          :  length of the sample
            - repr       :  log or dec (for generating pdf)
        return:
            - sample     :  1d np.array with the sample of n
        """
        getattr(self, repr)()
        d = distr(self, a=self.val - 4 * self.minus, b=self.val + 4 * self.plus)

        return d.rvs(size=n)

class distr(rv_continuous):

    def __init__(self, y, a, b):
        super().__init__(a=a, b=b)
        self.y = y
        self.norm = quad(self.y.L, a=a, b=b)

    def _pdf(self, x):
        return 1 / self.norm[0] * self.y.L(x)

if __name__ == '__main__':

    print("Running tests ...")

    #print(a(2,4,4, 'd').log())

    if 0:
        x = a("$11.37\pm0.03", 'l')
        x = a(13.78, 0.16, 0.40, 'l')
        y = a("$14.90\pm0.15", 'l')
        y = a(13.85, 0.50, 0.50, 'l')
        print(x.dec())
        print(y.dec())
        #x = a("$1\pm0.1", 'd')
        #y = a("$10\pm3", 'd')
        #y = a('$2\pm0.3$', 'd')
        z = (x + y)
        plt.show()
        print(z.log())
        print(z.dec())

    if 0:
        for d in np.linspace(0.01, 0.40, 10):
            #print(d, 47.78 + d, a(47.78, d, 2.40, 'l').log().latex())
            print(d, (a(47.78, d, 0.32, 'l')).latex(f=None, base=0))

    if 0:
        x = a(10, t='u')
    if 0:
        x = a("1.06^{+0.44}_{-0.10}")
        print(x.dec())
        x = a('2.42^{+0.34}_{-0.07}')
        print(x.dec())

    if 0:
        x = a("$19\pm0.2", 'l')
        x = a(19, 0.2, 0.1)
        n = [x]*3
        print(sum(n))
        y = 2e18
        print(x - y)
        #d = sum(n)
        #print(d.latex())
        y = a(2,1.7,0.5, 'd')
        z = x * y
        z.log()
        
        x = a(19,0.2,0.1)
        y = a(18.5,0.3,0.15)    
        z = a(19.1,0.1,0.3)
        w = a(19.3,0.2,0.1)
        z.plot_lnL()
        y.plot_lnL()
        z.plot_lnL()
        f = x + y
        e = z + w
        print((f + e).log())
        print(((x+z) + (y+w)).log())
    
    if 0:
        # various types how to set a value
        x = []
        x.append(a(1))
        x.append(a(2,'d'))
        x.append(a(3,1))
        x.append(a(4,0,'d'))
        x.append(a("5\pm2"))
        x.append(a("6^{+0.1}_{-0.2}"))
        x.append(a("7^{+0.1}_{-0.3}", 'l'))
        x.append(a([8,1,2]))
        x.append(a([9,1,2],'l'))
        x.append(a(10,1,2))
        x.append(a(11,1,2,'d'))
        for y in x:
            print(y, y.type)
    if 0:
        u = a('<19.0', 'l')
        l = a('<2.0', 'l')
        print(l.lnL(3), l.lnL(1))
        m = a('18.04^{+0.17}_{-0.14}', 'd')
        print(m.latex(f=3, base=0))
        print(u * m, u * u, m * u, l * m)

    if 0:
        print('hmmm')
        print(a(14.26, 0.02, 0.02) + a(14.02, 0.02, 0.02) + a(13.10, 0.02, 0.02))
    if 0:
        print(a(20.06, 0.10, 0.07, 'l') + a(19.99, 0.10, 0.11, 'l'))
        print(a(15.01, 0.47, 0.47, 'l') + a(14.55, 0.38, 0.38, 'l'))

    if 1:
        print((a(16.75, 0.02, 0.02) - a(20.92, 0.09, 0.09, f='l')).dec().latex(f=2, base=-5))
