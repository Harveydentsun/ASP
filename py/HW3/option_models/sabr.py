    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm
import pyfeng as pf
import scipy.integrate as spint

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None, cp=1):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        price = self.price(strike, spot, texp, sigma, cp)
        return self.bsm_model.impvol(price, strike, spot, texp, cp)
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1, random=False):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if not random:
            np.random.seed(12345)
        n_samples = 10000
        dt = 0.25
        N = int(texp / dt)
        znorm = np.random.normal(size=(N, n_samples))
        ynorm = np.random.normal(size=(N, n_samples))
        wnorm = znorm * self.rho + ynorm * np.sqrt(1 - self.rho ** 2)

        sigma_t = self.sigma if sigma is None else sigma
        s_t = spot
        for t in range(N):
            s_t = s_t * np.exp(sigma_t*np.sqrt(dt)*wnorm[t,:]-0.5*sigma_t**2*dt)
            sigma_t = sigma_t * np.exp(self.vov*np.sqrt(dt)*znorm[t,:] - 0.5*self.vov**2*dt)

        price = np.zeros_like(strike)
        for i in range(len(strike)):
            price[i] = np.exp(-self.intr * texp) * np.mean(np.fmax(cp*(s_t - strike[i]), 0))
        return price

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None, cp=1):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        price = self.price(strike, spot, texp, sigma, cp)
        return self.model_model.impvol(price, strike, spot, texp, cp)
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1, random=False):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if not random:
            np.random.seed(12345)
        n_samples = 10000
        dt = 0.25
        N = int(texp / dt)
        znorm = np.random.normal(size=(N, n_samples))
        ynorm = np.random.normal(size=(N, n_samples))
        wnorm = znorm * self.rho + ynorm * np.sqrt(1 - self.rho ** 2)

        sigma_t = self.sigma if sigma is None else sigma
        s_t = spot
        for t in range(N):
            s_t = s_t + sigma_t*np.sqrt(dt)*wnorm[t,:]
            sigma_t = sigma_t * np.exp(self.vov*np.sqrt(dt)*znorm[t,:] - 0.5*self.vov**2*dt)

        price = np.zeros_like(strike)
        for i in range(len(strike)):
            price[i] = np.exp(-self.intr * texp) * np.mean(np.fmax(cp*(s_t - strike[i]), 0))
        return price

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, cp=1):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        price = self.price(strike, spot, texp, cp)
        return self.bsm_model.impvol(price, strike, spot, texp, cp)
    
    def price(self, strike, spot, texp=None, cp=1, random=False):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if not random:
            np.random.seed(12345)
        n_samples = 10000
        dt = 0.25
        N = int(texp / dt)
        znorm = np.random.normal(size=(N+1, n_samples))
        sigma = np.ones_like(znorm) * self.sigma
        for t in range(N):
            sigma[t+1, :] = sigma[t,:] * np.exp(self.vov*np.sqrt(dt)*znorm[t,:] - 0.5*self.vov**2*dt)
        I_t = spint.simps(sigma ** 2, dx=1, axis=0) * dt
        S_0 = spot * np.exp(self.rho/self.vov*(sigma[-1:]-sigma[0,:])-0.5*self.rho**2*I_t)
        sigma_bs = np.sqrt((1-self.rho**2)*I_t/texp)

        price = np.zeros_like(strike)
        for i in range(len(strike)):
            price[i] = np.mean(bsm.price(strike[i],S_0,texp,sigma_bs,self.intr,self.divr,cp))

        return price

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, cp=1):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        price = self.price(strike, spot, texp, cp)
        return self.model_model.impvol(price, strike, spot, texp, cp)
        
    def price(self, strike, spot, texp=None, cp=1,random=False):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        if not random:
            np.random.seed(12345)
        n_samples = 10000
        dt = 0.25
        N = int(texp / dt)
        znorm = np.random.normal(size=(N+1, n_samples))
        sigma = np.ones_like(znorm) * self.sigma
        for t in range(N):
            sigma[t+1, :] = sigma[t,:] * np.exp(self.vov*np.sqrt(dt)*znorm[t,:] - 0.5*self.vov**2*dt)
        I_t = spint.simps(sigma ** 2, dx=1, axis=0) * dt
        S_0 = spot + self.rho/self.vov*(sigma[-1:]-sigma[0,:])
        sigma_n = np.sqrt((1-self.rho**2)*I_t/texp)

        price = np.zeros_like(strike)
        for i in range(len(strike)):
            price[i] = np.mean(normal.price(strike[i],S_0,texp,sigma_n,self.intr,self.divr,cp))
        return price