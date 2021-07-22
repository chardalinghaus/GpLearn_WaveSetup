# -*- coding: utf-8 -*-
"""
Created on Mon Jun 8 16:22:33 2021

@author: Char
"""

import os

import sys
sys.path.insert(0, 'C:\\Users\Char\Documents\CHAR - Python\Genetic_Programming\GpLearn\MDA')
sys.path.insert(0, 'C:\\Users\Char\Documents\CHAR - Python\Genetic_Programming\GpLearn\StockdonData')
from mda_Char import MaxDiss_Simplified_NoThreshold
from mda_Char import MaxDiss_Simplified_NoThresholdNOR
from mda_Char import DeNormalize
from mdaplot_Char import Plot_MDA_Data

import glob
import pandas as pd
import numpy as np 
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sympy import *
import graphviz
import matplotlib.pyplot as plt

os.chdir('C:\\Users\Char\Documents\CHAR - Python\Genetic_Programming\GpLearn/')

## ==================== Part 1: Reading Data =================================

data = pd.DataFrame()
for file in glob.glob("StockdonData\*.txt"):
    df = pd.read_csv(file, sep="\t", header=None, names=["r2", "n", "Stt", "Sinc", "Sig", "Hs0", "Tp", "tanB", "d50"])
    data = pd.concat([data, df,], ignore_index=True)


## ==================== Part 2: Add New Data =================================

# Wavelength
data ['L0'] = 1.56*(data['Tp']**2)
#same as: data ['L0']= (9.8 * np.power(data.Tp,2)) / (2*math.pi) #import math

#Irribaren Number
data ['Irr'] = data['tanB']/((data['Hs0']/data['L0'])**0.5)

## ====================== Part 3: MDA (Test Data)  ===========================

dataset = data [['n', 'Hs0', 'Tp', 'tanB', 'd50', 'L0']]

# variables to use
vns = ['n', 'Hs0', 'Tp', 'tanB', 'd50', 'L0']

# subset size and scalar index
n_subset = 150            # subset size ~30% data
ix_scalar = [0, 1, 2, 3, 4, 5]        # n, Hs0, Tp, tanB, d50, L0

#MDA return denormalize data - TRAIN and TEST set
trainDENOR, testDENOR  = MaxDiss_Simplified_NoThreshold(data[vns].values[:], n_subset, ix_scalar)
trainDENOR = pd.DataFrame(data=trainDENOR, columns=vns)
testDENOR = pd.DataFrame(data=testDENOR, columns=vns)

#MDA return normalize data - TRAIN and TEST set
trainNOR, testNOR, minis, maxis = MaxDiss_Simplified_NoThresholdNOR(data[vns].values[:], n_subset, ix_scalar)
trainNOR = pd.DataFrame(data=trainNOR, columns=vns)
testNOR = pd.DataFrame(data=testNOR, columns=vns)

# plot classification
Plot_MDA_Data(dataset, trainDENOR).savefig('MDA/MDA.png');
plt.show()

## =================== Part 4: Test and Trainset Data ========================    

X_train = trainNOR [['Hs0', 'Tp', 'tanB', 'd50', 'L0']] #NORMALIZED
y_train = trainDENOR ['n'] #DENORMALIZED
X_test = testNOR [['Hs0', 'Tp', 'tanB', 'd50', 'L0']] #NORMALIZED
y_test = testDENOR ['n']  #DENORMALIZED

## ======================= Part 5: Set Functions ==============================

def pow_2(x):
    f = x**2
    return f
pow_2 = make_function(function=pow_2,name='pow2',arity=1)

def pow_3(x):
    f = x**3
    return f
pow_3 = make_function(function=pow_3,name='pow3',arity=1)

function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'neg']# 'cos', 'sin', 'tan', 'log', 'abs', 'inv', 'max', 'min']
feature_names = ['Hs0', 'Tp', 'tanB', 'd50', 'L0']

## ========================= Part 6: GpLearn =================================

####Run GPlearn model
est_gp = SymbolicRegressor(population_size= 2000, generations=20, #ps at least 500/ g between 15 and 20 (the most productive search is usually performed in early generations)
                           tournament_size=20, stopping_criteria=0.01, #ts more or less? Verify
                           const_range=(-5., 5.), init_depth=(2, 6), 
                           init_method='half and half', function_set=function_set,
                           metric='mean absolute error', parsimony_coefficient=0.0005, #pc=0.01 simple equations - The higher this parameter will be, the shorter the regressor will try to keep the expression.
                           p_crossover=0.7,  p_subtree_mutation=0.1,        #Typically, crossover rate is often 90% or higher. Mutation rate is around 1%. #psm replaces a randomly selected subtree with another randomly created subtree                                                
                           p_hoist_mutation=0.05, p_point_mutation=0.1,     #phm creates a new offspring individual which is copy of a randomly chosen subtree of the parent.#ppm = a node in the tree is randomly selected and randomly changed
                           p_point_replace=0.05, max_samples=0.9, #ms - We’re telling the fitness function to compare the result with 90% of our data.
                           feature_names=feature_names, 
                           warm_start=(False), low_memory=(True),
                           n_jobs=1, verbose=1, random_state=0) #v - Controls the verbosity of the evolution building process (get to see the “out of bag” fitness of the best program)
est_gp.fit(X_train, y_train)

print('R2:',est_gp.score(X_test,y_test))
print('Equation:', est_gp._program)

converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'sqrt': lambda x: x**0.5,
    'pow2': lambda x: x**2,
    'pow3': lambda x: x**3
}

equation = sympify(str((est_gp._program)), locals=converter)
print ('Equation:', equation)


######### Best 3 models of last generation######
# Low_memory must be "False" to work.
df = pd.DataFrame(columns=['Gen','OOB_fitness','Equation'])

for idGen in range(len(est_gp._programs)):
  for idPopulation in range(est_gp.population_size):
    if(est_gp._programs[idGen][idPopulation] != None):
      df = df.append({'Gen': idGen, 'OOB_fitness': est_gp._programs[idGen][idPopulation].oob_fitness_, 'Equation': str(est_gp._programs[idGen][idPopulation])}, ignore_index=True)

print ('Best 3 models of last generation: ')
bestmodels = df[df['Gen']==df['Gen'].max()].sort_values('OOB_fitness')[:3]
print (bestmodels)


## ============= Part 7: Visualize Predicted X Real Setup =============

y_test_predicted = est_gp.predict(X_test)

plt.figure(2)
plt.plot(y_test, '*k', label = 'Tested'); plt.legend(loc = 'best')
plt.plot(y_test_predicted, '.r', label = 'Predicted'); plt.legend(loc = 'best')
plt.xlabel ('Sample ID')
plt.ylabel ('Setup (m)')
plt.title('GpLearn')
plt.show()

#Least squares polynomial fit
plt.figure(3)
plt.plot(y_test, y_test_predicted, '*k')
plt.xlabel ('Real Setup (m)')
plt.ylabel ('Predicted Setup (m)')
plt.title('GpLearn')

n_predd = np.squeeze(y_test_predicted)
# z = np.polyfit(y_test, n_predd, 1)
# p = np.poly1d(z)
# plt.plot(y_test,p(y_test),"r")
# plt.show
# print ("y=%.2fx+(%.2f)"%(z[0],z[1]))

## Starting from zero
x = np.array(y_test) [:, np.newaxis]
a, _, _, _ = np.linalg.lstsq(x, n_predd, rcond=None) #_, _, _ = Ignore other values.
plt.plot(x, a*x, 'b-')
plt.show()

# ###USING THE FORMULA
# #Hs0*Tp + (Hs0*tanB)**0.25*(-d50 + tanB)**0.5
# dataStockNOR=testNOR[['Hs0', 'Tp', 'tanB', 'd50']]
# formula = dataStockNOR['Hs0']*dataStockNOR['Tp'] + (dataStockNOR['Hs0']*dataStockNOR['tanB'])**0.25*(-dataStockNOR['d50'] + dataStockNOR['tanB'])**0.5

# plt.figure(4)
# plt.plot(formula, '*k', label = 'Formula'); plt.legend(loc = 'best')
# plt.plot(y_test_predicted, '.r', label = 'Predicted'); plt.legend(loc = 'best')
# plt.show()

## ============= Part 8: Compare with Stockdon =============

dataStock=testDENOR[['Hs0', 'Tp', 'tanB', 'd50']]
L0 = 1.56*(dataStock['Tp']**2)
n_stock = 0.35*dataStock['tanB']*(dataStock['Hs0']*L0)**0.5

plt.figure(5)
plt.plot(y_test, '*k', label = 'Tested'); plt.legend(loc = 'best')
plt.plot(y_test_predicted, '.r', label = 'Predicted'); plt.legend(loc = 'best')
plt.plot(n_stock, '.c', label = 'Stockdon'); plt.legend(loc = 'best')
plt.title('Stockdon')

plt.xlabel ('Sample ID')
plt.ylabel ('Setup (m)')
plt.show()

#Least squares polynomial fit
plt.figure(6)
plt.plot(y_test, n_stock, '*k')
plt.xlabel ('Real Setup (m)')
plt.ylabel ('Predicted Setup (m)')
plt.title('Stockdon')

n_predd = np.squeeze(n_stock)
# z = np.polyfit(y_test, n_predd, 1)
# p = np.poly1d(z)
# plt.plot(y_test,p(y_test),"r")
# plt.show
# print ("y=%.2fx+(%.2f)"%(z[0],z[1]))

## Starting from zero
x = np.array(y_test) [:, np.newaxis]
a, _, _, _ = np.linalg.lstsq(x, n_predd, rcond=None) #_, _, _ = Ignore other values.
plt.plot(x, a*x, 'b-')
plt.show()


###
#Both Stockdon and Predicted
plt.figure(7)
plt.plot(y_test, n_stock, '*c', label = 'Stockdon'); plt.legend(loc = 'best')
plt.plot(y_test, y_test_predicted, '*r',  label = 'Predicted'); plt.legend(loc = 'best')
plt.xlabel ('Real Setup (m)')
plt.ylabel ('Predicted Setup/Stockdon (m)')

## Starting from zero
x = np.array(y_test) [:, np.newaxis]
a, _, _, _ = np.linalg.lstsq(x, n_predd, rcond=None) #_, _, _ = Ignore other values.
plt.plot(x, a*x, 'b-')
plt.show()

#########################################
from sklearn import linear_model
from sklearn.metrics import r2_score

# Create linear regression object
regr = linear_model.LinearRegression()
print('Coefficient of determination - Predicted: %.2f'
      % r2_score(y_test, y_test_predicted))

print('Coefficient of determination - Stockdon: %.2f'
      % r2_score(y_test, n_stock))