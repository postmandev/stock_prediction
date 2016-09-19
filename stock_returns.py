
'''
	Author: Michael O'Meara <michael.omeara@gmail.com>
	date: May 19, 2016
'''

import csv
from datetime import datetime
from numpy import ones, zeros, where
import numpy as np 
from scipy.stats.stats import pearsonr
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

class StockReturns:

	def __init__(self):
		'''
        	Constructor
		'''
		self.X 			= None
		self.y 			= None
		self.myDates	= None
		self.tickers 	= {"S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"}
		self.filename 	= './data/stock_returns_base150.csv'
		self.pred_file	= 'predictions.csv'
		self.cor 		= np.empty( (9,1) )
		self.autoCor 	= np.zeros( (9,1) )
		self.X_train 	= None
		self.X_test  	= None
		self.model 		= None
		self.y_pred		= None
        

	def load(self):

		data = np.genfromtxt(self.filename, dtype=None, delimiter=',', names=True)

		self.myDates = np.array( data[0:100]['date'] )
		self.y = data[0:50]['S1']
		#self.y = self.percentToPrice(data[0:50]['S1'])
		
		self.X = np.empty( (len(self.tickers),100) )

		i = 0
		for t in self.tickers:
			self.X[i,:] = data[0:100][t]
			#self.X[i,:] = self.percentToPrice(data[0:100][t])
			i += 1


	def writeToFile(self, y_pred):

		i = 0
		with open(self.pred_file,'w') as outputfile:
		    wrtr = csv.writer(outputfile, delimiter=',')
		    row = ['Date', 'Value']
		    wrtr.writerow(row)
		    for d in self.myDates:
		        wrtr = csv.writer(outputfile, delimiter=',')
		        row[0] = d
		        row[1] = y_pred[i]
		        wrtr.writerow(row)
		        i += 1



	def calcCorrelation(self):
		

		print '\nCorrelation values with S1: \n'

		N = len(self.tickers)
		self.cor 		= np.empty( (N,1) )
		self.autoCor 	= np.zeros( (N,1) )

		for i in range(len(self.tickers)):
			p = pearsonr(self.y,self.X[i,0:50])
			self.cor[i] = p[0]
			
			if abs(p[0]) > 0.9:
				self.autoCor[i] = self.serial_corr(self.X[i,0:100])

			print '\t%0.0f \tS%i \t%f \t%f' % (i, (i+2), p[0], self.autoCor[i])

		inds = np.argsort(self.autoCor, axis=None)
		
		top_inds = inds[::][0:3]
		M = len(top_inds)
		self.X_train = np.empty( (50,M) )
		self.X_test  = np.empty( (100,M) )
		
		j = 0
		for i in top_inds:
			self.X_train[:,j] = self.X[i, range(50) ]
			self.X_test[:,j]  = self.X[i, range(100)]
			j += 1

		# To insert some random training set
		#self.X_train[:,j] = self.X[8, range(50) ]
		#self.X_test[:,j]  = self.X[8, range(100)]


	def percentToPrice(self, percent):

		N = len(percent)
		price = 1
		price_hist = np.empty( (N) )

		for i in range(N):
			price += (price * (percent[i] / 100))
			price_hist[i] = price

		return price_hist


	def calcPriceChange(self):

		N = len(self.y_pred)
		price1 = 100
		price2 = 100
		price_hist = np.empty( (N+1,1) )
		s1 = np.empty( (len(self.y),1) )

		for i in range(N):
			#if i > 49:
			price1 += (price1 * (self.y_pred[i] / 100))
			price_hist[i] = price1

			if i < 50:
				price2 += (price2 * (self.y[i] / 100))
				s1[i] += price2

		#s1 = s1 - 100;
		#price_hist = price_hist - 100

		print 'Cumulative change in S1 price: %s %%\n' % (price1 - 100)

		return (s1, price_hist)



	def searchParameters(self, X_train, y_train):

		C_range = np.logspace(1, 15, 10)
		gamma_range = np.logspace(-11, -8, 10)
		epsilon_range = np.logspace(-2, -4, 5)
		
		tuned_parameters = [
			{'kernel':['rbf'], 'gamma': gamma_range, 'C': C_range, 'epsilon': epsilon_range}
		]
		grid = GridSearchCV(svm.SVR(C=1), tuned_parameters, cv=10)
		grid.fit(X_train, y_train)

		print "\n", grid.best_estimator_, "\n"

		return grid.best_estimator_
		


	def fit(self, X_train, y_train):
		'''
            Trains the model
            Arguments:
                X_train is a n-by-m array
                y is an n-by-1 array
            Returns:
                No return value

		'''

		self.model = svm.SVR(C=1475000.0, cache_size=200, coef0=0.0, degree=3,
									epsilon=0.1872, gamma=3.500588833612782e-09,
									kernel='rbf', max_iter=-1, probability=False, random_state=None,
									shrinking=True, tol=0.00001, verbose=False)

		self.model.fit(X_train, y_train)


	def predict(self, X):
		'''
			Use the trained model to predict values for each instance in X
			Arguments:
				X is a n-by-m numpy array
		'''
		self.y_pred = self.model.predict(X)

		print '\nMean squared error of predicted vs. training data:', mean_squared_error(self.y_pred[0:50], self.y), '\n'




	def serial_corr(self, X, lag=1):
		'''
			Calculate autocorrelation
			Arguements:
				X is a n-by-1 numpy array
				lag is 1
			Returns:
				corr is a float
		'''
		N  = len(X)
		y1 = X[lag:]
		y2 = X[:N-lag]
		corr = np.corrcoef(y1, y2, ddof=0)[0, 1]
	    
		return corr


	def plot(self):

		s1, price_hist = self.calcPriceChange()
		#s1 = self.y
		#price_hist = self.y_pred

		# Create x-axis values for plotting
		res = sum(abs(self.y-self.y_pred[0:50]))

		print 'Sum(abs(Actual-Predicted)): %f \n' % (res) 

		plt.figure(1)
		plt.plot(range(1,51), (price_hist[50:100]-price_hist[50]), 'k-')
		plt.xlabel('Date Index')
		plt.ylabel('(%) Return')
		
		plt.figure(2)
		plt.plot(range(1,51), (price_hist[0:50]-price_hist[0]), 'b-', label=u'Predicted')
		plt.plot(range(1,51), s1 - s1[0], 'r-', label=u'Actual')
		plt.legend(loc='upper left')
		plt.xlabel('Date Index')
		plt.ylabel('(%) Return')
		
		plt.show()


##################################################################
#
#	Main Execution
# 
##################################################################

stockRet = StockReturns()
stockRet.load()
stockRet.calcCorrelation()
stockRet.fit(stockRet.X_train, stockRet.y)

#stockRet.searchParameters(stockRet.X_train, stockRet.y)
stockRet.predict(stockRet.X_test)

stockRet.writeToFile(stockRet.y_pred)
stockRet.plot()


##################################################################




