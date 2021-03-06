# Temperature-conversion program using PyQt
#import unicodedata
import sys
import csv
from PyQt4 import QtCore, QtGui, uic
from sklearn.linear_model import RandomizedLogisticRegression
import numpy as np

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import xlrd
import savReaderWriter
from sas7bdat import SAS7BDAT
from functools import partial
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
#from PyQt4.QtCore import *
from PyQt4.QtGui import QDialog
import time
import unicodedata
import time
import sklearn.linear_model
from sklearn import feature_selection


matchers = ['Net','Val','Amnt','Amt','Avg','Num','Tot','Avail','Min','Max']
matchers2 = ['Typ','Type']


form_class = uic.loadUiType("main.ui")[0]                 # Load the UI

form = uic.loadUiType("dialog.ui")[0]


class LazyDict(dict):
	def keylist(self, keys, value):
		for key in keys:
			self[key] = value



class StochasticDialog(QDialog,form):
    def __init__(self, parent = None):
        super(StochasticDialog, self).__init__(parent)

        self.setupUi(self)

    # get current date and time from the dialog
    def dateTime(self):
        pass

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getParams(parent = None):
    	dialog = StochasticDialog(parent)
        result = dialog.exec_()
        d = {}

        loss=str(dialog.loss.currentText())
        d["loss"]=loss
        learning_rate=float(dialog.learning_rate.text())
        d["learning_rate"]=learning_rate
        n_estimators = int(dialog.n_estimators.text())
        d["n_estimators"] = n_estimators
        subsample = float(dialog.subsample.text())
        d["subsample"] = (subsample)

        d["min_samples_split"] = int(dialog.min_samples_split.text())
        d["min_samples_leaf"] = int(dialog.min_samples_leaf.text())
        d["min_weight_fraction_leaf"] = float(dialog.min_weight_fraction_leaf.text())
        d["max_depth"] = int(dialog.max_depth.text())
        init=None
        d["random_state"] = int(dialog.random_state.currentText())
        max_features = str(dialog.max_features.text())
        if max_features!="None":
        	d["max_features"]=int(dialog.max_features.text())

        d["alpha"] = float(dialog.alpha.text())
        d["verbose"] = int(dialog.verbose.text())
        max_leaf_nodes=str(dialog.max_leaf_nodes.text())
        if max_leaf_nodes!="None":
        	d["max_leaf_nodes"]=int(dialog.max_leaf_nodes.text())
        warm_start=str(dialog.warm_start.currentText())
        if warm_start!="False":
        	d["warm_start"]=False
        else:
        	d["warm_start"]=True

        #print loss,learning_rate,n_estimators,subsample,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_depth,random_state,max_features,alpha,verbose,max_leaf_nodes,warm_start

        return d

class MyWindowClass(QtGui.QMainWindow, form_class):

	filename = ""
	predictModel = ""
	headerName = []
	runnable = {}
	nonSelectedVariables = []
	SelectedVariables = []
	#data = np.zeros((0,0),float)
	data = pd.DataFrame()
	temp={}
	var_type = {}
	category = ["Categorical","Numerical"]



	def __init__(self, parent=None):

		QtGui.QMainWindow.__init__(self, parent)
		self.setupUi(self)
		self.setupToolbar()
		self.resetTemp()
		self.setUpModel()
		self.label.setVisible(False)


		self.toolButton_2.clicked.connect(self.removeFromNonSelected)
		self.toolButton.clicked.connect(self.removeFromSelected)
		self.pushButton_2.clicked.connect(self.provideSuggestion)
		self.pushButton.clicked.connect(self.generateModel)

		QtCore.QObject.connect(self.tableWidget, QtCore.SIGNAL("clicked(QModelIndex)"), self.cellClickedNonSelectedTable)
		QtCore.QObject.connect(self.tableWidget_2, QtCore.SIGNAL("clicked(QModelIndex)"), self.cellClickedSelectedTable)


		#QtCore.QObject.connect(self.tableView_3, QtCore.SIGNAL("clicked(QModelIndex)"), self.cellClickedNonSelectedTable)
		#QtCore.QObject.connect(self.tableView_5, QtCore.SIGNAL("clicked(QModelIndex)"), self.cellClickedSelectedTable)


	def generateModel(self):

		yName = []

		yName.append(str(self.comboBox.currentText()))
		xNames = self.SelectedVariables

		self.textEdit.clear()

		if self.predictModel == "Business_Forcast_SGB":
			self.stochasticGradienBoostingRegressor(xNames,yName)
		if self.predictModel == "Business_Forcast_Linear":
			self.Business_Forcast_LinearModel(xNames,yName)


		if self.predictModel == "Binary_Logistics_Regression":
			self.LogisticRegression(xNames,yName)
		if self.predictModel == "Linear_Regression":
			self.LinearRegression(xNames,yName)

		if self.predictModel == "Stochastic_Gradient_boosting":
			self.stochasticGradienBoosting(xNames,yName)


		if self.predictModel == "Customer_Acquisition_Binary":
			self.LogisticRegression(xNames,yName)
		if self.predictModel == "Customer_Retention_SGB":
			self.stochasticGradienBoostingClassifier(xNames,yName)

		if self.predictModel == "Customer_Retention_Binary_Logistics":
			self.LogisticRegression(xNames,yName)


	def stochasticGradienBoostingRegressor(self,xNames,yNames):
		d= StochasticDialog.getParams()
		names = yNames+xNames
		dff = self.data[names]

		for i in names:
			if dff[i].dtype == "object":
				dff[i] = pd.DataFrame(data={i: np.unique(dff[i],return_inverse=True)[1]})
		X = np.asfortranarray(dff[xNames], dtype=np.float32)
		Y = np.asfortranarray(dff[yNames], dtype=np.float32)
		y = Y[:,0]
		#est = GradientBoostingRegressor(loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,subsample=subsample,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_depth=max_depth,random_state=random_state,max_features=max_features,alpha=alpha,verbose=verbose,max_leaf_nodes=max_leaf_nodes,warm_start=warm_start)
		est = GradientBoostingRegressor(n_estimators=d["n_estimators"], max_depth=d["max_depth"], learning_rate=d["learning_rate"],loss=d["loss"], random_state=d["random_state"])
		est.fit(X,y)
		a = est.feature_importances_
		self.textEdit.append("Relative Importance of the variables: \n")
		for i in range(0,len(a)):
			self.textEdit.append(str(xNames[i])+" "+str(a[i])+"\n")


	def stochasticGradienBoostingClassifier(self,xNames,yNames):
		d= StochasticDialog.getParams()
		names = yNames+xNames
		dff = self.data[names]
		for i in xNames:
			if dff[i].dtype=='object':
				fillvalue = dff[i].value_counts()
				fillvalue = fillvalue.index[0]
			else:
				fillvalue = np.mean(dff[i])
			dff[i] = dff[i].fillna(fillvalue)
		for i in yNames:
			fillvalue = dff[i].value_counts()
			fillvalue = fillvalue.index[0]
			dff[i] = dff[i].fillna(fillvalue)

		for i in names:
			if dff[i].dtype == "object":
				dff[i] = pd.DataFrame(data={i: np.unique(dff[i],return_inverse=True)[1]})
		X = np.asfortranarray(dff[xNames], dtype=np.float32)
		Y = np.asfortranarray(dff[yNames], dtype=np.float32)
		y = Y[:,0]
		#est = GradientBoostingRegressor(loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,subsample=subsample,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_depth=max_depth,random_state=random_state,max_features=max_features,alpha=alpha,verbose=verbose,max_leaf_nodes=max_leaf_nodes,warm_start=warm_start)
		est = GradientBoostingClassifier(n_estimators=d["n_estimators"], max_depth=d["max_depth"], learning_rate=d["learning_rate"],loss=d["loss"], random_state=d["random_state"])
		est.fit(X,y)
		a = est.feature_importances_
		self.textEdit.append("Relative Importance of the variables: \n")
		for i in range(0,len(a)):
			self.textEdit.append(str(xNames[i])+" "+str(a[i])+"\n")




	def get_dummies(self,df,var_name):
		ss = pd.Series(df[var_name].values.ravel()).unique()
		ss = ss[0:len(ss)]
		d = pd.DataFrame()
		for ii in ss:
			d[var_name+"_"+str(ii)] = df[var_name].apply(lambda x: 1 if x == ii else 0)
			#print var_name+"_"+str(ii)
		#print d.columns
		return d


	def LogisticRegression(self,x,y):
		lst = list(set(y + x))
		ddf = self.data[lst]
		#x=["COST","Gender",'Age','Education']
		#y=["CLICKS"]
		for i in x:
			if ddf[i].dtype=='object':
				fillvalue = ddf[i].value_counts()
				fillvalue = fillvalue.index[0]
			else:
				fillvalue = np.mean(ddf[i])
			ddf[i] = ddf[i].fillna(fillvalue)
		for i in y:
			fillvalue = ddf[i].value_counts()
			fillvalue = fillvalue.index[0]
			ddf[i] = ddf[i].fillna(fillvalue)
		categorical = []
		nonCategorical = []
		for i in x:
			if ddf[i].dtype=="object":
				categorical.append(i)
				print i
			else:
				print i
				nonCategorical.append(i)
		data = ddf[y+nonCategorical]
		for j in categorical:
			#dummy_b = self.get_dummies(ddf,j)
			dummy_b = pd.get_dummies(ddf[j],prefix=j)
			dummy_columns = dummy_b.columns
			cols = list(dummy_columns[1:len(dummy_columns)])
			data[cols] = dummy_b[dummy_columns[1:len(dummy_columns)]]
		#data['intercept'] = 1.0
		columns = data.columns
		y = columns[0]
		x = columns[1:len(columns)]
		print data.head()
		X = data[x].as_matrix()
		Y = data[y].as_matrix()
		model = sklearn.linear_model.LogisticRegression()
		model.fit(X,Y)
		a = model.coef_

		self.textEdit.append(str(a))
		self.textEdit.append("\n")



	def Business_Forcast_LinearModel(self,x,y):
		lst = list(set(y + x))
		ddf = self.data[lst]
		#x=["COST","Gender",'Age','Education']
		#y=["CLICKS"]
		categorical = []
		nonCategorical = []
		for i in x:
			if ddf[i].dtype=="object":
				categorical.append(i)
				print i
			else:
				print i
				nonCategorical.append(i)
		data = ddf[y+nonCategorical]
		for j in categorical:
			#dummy_b = self.get_dummies(ddf,j)
			dummy_b = pd.get_dummies(ddf[j],prefix=j)
			dummy_columns = dummy_b.columns
			cols = list(dummy_columns[1:len(dummy_columns)])
			data[cols] = dummy_b[dummy_columns[1:len(dummy_columns)]]
		#data['intercept'] = 1.0
		columns = data.columns
		y = columns[0]
		x = columns[1:len(columns)]
		print data.head()
		X = data[x].as_matrix()
		Y = data[y].as_matrix()
		model = sklearn.linear_model.LinearRegression()
		model.fit(X,Y)
		a = model.coef_

		self.textEdit.append(str(a))
		self.textEdit.append("\n")


	def LinearRegression(self,x,y):
		lst = list(set(y + x))
		ddf = self.data[lst]
		#x=["COST","Gender",'Age','Education']
		#y=["CLICKS"]
		for i in x:
			if ddf[i].dtype=='object':
				fillvalue = ddf[i].value_counts()
				fillvalue = fillvalue.index[0]
			else:
				fillvalue = np.mean(ddf[i])
			ddf[i] = ddf[i].fillna(fillvalue)
		for i in y:
			fillvalue = ddf[i].value_counts()
			fillvalue = fillvalue.index[0]
			ddf[i] = ddf[i].fillna(fillvalue)
		categorical = []
		nonCategorical = []
		for i in x:
			if ddf[i].dtype=="object":
				categorical.append(i)
				print i
			else:
				print i
				nonCategorical.append(i)
		data = ddf[y+nonCategorical]
		for j in categorical:
			#dummy_b = self.get_dummies(ddf,j)
			dummy_b = pd.get_dummies(ddf[j],prefix=j)
			dummy_columns = dummy_b.columns
			cols = list(dummy_columns[1:len(dummy_columns)])
			data[cols] = dummy_b[dummy_columns[1:len(dummy_columns)]]
		#data['intercept'] = 1.0
		columns = data.columns
		y = columns[0]
		x = columns[1:len(columns)]
		print data.head()
		X = data[x].as_matrix()
		Y = data[y].as_matrix()
		model = sklearn.linear_model.LinearRegression()
		model.fit(X,Y)
		a = model.coef_

		self.textEdit.append(str(a))
		self.textEdit.append("\n")




	def cellClickedNonSelectedTable(self):
		self.temp['row'] = self.tableWidget.currentItem().row()
		self.temp['col'] = self.tableWidget.currentItem().column()
		self.temp['name'] = str(self.tableWidget.currentItem().text())
		print self.temp

	def cellClickedSelectedTable(self):
		self.temp['row'] = self.tableWidget_2.currentItem().row()
		self.temp['col'] = self.tableWidget_2.currentItem().column()
		self.temp['name'] = str(self.tableWidget_2.currentItem().text())
		print self.temp

	def setUpModel(self):
		self.model = QtGui.QStandardItemModel(self)
		self.model_selected = QtGui.QStandardItemModel(self)
		self.model_nonselected = QtGui.QStandardItemModel(self)
		#self.tableView.setModel(self.model)

		#self.tableView_3.setModel(self.model_nonselected)
		#self.tableView_5.setModel(self.model_selected)
		#self.tableView_3.horizontalHeader().setStretchLastSection(True)
		#self.tableView_5.horizontalHeader().setStretchLastSection(True)


		#self.tableView.horizontalHeader().setStretchLastSection(True)


	def setupToolbar(self):
		#Fself.connect(self.comboBox, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.load_select_variable)
		self.actionRead_CSV.triggered.connect(self.readCSV)
		self.actionRead_Excel.triggered.connect(self.readExcel)
		self.actionRead_SPSS.triggered.connect(self.readSAV)
		self.actionRead_SAS.triggered.connect(self.readSAS)
		self.actionRead_STATA.triggered.connect(self.readSTATA)
		self.actionRead_Tab_Delimited.triggered.connect(self.readTabCSV)
		##Stochastic Gradient Boosting technique: dependent variable is a binary variable (yes/no) to predict the probability (from 0 to 1) that a prospect will become a customer.
		self.actionBusiness_Forcast_SGB.triggered.connect(self.Business_Forcast_SGB)
		self.actionBusiness_Forecast_Linear.triggered.connect(self.Business_Forcast_Linear)
		# Logistic Regression technique where the dependent variable is a binary variable (yes/no) to predict the probability (from 0 to 1) that a prospect will become a customer.
		self.actionCustomer_Acquisition_Binary.triggered.connect(self.Customer_Acquisition_Binary)
		# Stochastic Gradient Boosting technique where the dependent variable is a binary variable (yes/no) to predict the probability (from 0 to 1) that a customer will attrite (leave).
		self.actionCustomer_Retention_SGB.triggered.connect(self.Customer_Retention_SGB)
		#Logistic Regression technique where the dependent variable is a binary variable (yes/no) to predict the probability (from 0 to 1) that a customer will attrite.
		self.actionCustomer_Retention_Binary_Logistics.triggered.connect(self.Customer_Retention_Binary_Logistics)
		self.actionCustomer_Acquisition_Binary_Logistics.triggered.connect(self.Binary_Logistics_Regression)
		#self.actionBusiness_Forcast_Linear.triggered.connect(self.Linear_Regression)
		self.actionStochastic_Gradient_Boosting.triggered.connect(self.Stochastic_Gradient_boosting)
		#self.actionStochastic_Gradient_Boosting.triggered.connect(self.Stochastic_Gradient_boosting)
		#self.actionCustomer_Acquition_SGB.triggered.connect(self.Forcast_Linear)

	def provideSuggestion(self):
		y = str(self.comboBox.currentText())
		xList = self.headerName
		xList.remove(y)


		ddf = self.data[xList]
		#x=["COST","Gender",'Age','Education']
		#y=["CLICKS"]
		vardict = LazyDict()
		categorical = []
		nonCategorical = []
		for i in xList:
			if self.runnable:
				if ddf[i].dtype=="object":
					categorical.append(i)
				else:
					vardict.keylist([i], i)
					nonCategorical.append(i)
		df = self.data[nonCategorical]

		for j in categorical:
			dummy_b = pd.get_dummies(ddf[j],prefix=j)
			dummy_columns = dummy_b.columns
			cols = list(dummy_columns[1:len(dummy_columns)])
			vardict.keylist(cols, j)
			df[cols] = dummy_b[dummy_columns[1:len(dummy_columns)]]

		variables =  list(df.columns)
		X = df.as_matrix()
		Y = self.data[y].as_matrix()
		F, pval = feature_selection.f_regression(X, Y)
		final_variables = []
		for i in range(0,len(pval)):
			if(pval[i]<0.05):
				if vardict[variables[i]] not in final_variables:
					final_variables.append(vardict[variables[i]])

		self.SelectedVariables = final_variables
		self.nonSelectedVariables = [x for x in self.headerName if x not in final_variables]
		print self.SelectedVariables
		print self.nonSelectedVariables
		self.createNonselectedTable()
		self.createSelectedTable()



	def Business_Forcast_SGB(self):
		self.predictModel = "Business_Forcast_SGB"
		self.createSelectedTable()
		self.createNonselectedTable()
	def Business_Forcast_Linear(self):
		self.predictModel = "Business_Forcast_Linear"
		self.createSelectedTable()
		self.createNonselectedTable()
	
	def Customer_Acquisition_Binary(self):
		self.predictModel = "Customer_Acquisition_Binary"
		self.createSelectedTable()
		self.createNonselectedTable()
	def Customer_Retention_SGB(self):
		self.predictModel = "Customer_Retention_SGB"
		self.createSelectedTable()
		self.createNonselectedTable()
	def Customer_Retention_Binary_Logistics(self):
		self.predictModel = "Customer_Retention_Binary_Logistics"
		self.createSelectedTable()
		self.createNonselectedTable()



	def loadSTATA(self):
		try:
			df = pd.read_stata(str(self.filename))
		except:
			print "unexpected error occured"
			return
		print df.head()

		l = list(df.columns)
		print df.head()
		head = self.tableWidget_3.horizontalHeader()
		head.setStretchLastSection(True)
		nrow = len(df.index)
		if nrow>100:
			nrow = 100
		else:
			nrow = nrow

		#self.datatable = QtGui.QTableWidget(parent=self)
		self.tableWidget_3.setColumnCount(len(df.columns))
		self.tableWidget_3.setRowCount(nrow)
		for i in range(nrow):
			for j in range(len(df.columns)):
				self.tableWidget_3.setItem(i,j,QtGui.QTableWidgetItem(str(df.iget_value(i, j))))
		self.tableWidget_3.setHorizontalHeaderLabels(l)

		self.headerName = l
		self.nonSelectedVariables = self.headerName
		self.data = df
		st = str(nrow)+" of "+str(len(df.index))+" rows has been shown"
		self.label.setText(st)
		self.label.setVisible(True)
		self.initDict()
		self.initComboBox()


	def loadSAS(self):
		try:
			f = SAS7BDAT(str(self.filename))
			df = f.to_data_frame()
		except:
			print "Unexpected error occured"
			return


		l = list(df.columns)
		print l
		head = self.tableWidget_3.horizontalHeader()
		head.setStretchLastSection(True)
		nrow = len(df.index)
		if nrow>100:
			nrow = 100
		else:
			nrow = nrow

		#self.datatable = QtGui.QTableWidget(parent=self)
		self.tableWidget_3.setColumnCount(len(df.columns))
		self.tableWidget_3.setRowCount(nrow)
		for i in range(nrow):
			for j in range(len(df.columns)):
				self.tableWidget_3.setItem(i,j,QtGui.QTableWidgetItem(str(df.iget_value(i, j))))
		self.tableWidget_3.setHorizontalHeaderLabels(l)

		self.headerName = l
		self.nonSelectedVariables = self.headerName
		self.data = df
		st = str(nrow)+" of "+str(len(df.index))+" rows has been shown"
		self.label.setText(st)
		self.label.setVisible(True)
		self.initDict()
		self.initComboBox()





	def str_to_type (self,s):
		try:
			f = float(s)
			if "." not in s:
				return int
			return float
		except ValueError:
			value = s.upper()
			if value == "TRUE" or value == "FALSE":
				return bool
			return type(s)

	def loadExcel(self):
		#self.setUpModel()
		try:
			xl = pd.ExcelFile(str(self.filename))
		except:
			print "Unexpected error occured"
			return
		sheets = xl.sheet_names
		print sheets[0]
		df = xl.parse(sheets[0])
		print df.head()

		l = list(df.columns)
		print df.head()
		head = self.tableWidget_3.horizontalHeader()
		head.setStretchLastSection(True)
		nrow = len(df.index)
		if nrow>100:
			nrow = 100
		else:
			nrow = nrow

		#self.datatable = QtGui.QTableWidget(parent=self)
		self.tableWidget_3.setColumnCount(len(df.columns))
		self.tableWidget_3.setRowCount(nrow)
		for i in range(nrow):
			for j in range(len(df.columns)):
				self.tableWidget_3.setItem(i,j,QtGui.QTableWidgetItem(str(df.iget_value(i, j))))
		self.tableWidget_3.setHorizontalHeaderLabels(l)

		self.headerName = l
		self.nonSelectedVariables = self.headerName
		self.data = df
		st = str(nrow)+" of "+str(len(df.index))+" rows has been shown"
		self.label.setText(st)
		self.label.setVisible(True)
		self.initDict()
		self.initComboBox()



	def loadSAV(self):

		raw_data = savReaderWriter.SavReader(str(self.filename), returnHeader = True) # This is fast


		raw_data = savReaderWriter.SavReader(str(self.filename), returnHeader = True) # This is fast
		raw_data_list = list(raw_data) # this is slow
		df = pd.DataFrame(raw_data_list) # this is slow
		df = df.rename(columns=df.loc[0]).iloc[1:]
		print df.head()

		l = list(df.columns)
		print df.head()
		head = self.tableWidget_3.horizontalHeader()
		head.setStretchLastSection(True)
		nrow = len(df.index)
		if nrow>100:
			nrow = 100
		else:
			nrow = nrow

		#self.datatable = QtGui.QTableWidget(parent=self)
		self.tableWidget_3.setColumnCount(len(df.columns))
		self.tableWidget_3.setRowCount(nrow)
		for i in range(nrow):
			for j in range(len(df.columns)):
				self.tableWidget_3.setItem(i,j,QtGui.QTableWidgetItem(str(df.iget_value(i, j))))
		self.tableWidget_3.setHorizontalHeaderLabels(l)

		self.headerName = l
		self.nonSelectedVariables = self.headerName
		self.data = df
		st = str(nrow)+" of "+str(len(df.index))+" rows has been shown"
		self.label.setText(st)
		self.label.setVisible(True)
		self.initDict()
		self.initComboBox()


	def read_csv(self, file_path):
		csv_chunks = pd.read_csv(str(file_path), chunksize = 10000)
		df = pd.concat(chunk for chunk in csv_chunks)
		print df.head()
		return df
	def removeNonAscii(self,s):
		return "".join(i for i in s if ord(i)<128)


	def loadCsv(self):

		df = self.read_csv(self.filename)

		"""

		try:
			df  = pd.read_csv(str(self.filename))
		except:
			print self.filename
			print "Unexpected error occured"
			return
		"""

		ll = list(df.columns)
		l = []
		for ii in ll:
			l.append(self.removeNonAscii(ii))
			self.runnable[ii] = True
		df.columns = l
		print len(l)

		for i in l:
			if df[i].dtype == "object" and len(df[i].unique())>30:
				self.runnable[i] = False


		head = self.tableWidget_3.horizontalHeader()
		head.setStretchLastSection(True)
		nrow = len(df.index)
		if nrow>100:

			nrow = 100
		else:
			nrow = nrow

		#self.datatable = QtGui.QTableWidget(parent=self)
		self.tableWidget_3.setColumnCount(len(df.columns))
		self.tableWidget_3.setRowCount(nrow)
		for i in range(nrow):
			for j in range(len(df.columns)):
				self.tableWidget_3.setItem(i,j,QtGui.QTableWidgetItem(str(df.iget_value(i, j))))
		self.tableWidget_3.setHorizontalHeaderLabels(l)


		self.headerName = l
		self.nonSelectedVariables = self.headerName
		self.data = df
		self.fillMissingValue()
		st = str(nrow)+" of "+str(len(df.index))+" rows has been shown"
		self.label.setText(st)
		self.label.setVisible(True)
		self.initDict()
		self.initComboBox()

	#def preparedata(self):





	def fillMissingValue(self):
		for i in self.headerName:
			if self.data[i].dtype=='object':
				fillvalue = self.data[i].value_counts()
				fillvalue = fillvalue.index[0]
			else:
				fillvalue = np.mean(self.data[i])
			self.data[i] = self.data[i].fillna(fillvalue)
		print "Any Missing value: ",self.data.isnull().any().any()


	def initComboBox(self):
		self.comboBox.clear()
		for text in self.headerName:
			self.comboBox.addItem(text)



	def initDict(self):
		for i in self.headerName:
			self.var_type[i]=self.typeofTheVariable(i)
		#print self.var_type


	def typeofTheVariable(self,name):
		#print self.data[name].dtype
		if self.data[name].dtype == "object":
			return "Categorical"
		return "Numerical"

	def clearEverything(self):
		self.data = None
		self.predictModel = ""
		self.headerName = []
		self.nonSelectedVariables = []
		self.SelectedVariables = []
		#data = np.zeros((0,0),float)
		self.temp={}
		self.var_type = {}
		#self.tableView.clearSpans()
		self.tableWidget.clear()
		self.tableWidget_2.clear()
		self.textEdit.clear()
		n = self.tableWidget.rowCount()
		while n>0:
			self.tableWidget.removeRow(0)
			n = self.tableWidget.rowCount()
		n = self.tableWidget_2.rowCount()
		while n>0:
			self.tableWidget_2.removeRow(0)
			n = self.tableWidget_2.rowCount()



	def readCSV(self):
		self.clearEverything()
		try:
			self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File',".","(*.csv)")
		except ValueError:
			print "something Went wrong"
		#print self.filename
		self.loadCsv()
	def readTabCSV(self):
		self.clearEverything()
		try:
			self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File',".","(*.dat)")
		except ValueError:
			print "something Went wrong"
		print self.filename
		self.loadCsv()
	def readExcel(self):
		self.clearEverything()

		try:
			self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File',".","(*.xlsx)")
		except ValueError:
			print "something Went wrong"
		print self.filename
		self.loadExcel()
	def readSAV(self):
		self.clearEverything()
		try:
			self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File',".","(*.sav)")
		except ValueError:
			print "something Went wrong"
		print self.filename
		self.loadSAV()

	def readSAS(self):
		self.clearEverything()
		try:
			self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File',".","(*.sas7bdat)")
		except ValueError:
			print "something Went wrong"


		print self.filename
		self.loadSAS()
	def readSTATA(self):
		self.clearEverything()
		try:
			self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File',".","(*.dta)")
		except ValueError:
			print "something Went wrong"

		print self.filename
		self.loadSTATA()


	def makeDictionary(self):
		a=[]
		h,w = self.data.shape
		for i in range(0,h):
			b={}
			for j in range(0,w):
				b[self.headerName[j]]=self.data[i,j]
			a.append(b)
		#print a



	def Binary_Logistics_Regression(self):
		#self.makeDictionary()
		self.predictModel = "Binary_Logistics_Regression"
		self.createSelectedTable()
		self.createNonselectedTable()
		#self.randomized_Logistic_regression()

	def Linear_Regression(self):
		#self.makeDictionary()
		self.predictModel = "Linear_Regression"
		self.createSelectedTable()
		self.createNonselectedTable()
		#self.randomized_Logistic_regression()
	def Stochastic_Gradient_boosting(self):
		self.predictModel = "Stochastic_Gradient_boosting"
		self.createSelectedTable()
		self.createNonselectedTable()



	def randomized_Logistic_regression(self):
		X = self.data[:,1:len(self.data[0])]
		y = self.data[:,0]
		randomized_logistic = RandomizedLogisticRegression()
		randomized_logistic.fit(X,y)
		a = randomized_logistic.get_support()
		selected = np.where(a)
		#nonSelected = np.where(not a)
		#print selected




	def createSelectedTable(self):
		head = self.tableWidget_2.horizontalHeader()
		head.setStretchLastSection(True)

		self.tableWidget_2.setRowCount(len(self.SelectedVariables))
		self.tableWidget_2.setColumnCount(2)
		self.tableWidget_2.setHorizontalHeaderLabels(['Selected Variables','type'])

		for i in range(0,len(self.SelectedVariables)):
			item = QtGui.QTableWidgetItem(self.SelectedVariables[i])
			self.tableWidget_2.setItem(i, 0, item)
			combo = QtGui.QComboBox()
			for t in self.category:
				combo.addItem(t)
			index = combo.findText(self.var_type[self.SelectedVariables[i]])
			combo.setCurrentIndex(index)
			self.tableWidget_2.setCellWidget(i,1,combo)
			combo.currentIndexChanged.connect(partial(self.categoryChangedSelected, i))


	def createNonselectedTable(self):
		head = self.tableWidget.horizontalHeader()
		head.setStretchLastSection(True)

		self.tableWidget.setRowCount(len(self.nonSelectedVariables))
		self.tableWidget.setColumnCount(2)
		#print len(self.nonSelectedVariables)

		for i in range(0,len(self.nonSelectedVariables)):
			item = QtGui.QTableWidgetItem(self.nonSelectedVariables[i])
			self.tableWidget.setItem(i, 0, item)
			combo = QtGui.QComboBox()
			for t in self.category:
				combo.addItem(t)
			index = combo.findText(self.var_type[self.nonSelectedVariables[i]])
			combo.setCurrentIndex(index)
			self.tableWidget.setCellWidget(i,1,combo)
			combo.currentIndexChanged.connect(partial(self.categoryChangedNonSelected, i))

		self.tableWidget.setHorizontalHeaderLabels(['Non Selected Variables','type'])
	def categoryChangedNonSelected(self,rowIndex, comboBoxIndex):
		itemname = self.tableWidget.item(rowIndex,0)
		self.var_type[str(itemname.text())]=self.category[comboBoxIndex]
		#print self.var_type
	def categoryChangedSelected(self,rowIndex, comboBoxIndex):
		itemname = self.tableWidget_2.item(rowIndex,0)
		self.var_type[str(itemname.text())]=self.category[comboBoxIndex]
		#print self.var_type


	def resetTemp(self):
		self.temp['name'] = ""
		self.temp['row'] = -1
		self.temp['col'] = -1


	def updateSelected(self):
		n = self.tableWidget_2.rowCount()
		for i in range(0,n):
			self.tableWidget.item(i,0)





	def removeFromNonSelected(self):

		if self.temp['name'] !="":
			self.tableWidget.removeRow(self.temp['row'])
			n = self.tableWidget_2.rowCount()
			item = QtGui.QTableWidgetItem(self.temp['name'])
			combo = QtGui.QComboBox()
			for t in self.category:
				combo.addItem(t)
			index = combo.findText(self.var_type[self.temp['name']])
			combo.setCurrentIndex(index)
			self.tableWidget_2.insertRow(n)
			self.tableWidget_2.setItem(n, 0, item)
			self.tableWidget_2.setCellWidget(n,1,combo)
			combo.currentIndexChanged.connect(partial(self.categoryChangedSelected, n))

			self.nonSelectedVariables.remove(self.temp['name'])
			self.SelectedVariables.append(self.temp['name'])

			#print self.nonSelectedVariables
			#print self.SelectedVariables

			self.resetTemp()

	def removeFromSelected(self):
		if self.temp['name'] !="":
			self.tableWidget_2.removeRow(self.temp['row'])
			n = self.tableWidget.rowCount()
			item = QtGui.QTableWidgetItem(self.temp['name'])
			combo = QtGui.QComboBox()
			for t in self.category:
				combo.addItem(t)
			index = combo.findText(self.var_type[self.temp['name']])
			combo.setCurrentIndex(index)
			self.tableWidget.insertRow(n)
			self.tableWidget.setItem(n, 0, item)
			self.tableWidget.setCellWidget(n,1,combo)
			combo.currentIndexChanged.connect(partial(self.categoryChangedNonSelected, n))

			self.SelectedVariables.remove(self.temp['name'])
			self.nonSelectedVariables.append(self.temp['name'])
			print self.SelectedVariables
			#print self.nonSelectedVariables
			#print self.SelectedVariables
			self.resetTemp()
#from tr import DateDialog

app = QtGui.QApplication(sys.argv)
myWindow = MyWindowClass(None)
myWindow.show()
app.exec_()
