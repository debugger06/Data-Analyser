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


matchers = ['Net','Val','Amnt','Amt','Avg','Num','Tot','Avail','Min','Max']
matchers2 = ['Typ','Type']


form_class = uic.loadUiType("main.ui")[0]                 # Load the UI

form = uic.loadUiType("dialog.ui")[0]
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
		if self.predictModel == "Binary_Logistics_Regression":
			self.LogisticRegression(xNames,yName)
		if self.predictModel == "Linear_Regression":
			self.LinearRegression(xNames,yName)

		if self.predictModel == "Stochastic_Gradient_boosting":
			self.stochasticGradienBoosting(xNames,yName)


	def stochasticGradienBoosting(self,xNames,yNames):
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
		est = GradientBoostingRegressor(n_estimators=d["n_estimators"], max_depth=d["max_depth"], learning_rate=d["learning_rate"],loss=d["loss"], random_state=d["random_state"])
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
		data['intercept'] = 1.0
		columns = data.columns
		y = columns[0]
		x = columns[1:len(columns)]
		print data.head()
		logit = sm.Logit(data[y], data[x])
		result = logit.fit()
		#print result.summary()
		print result.summary()
		#print result.conf_int()
		#print np.exp(result.params)
		
		

		a = data.describe()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = data.std()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = data.hist()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = result.summary()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = result.conf_int()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = np.exp(result.params)
		self.textEdit.append(str(a))
		self.textEdit.append("\n")
		#self.predictModel=""
		


	def LinearRegression(self,xNames,yName):
		#y,X = dmatrices('admit ~ gre + gpa + C(rank)',self.data, return_type = "dataframe")
		#print X.columns
		#X = pd.DataFrame(x, columns = xNames)
		#Y = pd.DataFrame(y, columns = yName)
		#print X,Y

		df = self.data[yName+xNames]
		print df.head()

		for i in xNames:
			if self.var_type[i]=="Categorical":
				df[i] = pd.Categorical(df[i]).labels
		print df.describe()
		print df.std()
		print df.hist()
		#plt.show()
		#df['intercept'] = 1.0
		
		
		logit = sm.OLS(df[yName], df[xNames])
		result = logit.fit()
		print result.summary()
		print result.conf_int()
		print np.exp(result.params)

		a = df.describe()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = df.std()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = df.hist()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = result.summary()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = result.conf_int()
		self.textEdit.append(str(a))
		self.textEdit.append("\n")

		a = np.exp(result.params)
		self.textEdit.append(str(a))
		self.textEdit.append("\n")
		#self.predictModel=""




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
		self.actionRead_CSV.triggered.connect(self.readCSV)
		self.actionRead_Excel.triggered.connect(self.readExcel)
		self.actionRead_SPSS.triggered.connect(self.readSAV)
		self.actionRead_SAS.triggered.connect(self.readSAS)
		self.actionRead_STATA.triggered.connect(self.readSTATA)
		self.actionRead_Tab_Delimited.triggered.connect(self.readTabCSV)
		self.actionCustomer_Acquisition_Binary_Logistics.triggered.connect(self.Binary_Logistics_Regression)
		self.actionBusiness_Forcast_Linear.triggered.connect(self.Linear_Regression)
		self.actionStochastic_Gradient_Boosting.triggered.connect(self.Stochastic_Gradient_boosting)

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


		raw_data = spss.SavReader(str(self.filename), returnHeader = True) # This is fast
		raw_data_list = list(raw_data) # this is slow
		data = pd.DataFrame(raw_data_list) # this is slow
		data = data.rename(columns=data.loc[0]).iloc[1:]
		print data.head()





		"""
		raw_data_list = list(raw_data) # this is slow
		data = pd.DataFrame(raw_data_list) # this is slow
		data = data.rename(columns=data.loc[0]).iloc[1:] # setting columnheaders, this is slow too.

		print data.head()


		
		self.setUpModel()
		b= []

		savFileName = "ACME Predicter sample data - SPSS.sav"
		with savReaderWriter.SavReader(savFileName, returnHeader=True) as reader:
			header = next(reader)

			for row in reader:
				items = [QtGui.QStandardItem(str(field)) for field in row]
				self.model.appendRow(items)
				
				for ii in row:
					rr = []
					if self.str_to_type(ii)==float:
						rr.append(float(ii))
					elif self.str_to_type(ii)==int:
						rr.append(int(ii))
					else:
						rr.append(ii)

				b.append(row)
			for j in header:
				self.headerName.append(j)
		
		self.comboBox.clear()
		for text in self.headerName:
			self.comboBox.addItem(text)
		self.model.setHorizontalHeaderLabels(self.headerName)
		self.data = pd.DataFrame.from_records(b, columns=self.headerName)
		"""
	"""
	def read_csv(self,file_path):
		header = []
		i=0
		with open(file_path) as f:
			#h = str(f.readline())
			#head = header.split(",")
			#header.append([str(x) for x in head])
			#print h


			data = []

			for line in f:
				line = line.strip().split(",")
				if i == 0:
					header.append([str(x) for x in line])
					i=1

				data.append([str(x) for x in line])
		#print data
		#return 0
		return pd.DataFrame.from_records(data, columns=header)
	"""
	def read_csv(self, file_path):
		csv_chunks = pd.read_csv(str(file_path), chunksize = 10000)
		df = pd.concat(chunk for chunk in csv_chunks)
		print df.head()
		return df
	def removeNonAscii(self,s):
		return "".join(i for i in s if ord(i)<128)

		

	def loadCsv(self):

		#self.setUpModel()

		df = self.read_csv(self.filename)

		#df  = self.read_csv(str(self.filename))


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
		df.columns = l
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


	def createNonselectedTable(self):
		head = self.tableWidget.horizontalHeader()
		head.setStretchLastSection(True)

		self.tableWidget.setRowCount(len(self.nonSelectedVariables))
		self.tableWidget.setColumnCount(2)
		print len(self.nonSelectedVariables)

		for i in range(0,len(self.nonSelectedVariables)):
			print self.nonSelectedVariables[i]
			item = QtGui.QTableWidgetItem(self.nonSelectedVariables[i])
			print item
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
		print self.var_type
	def categoryChangedSelected(self,rowIndex, comboBoxIndex):
		itemname = self.tableWidget_2.item(rowIndex,0)
		self.var_type[str(itemname.text())]=self.category[comboBoxIndex]
		print self.var_type


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