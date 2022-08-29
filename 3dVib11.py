# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:54:58 2022

@author: jsogh

Read in data from Sunil's paper and create models of cells within the organ of Corti
Then, align OCT image with the model and save the data so that it can be read by Blender
to make a 3D vibratory movie
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage
from PyQt5 import QtWidgets, uic
from scipy.linalg import norm


#from ROIImageGraphicsView import ROIImageGraphicsView

class time():
    # generate a time series to generate a traveling wave.
    def __init__(self):
        self.samplingRate=400000
        self.timeMax=0.0005
        self.numPts=np.uint32(np.floor(self.samplingRate*self.timeMax))
        self.time=np.linspace(0,self.timeMax,self.numPts)

class cochlea():
    # define the mouse cochlear parameters
    def __init__(self):
        self.numSegments=200     # number of cochlear segments
        self.segments=np.arange(self.numSegments,dtype=np.uint32)
        
        # load in the cochlear parameters for a mouse
        # use mouse cochlea frequency-place map from Muller et al., Hearing
        # Research, 2005    d=156.5-82.5*log10(f)  (D is in percent from base)
        self.cochleaLength=5.13  #in mm along the IHC region
        self.cochleaWidth=0.1  #in mm along the IHC region
        self.freqStart=79100
        self.freqEnd=4800
        self.segmentDistance=self.cochleaLength*((self.segments)/self.numSegments)
        self.freqSegments=1000*(10**((100*(self.segments/self.numSegments)-156.5)/(-82.5)))  #CF for each segment

class params():
    def __init__(self,filename):  
        self.filename=filename
        
    def readParams(self):      
        # read in cell length
        self.length=pd.read_excel(self.filename, sheet_name='s6', index_col=0)
        for i in range(len(self.length)):  #clean the data
            x=self.length.iat[i,0]
            y=x.split('> ')
            self.length.iat[i,0]=y[-1]           
            x=self.length.iat[i,1]
            y=x.split()
            self.length.iat[i,1]=y[0]
            x=self.length.iat[i,2]
            y=x.split()
            self.length.iat[i,2]=y[0]           
        self.length['Pos']=self.length['Pos'].astype(float)
        self.length['a']=self.length['a'].astype(float)
        self.length['b']=self.length['b'].astype(float)
      
        # read in radial angles
        self.radialAngle=pd.read_excel(self.filename, sheet_name='s7', index_col=0)
        for i in range(len(self.radialAngle)):  #clean the data
            x=self.radialAngle.iat[i,0]
            y=x.split('> ')
            self.radialAngle.iat[i,0]=y[-1]           
            x=self.radialAngle.iat[i,1]
            y=x.split()
            self.radialAngle.iat[i,1]=y[0]
            x=self.radialAngle.iat[i,2]
            y=x.split()
            self.radialAngle.iat[i,2]=y[0]           
        self.radialAngle['Pos']=self.radialAngle['Pos'].astype(float)
        self.radialAngle['a']=self.radialAngle['a'].astype(float)
        self.radialAngle['b']=self.radialAngle['b'].astype(float)

        # read in longitudinal angles
        self.longitudinalAngle=pd.read_excel(self.filename, sheet_name='s8', index_col=0)
        for i in range(len(self.longitudinalAngle)):  #clean the data
            x=self.longitudinalAngle.iat[i,0]
            y=x.split('> ')
            self.longitudinalAngle.iat[i,0]=y[-1]           
            x=self.longitudinalAngle.iat[i,1]
            y=x.split()
            self.longitudinalAngle.iat[i,1]=y[0]
            x=self.longitudinalAngle.iat[i,2]
            y=x.split()
            self.longitudinalAngle.iat[i,2]=y[0]           
        self.longitudinalAngle['Pos']=self.longitudinalAngle['Pos'].astype(float)
        self.longitudinalAngle['a']=self.longitudinalAngle['a'].astype(float)
        self.longitudinalAngle['b']=self.longitudinalAngle['b'].astype(float)

        # read in intercellular distances
        self.intercellDist=pd.read_excel(self.filename, sheet_name='s9_1', index_col=0)
        for i in range(len(self.intercellDist)):  #clean the data
            x=self.intercellDist.iat[i,0]
            y=x.split('> ')
            self.intercellDist.iat[i,0]=y[-1]           
            x=self.intercellDist.iat[i,1]
            y=x.split()
            self.intercellDist.iat[i,1]=y[0]
            x=self.intercellDist.iat[i,2]
            y=x.split()
            self.intercellDist.iat[i,2]=y[0]           
        self.intercellDist['Pos']=self.intercellDist['Pos'].astype(float)
        self.intercellDist['a']=self.intercellDist['a'].astype(float)
        self.intercellDist['b']=self.intercellDist['b'].astype(float)

        # read in cell diameters
        self.diameter=pd.read_excel(self.filename, sheet_name='s9_2', index_col=0)
        for i in range(len(self.diameter)):  #clean the data
            x=self.diameter.iat[i,0]
            y=x.split('> ')
            self.diameter.iat[i,0]=y[-1]           
            x=self.diameter.iat[i,1]
            y=x.split()
            self.diameter.iat[i,1]=y[0]
            x=self.diameter.iat[i,2]
            y=x.split()
            self.diameter.iat[i,2]=y[0]           
        self.diameter['Pos']=self.diameter['Pos'].astype(float)
        self.diameter['a']=self.diameter['a'].astype(float)
        self.diameter['b']=self.diameter['b'].astype(float)

class cell():
    def __init__(self,cellType): 
        
        # x=longitudinal location
        # y=radial location
        # z=transverse location
        # theta=angle in XY plane
        # phi=azimuth angle (z-angle)
        
        self.cellType=cellType
        self.longLocBottom=0
        self.radLocBottom=0
        self.transLocBottom=0
        self.longLocTop=0
        self.radLocTop=0
        self.transLocTop=0
        self.diameter=0
        self.intercellDist=0
        self.length=0
        self.radialAngle=0
        self.longitudinalAngle=0
        self.length1=0
        self.radialAngle1=0
        self.longitudinalAngle1=0
        self.theta=0
        self.phi=0
        self.color=''
        
    def anatomy(self,pos,p):  #get cell parameters 
            # get position (%) from base (0) to apex (100) and return variables for the orientation of each cell
            # from Soons et al,2015. Cytoarchitecture of the Mouse Organ of Corti from Base to Apex, Determined Using In Situ Two-Photon Imaging
            
            #length
            label='L'+self.cellType
            if (label in list(p.length.index.values)): 
                if pos<=p.length.at[label,'Pos']:
                    pass
                else:
                    label=label+'_1'  # if there are two rows for the fits, make sure the correct row is selected  
                self.length=p.length.at[label,'a']*pos + p.length.at[label,'b']
            
            #radial angles
            label='b'+self.cellType
            if (label in list(p.radialAngle.index.values)): 
                if pos<=p.radialAngle.at[label,'Pos']:
                    pass
                else:
                    label=label+'_1'  # if there are two rows for the fits, make sure the correct row is selected  
                self.radialAngle=p.radialAngle.at[label,'a']*pos + p.radialAngle.at[label,'b']

            #longitudinalAngle
            label='a'+self.cellType
            if (label in list(p.longitudinalAngle.index.values)):            
                if pos<=p.longitudinalAngle.at[label,'Pos']:
                    pass
                else:
                    label=label+'_1'  # if there are two rows for the fits, make sure the correct row is selected  
                self.longitudinalAngle=p.longitudinalAngle.at[label,'a']*pos + p.longitudinalAngle.at[label,'b']

            #intercellDist
            label='D'+self.cellType
            if (label in list(p.intercellDist.index.values)):            
                if pos<=p.intercellDist.at[label,'Pos']:
                    pass
                else:
                    label=label+'_1'  # if there are two rows for the fits, make sure the correct row is selected  
                self.intercellDist=p.intercellDist.at[label,'a']*pos + p.intercellDist.at[label,'b']

            #diameter
            label='d'+self.cellType
            if (label in list(p.diameter.index.values)):            
                if pos<=p.diameter.at[label,'Pos']:
                    pass
                else:
                    label=label+'_1'  # if there are two rows for the fits, make sure the correct row is selected  
                self.diameter=p.diameter.at[label,'a']*pos + p.diameter.at[label,'b']

    def angles(self,cellData):  
        #calculate the beginning and ending points of the cylinder representing the cell
        x=0
        y=0
        z=0
        if self.cellType=='OP':
            x=0
            y=0  
            z=0
            self.longitudinalAngle=90
            self.diameter=self.intercellDist
            self.color='darkturquoise'     
        elif self.cellType=='IP':
            x=0
            z=0
            z1=self.length*np.sin(self.radialAngle*2*np.pi/360)
            y1=-1*z1/np.tan((180-cellData[0].radialAngle)*2*np.pi/360)
            y=y1-z1/np.tan(self.radialAngle*2*np.pi/360)            
            self.longitudinalAngle=90
            self.diameter=self.intercellDist
            self.color='cyan'
        elif self.cellType=='IHC':
            x=0
            y=cellData[0].radLocTop-2*self.intercellDist 
            z=cellData[0].transLocTop-self.length
            self.longitudinalAngle=90
            self.diameter=self.intercellDist
            self.color='red'
        elif self.cellType=='DC1':
            x=0
            y=cellData[0].radLocBottom + 2*self.intercellDist
            z=0
            self.diameter=self.intercellDist
            self.color='green'
        elif self.cellType=='DC2':
            x=0
            y=cellData[3].radLocBottom +self.intercellDist
            z=0
            self.diameter=self.intercellDist
            self.color='green'
        elif self.cellType=='DC3':
            x=0
            y=cellData[4].radLocBottom +self.intercellDist
            z=0
            self.diameter=self.intercellDist
            self.color='green'
        elif self.cellType=='OHC1':
            x=cellData[3].longLocTop
            y=cellData[3].radLocTop
            z=cellData[3].transLocTop
            self.color='orange'
        elif self.cellType=='OHC2':
            x=cellData[4].longLocTop
            y=cellData[4].radLocTop
            z=cellData[4].transLocTop
            self.color='orange'
        elif self.cellType=='OHC3':
            x=cellData[5].longLocTop
            y=cellData[5].radLocTop
            z=cellData[5].transLocTop
            self.color='orange'
        elif self.cellType=='PhP1':
            x=cellData[3].longLocTop+0.707*cellData[3].diameter/2
            y=cellData[3].radLocTop+0.707*cellData[3].diameter/2
            z=cellData[3].transLocTop
            self.color='deeppink'
        elif self.cellType=='PhP2':
            x=cellData[4].longLocTop+0.707*cellData[4].diameter/2
            y=cellData[4].radLocTop+0.707*cellData[4].diameter/2
            z=cellData[4].transLocTop
            self.color='deeppink'
        elif self.cellType=='PhP3':
            x=cellData[5].longLocTop+0.707*cellData[5].diameter/2
            y=cellData[5].radLocTop+0.707*cellData[5].diameter/2
            z=cellData[5].transLocTop
            self.color='deeppink'
        else:
            print('error')

        self.longLocBottom=x
        self.radLocBottom=y
        self.transLocBottom=z

        [x1,y1,z1] = calcEnd(self.length,self.longitudinalAngle,self.radialAngle)
        
        #calculate center points for the bottom and top of cylinder
        self.longLocTop = self.longLocBottom + x1
        self.radLocTop = self.radLocBottom + y1
        self.transLocTop = self.transLocBottom + z1

    def calcVibData(self,x1,y1,x2,y2,x_j1,y_j1,z_j1,xp_j,yp_j,zp_j):   #save vib mag and phase data for ends of each cell into cellData            
        self.topMagX=x_j1[x1,y1]
        self.topMagY=y_j1[x1,y1]
        self.topMagZ=z_j1[x1,y1]
        self.topPhX=xp_j[x1,y1]
        self.topPhY=yp_j[x1,y1]
        self.topPhZ=zp_j[x1,y1]
        self.bottomMagX=x_j1[x2,y2]
        self.bottomMagY=y_j1[x2,y2]
        self.bottomMagZ=z_j1[x2,y2]
        self.bottomPhX=xp_j[x2,y2]
        self.bottomPhY=yp_j[x2,y2]
        self.bottomPhZ=zp_j[x2,y2]
        
    def phpFix(self,x,y,z,nSteps,stepSize):

        #nSteps=0
        self.longLocTop=x+stepSize*nSteps
        self.radLocTop=y
        self.transLocTop=z 
#        self.diameter=8

        # self.longitudinalAngle1=np.copy(self.longitudinalAngle)
        # self.radialAngle1=np.copy(self.radialAngle)
        # self.length1=np.copy(self.length)
        self.longitudinalAngle=np.arctan((self.transLocTop-self.transLocBottom)/(self.longLocTop-self.longLocBottom))*360/(2*np.pi)
        self.radialAngle=np.arctan((self.transLocTop-self.transLocBottom)/(self.radLocTop-self.radLocBottom))*360/(2*np.pi)
        self.length=((self.longLocTop-self.longLocBottom)**2 + (self.radLocTop-self.radLocBottom)**2 + (self.transLocTop-self.transLocBottom)**2)**0.5
        if self.radialAngle<0:
            self.radialAngle=self.radialAngle+180
        return 

    def plotCylinder(self):       
        #axis and radius
        p0 = np.array([self.longLocBottom, self.radLocBottom, self.transLocBottom]) #point at one end
        p1 = np.array([self.longLocTop, self.radLocTop, self.transLocTop]) #point at other end
        R = self.diameter/2
        
        #vector in direction of axis
        v = p1 - p0
        
        #find magnitude of vector
        mag = norm(v)
        
        #unit vector in direction of axis
        v = v / mag
        
        #make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        
        #make vector perpendicular to v
        n1 = np.cross(v, not_v)
        #normalize n1
        n1 /= norm(n1)
        
        #make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)
        
        #surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 2)
        theta = np.linspace(0, 2 * np.pi, 100)
        rsample = np.linspace(0, R, 2)
        
        #use meshgrid to make 2d arrays
        t, theta2 = np.meshgrid(t, theta)
        
        rsample,theta = np.meshgrid(rsample, theta)
        
        #generate coordinates for surface
        # "Tube"
        self.X, self.Y, self.Z = [p0[i] + v[i] * t + R * np.sin(theta2) * n1[i] + R * np.cos(theta2) *       n2[i] for i in [0, 1, 2]]
        # "Bottom"
        self.X2, self.Y2, self.Z2 = [p0[i] + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        # "Top"
        self.X3, self.Y3, self.Z3 = [p0[i] + v[i]*mag + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        
def calcEnd(L,theta1,theta2):  
    # convert cell length and the 2 angles from Sunils paper into the x,y,z end coordinates of the cylinder
    # these equations were calculated using matlab symbolic toolbox (CellCoord.m)
    T1=theta1*2*np.pi/360
    T2=theta2*2*np.pi/360
    x=L*np.cos(T1)*np.sin(T2)*(1/(np.cos(T1)**2*np.sin(T2)**2 + np.cos(T2)**2*np.sin(T1)**2 + np.sin(T1)**2*np.sin(T2)**2))**(1/2)
    y=L*np.cos(T2)*np.sin(T1)*(1/(np.cos(T1)**2*np.sin(T2)**2 + np.cos(T2)**2*np.sin(T1)**2 + np.sin(T1)**2*np.sin(T2)**2))**(1/2)
    z=L*np.sin(T1)*np.sin(T2)*(1/(np.cos(T1)**2*np.sin(T2)**2 + np.cos(T2)**2*np.sin(T1)**2 + np.sin(T1)**2*np.sin(T2)**2))**(1/2)
    return x,y,z        

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def perp(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2,b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
 
        #Load the UI Page
        uic.loadUi('mainwindow.ui', self)

        #set baseline parameters
        self.cellType=['OP','IP','IHC','DC1','DC2','DC3','OHC1','OHC2','OHC3','PhP1','PhP2','PhP3'] 
        self.stepSize=10 #width of one OHC
        self.pixelDim=1.8        # this comes from Brian
        self.PixelDim_doubleSpinBox.setValue(self.pixelDim)


        #Setup the slots
        self.load_pushButton.clicked.connect(self.load_pushButton_pressed)        
        self.save_pushButton.clicked.connect(self.save_pushButton_pressed)        
        self.PixelDim_doubleSpinBox.valueChanged.connect(self.PixelDim_changed)
        self.verticalScrollBar.valueChanged.connect(self.verticalScrollBar_changed)
        self.horizontalScrollBar.valueChanged.connect(self.horizontalScrollBar_changed)
        self.thresholdScrollBar.valueChanged.connect(self.thresholdScrollBar_changed)
        
        #Setup the default values
        self.file_label.setText('MotionVector.mat')
        
        #Run using the default data
        self.load_pushButton_pressed()
        
        #self.plot([1,2,3,4,5,6,7,8,9,10], [30,32,34,32,33,31,29,32,35,45])

#    def plot(self, hour, temperature):
#       self.graphWidget.plot(hour, temperature)


    def load_pushButton_pressed(self):
        filename= 'puriaData1.xlsx'
    #    cellType=['OHC1','OHC2','OHC3','PhP1','PhP2','PhP3','DC1','DC2','DC3','IP','OP','IHC'] 
    #    cellType=['IP','OP','IHC','DC1','DC2','DC3','PhP1','PhP2','PhP3','OHC1','OHC2','OHC3',] 
    #    cellType=['OP','IP','IHC','DC1','DC2','DC3','OHC1','OHC2','OHC3','PhP1','PhP2','PhP3'] 
        # load cell length parameters
        self.p=params(filename)
        self.p.readParams()

        dataset=1
        # Load data from Brian: z is vertical (transverse), x is lateral (radial), y comes out of the page (longitudinal)
        if dataset==1:
            self.fileIn='MotionVector.mat'
            self.yCenter=188         # this is the BM under the outer pillar cell
            self.zCenter=187         # this is the BM under the outer pillar cell
        elif dataset==2:                 
            self.fileIn='MotionVector.mat'  # put whatever file you want here, and if you want set y an z center
            self.yCenter=188         
            self.zCenter=187         
        
        self.fileOut=self.fileIn[0:-4]+'_out.npz'
        self.fileOutXL=self.fileIn[0:-4]+'_out.xlsx'
        self.fileOutCSV=self.fileIn[0:-4]+'_out.csv'
        vibData=scipy.io.loadmat(self.fileIn)  
        self.label_fileIn.setText(self.fileIn)
        
        del vibData['__globals__']
        del vibData['__header__']
        del vibData['__version__']
        
        # switch x and y data to fit with my dimensions (z=transverse, y=radial, x=longitudinal) (Brian's data has x as radial and y as longitudinal)
        tempX=vibData['x_j']
        tempXP=vibData['xp_j']
        vibData['x_j']=vibData['y_j']
        vibData['xp_j']=vibData['yp_j']
        vibData['y_j']=tempX
        vibData['yp_j']=tempXP
        
        # flip array to make appropriate image
        for ii in list(vibData):
            vibData[ii]=np.rot90(np.flipud(vibData[ii]),-1)
            
        magStrings=['x_j','y_j','z_j']
        mVib=np.zeros(3)
        i=0
        for ii in magStrings:
            vibData[ii][np.isnan(vibData[ii])]=0            # remove NaNs from magnitude data 
            
            index=vibData[ii]<0                             # if magnitude is <0, make it positive and add 180 degrees to the phase
            vibData[ii][index]=abs(vibData[ii][index])
            iip=ii[:1]+'p'+ii[1:]
            vibData[iip][index]=vibData[iip][index]+np.pi   
            
            index=vibData[ii]<0.01                           # if mag<threshold, remove phase data
            vibData[iip][index]=np.nan   
            
            mVib[i]=np.max(np.max(vibData[ii]))             # find the max vibration
            i=i+1
        self.maxVib=np.max(mVib)
        
        phStrings=['xp_j','yp_j','zp_j']
        for ii in phStrings:                #get all phase data between -pi and pi
            vibData[ii][vibData[ii]>np.pi]=vibData[ii][vibData[ii]>np.pi]-2*np.pi
            vibData[ii][vibData[ii]<-1*np.pi]=vibData[ii][vibData[ii]<-1*np.pi]+2*np.pi


        # fig, axs = plt.subplots(4, 2, sharex='all', sharey='all')
        # axs[0,0].imshow(vibData['vol_j'])
        # axs[0,1].imshow(vibData['vol_j'])
        # axs[1,0].imshow(vibData['x_j'], vmin=0, vmax=self.maxVib, cmap='hot')
        # axs[1,1].imshow(vibData['xp_j'], vmin=-np.pi, vmax=np.pi, cmap='hsv')
        # axs[2,0].imshow(vibData['y_j'], vmin=0, vmax=self.maxVib, cmap='hot')
        # axs[2,1].imshow(vibData['yp_j'], vmin=-np.pi, vmax=np.pi, cmap='hsv')
        # axs[3,0].imshow(vibData['z_j'], vmin=0, vmax=self.maxVib, cmap='hot')
        # axs[3,1].imshow(vibData['zp_j'], vmin=-np.pi, vmax=np.pi, cmap='hsv')
        # plt.savefig('vibData.jpeg')
        
       # self.graphWidget.imshow(vibData['vol_j'])
#        self.graphWidget.image(vibData['vol_j'])
#        imv=pg.ImageView()
#        self.graphicsView(imv)
#        self.widget=pg.ImageView()
#       self.widget.show()
#        self.graphicsView.setPhoto(vibData['vol_j'])
        
        self.initialSetup=1
        self.vibData=vibData
        self.horizontalScrollBar.setMaximum(np.shape(self.vibData['vol_j'])[1])
        self.verticalScrollBar.setMaximum(np.shape(self.vibData['vol_j'])[0])
        self.horizontalScrollBar.setValue(self.yCenter)
        self.verticalScrollBar.setValue(self.zCenter)                    
        self.setImageAxes()
        self.thresholdScrollBar_changed()
        self.initialSetup=0  
        
#        self.graphWidget.setImage(np.transpose(vibData['vol_j']), autoLevels=True)
#        plt.figure()
#        plt.imshow(vibData['vol_j'])
#        plt.annotate('Center', xy=(self.yCenter, self.zCenter), xycoords='data',
#                 xytext=(20,20), textcoords='offset points',
#                 arrowprops=dict(arrowstyle="->"))         


    def save_pushButton_pressed(self):
        #now store the information needed to recreate each cell in Blender in a .npz file
        self.data=np.zeros((self.nCols,len(self.cellType),8))
        for n in range(self.nCols):
            step=n*self.stepSize
            for i in range(len(self.cellType)):
                c=self.cellData[i] 
                self.data[n,i,0]=c.diameter/2
                self.data[n,i,1]=c.length
                self.data[n,i,2]=c.longLocBottom+step
                self.data[n,i,3]=c.radLocBottom
                self.data[n,i,4]=c.transLocBottom
                self.data[n,i,5]=c.radialAngle
                self.data[n,i,6]=c.longitudinalAngle       
        np.savez(self.fileOut,nCols=self.nCols,cellType=self.cellType,stepSize=self.stepSize,data=self.data,
                 vibData=self.vibData, maxVib=self.maxVib, vol_j=self.vol_j,x_j=self.x_j,y_j=self.y_j,z_j=self.z_j,
                 xp_j=self.xp_j,yp_j=self.yp_j,zp_j=self.zp_j,zLoc=self.zLoc, yLoc=self.yLoc)

        #put anatomic and vibratory data for each cell into a pandas data table and save in Excel format for averaging
        vibratoryData=np.zeros([len(self.cellType),12])
        anatomicData=np.zeros([len(self.cellType),6])
        cellName=[]
        dataFile=[]
        contentLabels=['xTopLoc','yTopLoc','zTopLoc','xBottomLoc','yBottomLoc','zBottomLoc',
                 'xTopMag','yTopMag','zTopMag','xTopPh','yTopPh','zTopPh',
                 'xBottomMag','yBottomMag','zBottomMag','xBottomPh','yBottomPh','zBottomPh']
        for i in range(len(self.cellType)):
            dataFile.append(self.fileIn)
            c=self.cellData[i]        
            cellName.append(c.cellType)
            vibratoryData[i,:]=np.array([c.topMagX,c.topMagY,c.topMagZ,
                                    c.topPhX,c.topPhY,c.topPhZ,
                                    c.bottomMagX,c.bottomMagY,c.bottomMagZ,
                                    c.bottomPhX,c.bottomPhY,c.bottomPhZ])
            anatomicData[i,:]=np.array([c.longLocTop,c.radLocTop,c.transLocTop,
                                   c.longLocBottom,c.radLocBottom,c.transLocBottom])
        allData=np.concatenate((anatomicData,vibratoryData),axis=1)
        df = pd.DataFrame(allData, index=cellName, columns=contentLabels)
#        df.to_excel(self.fileOutXL, sheet_name=self.fileIn[0:-4])
        df.to_csv(self.fileOutCSV)

        self.label_fileOut.setText(self.fileOut)       
        print('Data has been saved')
        
    def thresholdScrollBar_changed(self):
        self.threshold=(self.thresholdScrollBar.value()/100)*np.max(self.vibData['vol_j'])+1             
        if self.initialSetup==1:
            self.plot_cells() 
        self.plotCroppedImage()

        
    def PixelDim_changed(self):
        self.setImageAxes()
        self.plot_cells()  
        self.plotCroppedImage()
   
        
    def verticalScrollBar_changed(self):       
        if self.initialSetup==0:
            self.setImageAxes()
            self.plotCroppedImage()


    def horizontalScrollBar_changed(self):
        if self.initialSetup==0:
            self.setImageAxes()
            self.plotCroppedImage()

        
    def setImageAxes(self):        
        self.pixelDim=self.PixelDim_doubleSpinBox.value()
        self.yCenter=self.horizontalScrollBar.value()
        self.zCenter=self.verticalScrollBar.value()
#        print('self.yCenter,self.zCenter',self.yCenter,self.zCenter)
        # create y and z pixel locations vectors to help overlap vibratory data with the model
        self.vibData['yLoc']=np.arange(0,np.shape(self.vibData['vol_j'])[1])
        self.vibData['zLoc']=np.arange(0,np.shape(self.vibData['vol_j'])[0])
        self.vibData['yLoc']=self.vibData['yLoc'] - self.yCenter
        self.vibData['zLoc']=-1*(self.vibData['zLoc'] - self.zCenter)
        self.vibData['yLoc']=self.vibData['yLoc']*self.pixelDim
        self.vibData['zLoc']=self.vibData['zLoc']*self.pixelDim   

        
    def plotCroppedImage(self):  
        vibData=self.vibData
        
        # Now plot the OCT image that corresponds to the model to make sure they align
        indexZ=np.where((vibData['zLoc']>=self.zLimit[0]) & (vibData['zLoc']<=self.zLimit[1]))[0]
        indexY=np.where((vibData['yLoc']>=self.yLimit[0]) & (vibData['yLoc']<=self.yLimit[1]))[0]
    #    print('indexZ,indexY',indexZ,indexY )
        vol_j=np.copy(vibData['vol_j'][indexZ[0]:indexZ[-1]+1,indexY[0]:indexY[-1]+1]).T
        x_j1=np.copy(vibData['x_j'][indexZ[0]:indexZ[-1]+1,indexY[0]:indexY[-1]+1]).T
        y_j1=np.copy(vibData['y_j'][indexZ[0]:indexZ[-1]+1,indexY[0]:indexY[-1]+1]).T
        z_j1=np.copy(vibData['z_j'][indexZ[0]:indexZ[-1]+1,indexY[0]:indexY[-1]+1]).T
        xp_j=np.copy(vibData['xp_j'][indexZ[0]:indexZ[-1]+1,indexY[0]:indexY[-1]+1]).T
        yp_j=np.copy(vibData['yp_j'][indexZ[0]:indexZ[-1]+1,indexY[0]:indexY[-1]+1]).T
        zp_j=np.copy(vibData['zp_j'][indexZ[0]:indexZ[-1]+1,indexY[0]:indexY[-1]+1]).T
        self.zLoc=vibData['zLoc'][indexZ]
        self.yLoc=vibData['yLoc'][indexY]
        zLoc=self.zLoc
        yLoc=self.yLoc
        
        # don't use points below the threshold of the anatomic image
        index=vol_j<self.threshold                           
        vol_j[index]=0   
        x_j1[index]=np.nan   
        y_j1[index]=np.nan   
        z_j1[index]=np.nan   
        xp_j[index]=np.nan   
        yp_j[index]=np.nan   
        zp_j[index]=np.nan   
        
        allV=np.swapaxes(np.array([x_j1.T,y_j1.T,z_j1.T]),0,2)
#        vibOverlay=np.swapaxes(np.array([self.imgOverlay.T,self.imgOverlay.T,self.imgOverlay.T]),0,2)*self.maxVib/255
        cropImage=np.maximum(self.imgOverlay,vol_j)
#        cropImageVib=np.maximum(vibOverlay,allV)
        cropImageVib=allV
        maxBrightness=12

        for i in range(12):
            c=self.cellData[i]
            #find x,y points in the vibratory data that correspond to the top and bottom of the cells
            x1=np.argmin(np.abs(c.radLocTop-yLoc))
            y1=np.argmin(np.abs(c.transLocTop-zLoc))                     
            x2=np.argmin(np.abs(c.radLocBottom-yLoc))
            y2=np.argmin(np.abs(c.transLocBottom-zLoc))    
            c.calcVibData(x1,y1,x2,y2,x_j1,y_j1,z_j1,xp_j,yp_j,zp_j)   #save vib mag and phase data for ends of each cell into cellData     
#            print(c.cellType,c.topMagX,c.topMagY,c.topMagZ)
            if i<9:   # plot white dots on 3 color vib mag image
                cropImageVib[x1,y1]=maxBrightness  
                cropImageVib[x2,y2]=maxBrightness
#        print()       
        zeros_j1=np.zeros(np.shape(x_j1))
        x_j=np.swapaxes(np.array([x_j1.T,zeros_j1.T,zeros_j1.T]),0,2)
        y_j=np.swapaxes(np.array([zeros_j1.T,y_j1.T,zeros_j1.T]),0,2)
        z_j=np.swapaxes(np.array([zeros_j1.T,zeros_j1.T,z_j1.T]),0,2)
        
        self.graphWidget.setImage(np.transpose(self.vibData['vol_j']), autoLevels=True)
        self.overlayImage.setImage(cropImage, autoLevels=True)
        self.xVib.setImage(x_j,levels=[0,maxBrightness])
        self.yVib.setImage(y_j,levels=[0,maxBrightness])
        self.zVib.setImage(z_j,levels=[0,maxBrightness])
        self.vibAll.setImage(cropImageVib, autoLevels=True)
        self.xVibP.setImage(xp_j)
        self.yVibP.setImage(yp_j)
        self.zVibP.setImage(zp_j)      
        
        # Set color map for phase plots
        self.xVibP.ui.histogram.gradient.loadPreset('cyclic')
        self.yVibP.ui.histogram.gradient.loadPreset('cyclic')
        self.zVibP.ui.histogram.gradient.loadPreset('cyclic')
       
        # remove controls
        self.graphWidget.ui.histogram.hide()
        self.graphWidget.ui.roiBtn.hide()
        self.graphWidget.ui.menuBtn.hide()

        self.overlayImage.ui.histogram.hide()
        self.overlayImage.ui.roiBtn.hide()
        self.overlayImage.ui.menuBtn.hide()
        
        self.vibAll.ui.histogram.hide()
        self.vibAll.ui.roiBtn.hide()
        self.vibAll.ui.menuBtn.hide()
        
        self.xVib.ui.histogram.hide()
        self.xVib.ui.roiBtn.hide()
        self.xVib.ui.menuBtn.hide()
        self.yVib.ui.histogram.hide()
        self.yVib.ui.roiBtn.hide()
        self.yVib.ui.menuBtn.hide()
        self.zVib.ui.histogram.hide()
        self.zVib.ui.roiBtn.hide()
        self.zVib.ui.menuBtn.hide()

        self.xVibP.ui.histogram.hide()
        self.xVibP.ui.roiBtn.hide()
        self.xVibP.ui.menuBtn.hide()
        self.yVibP.ui.histogram.hide()
        self.yVibP.ui.roiBtn.hide()
        self.yVibP.ui.menuBtn.hide()
        self.zVibP.ui.histogram.hide()
        self.zVibP.ui.roiBtn.hide()
        self.zVibP.ui.menuBtn.hide()
       

        self.vol_j=vol_j 
        self.x_j=x_j1
        self.xp_j=xp_j
        self.y_j=y_j1
        self.yp_j=yp_j
        self.z_j=z_j1
        self.zp_j=zp_j
        
#        plt.figure()
#        plt.imshow(self.vol_j) 
#        plt.show() 
#        plt.imsave('background.jpeg', self.vol_j)       
    
    def plot_cells(self):
        # plot 3D image of organ of Corti
        pos=75  # % length from the base
        self.nCols=10 # number of OHC columns to put in the figure
        nCols=self.nCols
        self.cellData=[]
        for i in self.cellType:
            c=cell(i)
            c.anatomy(pos,self.p) 
            self.cellData.append(c)  
        
        # plot the cells
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        for n in range(nCols):
            q=0
            step=n*self.stepSize
            for i in range(9):
                c=self.cellData[q]
                c.angles(self.cellData)
                c.plotCylinder()
                ax.plot_surface(c.X + step, c.Y, c.Z, color= c.color, alpha=1)
                ax.plot_surface(c.X2 + step, c.Y2, c.Z2, color= c.color, alpha=1)
                ax.plot_surface(c.X3 + step, c.Y3, c.Z3, color= c.color, alpha=1)
                q+=1
    #    ax.view_init(70,70)
    #    plt.show()            
     
        # now figure out the tops of the phalangeal processes of the Deiter's cells, and then plot the processes
        x1=self.cellData[6].longLocTop
        x2=self.cellData[7].longLocTop
        x3=self.cellData[8].longLocTop
        y1=self.cellData[6].radLocTop
        y2=self.cellData[7].radLocTop
        y3=self.cellData[8].radLocTop
        x12=x1+self.stepSize/2
        y12=(y1+y2)/2
        x23=x2+self.stepSize/2
        y23=(y2+y3)/2
        x34=x3+self.stepSize/2
        y34=y23+y23-y12
        
        for n in range(nCols):
            q=9
            step=n*self.stepSize
            for i in [9,10,11]:
                c=self.cellData[q]
                c.angles(self.cellData)
                if i==9:
                    nSteps=3 # first row skips 3 OHCs
                    x=x12 
                    y=y12
                    z=(self.cellData[i-3].transLocTop + self.cellData[i-2].transLocTop)/2
                elif i==10:
                    nSteps=3 # second row skips 3 OHCs
                    x=x23
                    y=y23
                    z=(self.cellData[i-3].transLocTop + self.cellData[i-2].transLocTop)/2
                elif i==11:
                    nSteps=1 # third row skips only 1 OHC
                    x=x34
                    y=y34
                    z=self.cellData[i-3].transLocTop + (self.cellData[i-3].transLocTop - self.cellData[i-4].transLocTop)/2
                    
                c.phpFix(x,y,z,nSteps,self.stepSize)
                c.plotCylinder()
                ax.plot_surface(c.X + step, c.Y, c.Z, color= c.color)
                ax.plot_surface(c.X2 + step, c.Y2, c.Z2, color= c.color)
                ax.plot_surface(c.X3 + step, c.Y3, c.Z3, color= c.color)
                q+=1
        ax.view_init(0,0)
        ax.axes.set_xlim3d(left=0, right=100) 
        ax.axes.set_ylim3d(bottom=-60, top=40) 
        ax.axes.set_zlim3d(bottom=0, top=100)
        plt.show()            
        self.zLimit=ax.get_zlim()
        self.yLimit=ax.get_ylim()
        self.indexZ=np.where((self.vibData['zLoc']>=self.zLimit[0]) & (self.vibData['zLoc']<=self.zLimit[1]))[0]
        self.indexY=np.where((self.vibData['yLoc']>=self.yLimit[0]) & (self.vibData['yLoc']<=self.yLimit[1]))[0]

        # now, create a simple 2D image of the key cells to overlay on the cropped image 
        zNum=np.size(self.indexZ)
        yNum=np.size(self.indexY)
        zLoc=self.vibData['zLoc'][self.indexZ]
        yLoc=self.vibData['yLoc'][self.indexY]
        
        imgList=[]
        self.imgOverlay=np.zeros((np.size(zLoc),np.size(yLoc)))
        for i in range(11):
            c=self.cellData[i]
            c.angles(self.cellData)
            imgList.append(draw_line(self.pixelDim, zLoc, yLoc, c.transLocTop, c.radLocTop, c.transLocBottom, c.radLocBottom))
            self.imgOverlay=np.logical_or(imgList[i],self.imgOverlay)
        self.imgOverlay=np.fliplr(np.transpose(self.imgOverlay*255))
        self.getVib(zLoc, yLoc)
#            ax.plot_surface(c.X + step, c.Y, c.Z, color= c.color, alpha=1)
#            ax.plot_surface(c.X2 + step, c.Y2, c.Z2, color= c.color, alpha=1)
#            ax.plot_surface(c.X3 + step, c.Y3, c.Z3, color= c.color, alpha=1)
           
    def getVib(self,zLoc,yLoc):  
        for i in range(9):
            c=self.cellData[i]
            c.transLocTop_crop=int((c.transLocTop-zLoc[-1])/self.pixelDim)
            c.transLocBottom_crop=int((c.transLocBottom-zLoc[-1])/self.pixelDim)
            c.radLocTop_crop=int((c.radLocTop-yLoc[0])/self.pixelDim)
            c.radLocBottom_crop=int((c.radLocBottom-yLoc[0])/self.pixelDim)

def draw_line(pixelDim, zLoc, yLoc, x0, y0, x1, y1, inplace=False):
#    print(y0,x0,y1,x1)  
#    print(zLoc[-1],yLoc[0])
    x0=int((x0-zLoc[-1])/pixelDim)
    x1=int((x1-zLoc[-1])/pixelDim)
    y0=int((y0-yLoc[0])/pixelDim)
    y1=int((y1-yLoc[0])/pixelDim)
#    print(y0,x0,y1,x1)  
    
    mat=np.zeros((np.size(zLoc),np.size(yLoc)))
                 
    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
        raise ValueError('Invalid coordinates.')
    if not inplace:
        mat = mat.copy()
    if (x0, y0) == (x1, y1):
        mat[x0, y0] = 1
        return mat if not inplace else None
    # Swap axes if Y slope is smaller than X slope
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1
    # Swap line direction to go left-to-right if necessary
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # Write line ends
    mat[x0, y0] = 1
    mat[x1, y1] = 1
    # Compute intermediate coordinates using line equation
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
    # Write intermediate coordinates
    mat[x, y] = 1
    if not inplace:
        return mat if not transpose else mat.T                  

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()