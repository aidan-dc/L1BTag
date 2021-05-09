import json, math
import ROOT, sys, os, re, string
from ROOT import *
import numpy
import h5py
from numpy import asarray
from numpy import savetxt
import math

#Loads in the ROOT File specified as the first argument after the script, with the data file tag as the second argument
if len(sys.argv)!=3:
	print("USAGE: <input file> <date/file tag>")
	sys.exit(1)

#Loads the TTree from the ROOT File and selects the appropriate objects (either PF or PUPPI candidates)
ROOT.gROOT.SetBatch(1)
inFileName = sys.argv[1]
print("Reading from "+str(inFileName))
inFile = ROOT.TFile.Open(inFileName,"READ")
tree = inFile.Get("ntuple0/objects")
obj = inFile.Get("ntuple0/objects/pf")
#obj = inFile.Get("ntuple0/objects/pup")
ver = inFile.Get("ntuple0/objects/vz")
verPf = inFile.Get("ntuple0/objects/pf_vz")
verPfX = inFile.Get("ntuple0/objects/pf_vx")
verPfY = inFile.Get("ntuple0/objects/pf_vy")
#verPup = inFile.Get("ntuple0/objects/pup_vz")
#verPupX = inFile.Get("ntuple0/objects/pup_vx")
#verPupY = inFile.Get("ntuple0/objects/pup_vy")

eventNum = tree.GetEntries()

#Initializing all the lists needed throughout the jet reconstruction
jetList = []
jetNum = 0
eventJets = []
bQuarkCount = 0
jetPartList = []
trainArray = []
testArray = []
jetFullData = []
trainingFullData = []
'''
Typically unused lists but previously helpful for separating jets into distinct pT bins and counting the number of signal events in each
a040 = []
c040 = 0
a4080 = []
c4080 = 0
a80120 = []
c80120 = 0
a120160 = []
c120160 = 0
a160200 = []
c160200 = 0
'''
totalPartCount = 10
chargedHadrons = [211,-211]
eventCap = 200000 #Specify the number of events you wish to pass to the training data before adding to the testing data

partTypes = [11,13,22,130,211]

#Function used for one hot encoding
def scalePartType(a, n):
	if n==11:
		a.extend((1,0,0,0,0,0,0,0))
	elif n==-11:
		a.extend((0,1,0,0,0,0,0,0))
	elif n==13:
		a.extend((0,0,1,0,0,0,0,0))
	elif n==-13:
		a.extend((0,0,0,1,0,0,0,0))
	elif n==22:
		a.extend((0,0,0,0,1,0,0,0))
	elif n==130:
		a.extend((0,0,0,0,0,1,0,0))
	elif n==211:
		a.extend((0,0,0,0,0,0,1,0))
	elif n==-211:
		a.extend((0,0,0,0,0,0,0,1))
	else:
		a.extend((0,0,0,0,0,0,0,0))

#Function used to normalize phi WRT jet
def signedDeltaPhi(phi1, phi2):
    dPhi = phi1 - phi2
    if (dPhi < -numpy.pi):
        dPhi = 2 * numpy.pi + dPhi
    elif (dPhi > numpy.pi):
        dPhi = -2 * numpy.pi + dPhi
    return dPhi

print('Beginning Jet Construction')
for entryNum in range(eventNum):
	if entryNum%150==0:
		print('Progress: '+str(entryNum)+" out of " + str(eventNum)+", approximately "+str(int(100*entryNum/eventNum))+"%"+" complete.")
		print('Current Length of Training Array: '+str(len(trainArray)))
		print('Current Length of Testing Array: '+str(len(testArray)))
    
  #Get the tree for a specific instance, can be changed manually to specify PUPPI candidates
	tree.GetEntry(entryNum)
	obj = tree.pf
	#obj = tree.pup
	ver = tree.vz
	verPf = tree.pf_vz
	verPfX = tree.pf_vx
	verPfY = tree.pf_vy
	jetNum = 0
	for i in range(len(obj)):
		jetPartList = []
    #Only take 5 jets per event to not have an unnecessary excess of data
		if jetNum>=5:
			jetNum = 0
			break
    #Identify seed particle and add it to the list of data for the jet
		if obj[i][1] in chargedHadrons:
			tempTLV = obj[i][0]
			scalePartType(jetPartList,abs(obj[i][1]))
      jetPartList.extend([verPf[i]-ver[0],verPfX[i],verPfY[i],obj[i][0].Pt(),obj[i][0].Eta(),obj[i][0].Phi()])
      #Add in 9 more particles within a DeltaR<=0.4 of the seed particle
			for j in range(len(obj)):
				if obj[i][0].DeltaR(obj[j][0])<=0.4 and i!=j:
					tempTLV=tempTLV+obj[j][0]
					scalePartType(jetPartList,obj[j][1])
          jetPartList.extend([verPf[j]-ver[0],verPfX[j],verPfY[j],obj[j][0].Pt(),obj[j][0].Eta(),obj[j][0].Phi()])
				if len(jetPartList)>=totalPartCount*14:
					break
      #Scale pT, Eta, and Phi for each particle WRT jet
			c=11
			while c<len(jetPartList)-2:
				jetPartList[c]=jetPartList[c]/tempTLV.Pt()
				jetPartList[c+1]=jetPartList[c+1]-tempTLV.Eta()
				tempPhi = jetPartList[c+2]
				jetPartList[c+2] = signedDeltaPhi(tempPhi,tempTLV.Phi())
				c+=14
      #Insert zeroes for all jets with less than 10 particles
			while len(jetPartList)<totalPartCount*14:
				jetPartList.append(0)
      #Insert a boolean for if the jet contains a b quark and test if this jet is matched to a b quark in gen
			jetPartList.append(0)
			for e in range(len(tree.gen)):
				if abs(tree.gen[e][1])==5:
					if tree.gen[e][0].DeltaR(tempTLV)<=0.4:
						jetPartList[-1]=1
						bQuarkCount+=1
						break
      #Remove all jets below 30 GeV
			if abs(tempTLV.Pt())<30:
				break
      #Store the jet in training or testing data
			if len(trainArray)>=eventCap:
				testArray.append(jetPartList)
				jetFullData.append((tempTLV.Pt(),tempTLV.Eta(),tempTLV.Phi(),tempTLV.M()))
			if len(trainArray)<eventCap:
				trainArray.append(jetPartList)
				trainingFullData.append((tempTLV.Pt(),tempTLV.Eta(),tempTLV.Phi(),tempTLV.M()))
			jetNum+=1

#The dataset used for testing performance, eff curves, ROC curves, etc.
with h5py.File('testingData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Testing Data", data=testArray)
#The unscaled pT, Eta, Phi, and Mass of the total jet in each entry for the testing data, 1-to-1 correspondence; helpful for eff curves
with h5py.File('jetData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Jet Data", data=jetFullData)
#The dataset used for training networks
with h5py.File('trainingData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Training Data", data=trainArray)
#The unscaled pT, Eta, Phi, and Mass of the total jet for each entry for the training data, 1-to-1 correspondence; helpful for constructing sample weights for training
with h5py.File('sampleData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Sample Data", data=trainingFullData)
