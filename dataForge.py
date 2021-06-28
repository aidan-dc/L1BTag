import json, math
import ROOT, sys, os, re, string
from ROOT import *
import numpy
import h5py
from numpy import asarray
from numpy import savetxt
import math

#Loads in the input ROOT File from which to load the TTree, the filename for the datasets, pT cuts on jets, training/testing split, and particle candidate choice
if len(sys.argv)!=6 or (int(sys.argv[4])<0 or int(sys.argv[4])>100) or (int(sys.argv[5])!=0 and int(sys.argv[5])!=1):
	print("USAGE: <input file> <date/file tag> <pT cut> <percent ratio of training/testing data (0-100)> <candidates (0 for PF, 1 for PUPPI)>")
	sys.exit(1)

#Loads the TTree from the ROOT File and selects the appropriate objects (either PF or PUPPI candidates)
inFileName = sys.argv[1]
print("Reading from "+str(inFileName))
inFile = ROOT.TFile.Open(inFileName,"READ")
tree = inFile.Get("ntuple0/objects")
ver = inFile.Get("ntuple0/objects/vz")
if sys.argv[5]==0:
	obj = inFile.Get("ntuple0/objects/pf")
	verPf = inFile.Get("ntuple0/objects/pf_vz")
	verPfX = inFile.Get("ntuple0/objects/pf_vx")
	verPfY = inFile.Get("ntuple0/objects/pf_vy")
if sys.argv[5]==1:
	obj = inFile.Get("ntuple0/objects/pup")
	verPup = inFile.Get("ntuple0/objects/pup_vz")
	verPupX = inFile.Get("ntuple0/objects/pup_vx")
	verPupY = inFile.Get("ntuple0/objects/pup_vy")

#Initializing all the lists needed throughout the jet reconstruction
eventNum = tree.GetEntries()
pTCut = float(sys.argv[3])
jetList = []
jetNum = 0
eventJets = []
bQuarkCount = 0
jetPartList = []
trainArray = []
testArray = []
jetFullData = []
trainingFullData = []
totalPartCount = 10
chargedHadrons = [211,-211]
bQTrainCount = 0
hasBQC = 0
unknowncount = 0
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

jetPartsArray = []
jetDataArray = []
print('Beginning Jet Construction')
for entryNum in range(10):
	if entryNum%(int(eventNum/100))==0:
		print('Progress: '+str(entryNum)+" out of " + str(eventNum)+", approximately "+str(int(100*entryNum/eventNum))+"%"+" complete.")
		print('Current No. of Jets: '+str(len(jetPartsArray)))
		print('Current No. of Signal Jets: '+str(bQuarkCount))
	tree.GetEntry(entryNum)
	ver = tree.vz
	if int(sys.argv[5])==0:
		obj = tree.pf
		verPf = tree.pf_vz
		verPfX = tree.pf_vx
		verPfY = tree.pf_vy
	if int(sys.argv[5])==1:
		obj = tree.pup
		verPf = tree.pup_vz
		verPfX = tree.pup_vx
		verPfY = tree.pup_vy
	jetNum = 0
	for i in range(len(obj)):
		jetPartList = []
		seedParticle = []
		if jetNum>=5:
			jetNum = 0
			break
		if obj[i][1] in chargedHadrons:
			tempTLV = obj[i][0]
			scalePartType(seedParticle,abs(obj[i][1]))
			seedParticle.extend([verPf[i]-ver[0],verPfX[i],verPfY[i],obj[i][0].Pt(),obj[i][0].Eta(),obj[i][0].Phi()])
			jetPartList.extend(seedParticle)
			for j in range(len(obj)):
				partFts = []
				if obj[i][0].DeltaR(obj[j][0])<=0.4 and i!=j:
					tempTLV=tempTLV+obj[j][0]
					scalePartType(partFts,obj[j][1])
					partFts.extend([verPf[j]-ver[0],verPfX[j],verPfY[j],obj[j][0].Pt(),obj[j][0].Eta(),obj[j][0].Phi()])
					jetPartList.extend(partFts)
				if len(jetPartList)>=10*14:
					break
			if abs(tempTLV.Pt())<pTCut:
				break
			c = 11
			while c<len(jetPartList)-2:
				jetPartList[c]=jetPartList[c]/tempTLV.Pt()
				jetPartList[c+1]=jetPartList[c+1]-tempTLV.Eta()
				tempPhi = jetPartList[c+2]
				jetPartList[c+2] = signedDeltaPhi(tempPhi,tempTLV.Phi())
				c+=14			
			while len(jetPartList)<10*14:
				jetPartList.append(0)
			jetPartList.append(0)
			for e in range(len(tree.gen)):
				if abs(tree.gen[e][1])==5:
					if tree.gen[e][0].DeltaR(tempTLV)<=0.4:
						jetPartList[-1]=1
						bQuarkCount+=1
						break
			jetPartsArray.append(jetPartList)
			jetDataArray.append((tempTLV.Pt(),tempTLV.Eta(),tempTLV.Phi(),tempTLV.M()))
			jetNum+=1

trainTestSplit = int(sys.argv[4])
splitIndex = int(float(trainTestSplit)/100*len(jetPartsArray))
trainArray = jetPartsArray[:splitIndex]
trainingFullData = jetDataArray[:splitIndex]

testArray = jetPartsArray[splitIndex:]
jetFullData = jetDataArray[splitIndex:]

print('Total Jets '+str(len(jetPartsArray)))
print('Total No. of Signal Jets: '+str(bQuarkCount))
print('No. of Jets in Training Data: '+str(len(trainArray)))
print('No. of Jets in Testing Data: '+str(len(testArray)))

print('Debug that everything matches up in length:')
print(len(testArray)==len(jetFullData) and len(trainArray)==len(trainingFullData))

exit()

with h5py.File('testingData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Testing Data", data=testArray)
with h5py.File('jetData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Jet Data", data=jetFullData)
with h5py.File('trainingData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Training Data", data=trainArray)
with h5py.File('sampleData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Sample Data", data=trainingFullData)
