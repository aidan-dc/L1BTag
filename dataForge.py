import time

start = time.time()

import ROOT, sys
from ROOT import *
import numpy
import h5py

#Customizable inputs, pT cut is applied to individual jets, data/file tag helps label output dataset files
if len(sys.argv)!=6 or (int(sys.argv[4])<0 or int(sys.argv[4])>100) or (int(sys.argv[5])!=0 and int(sys.argv[5])!=1):
	print("USAGE: <input ROOT file> <data/file tag> <pT cut> <percent ratio of training/testing data (0-100)> <candidates (0 for PF, 1 for PUPPI)>")
	sys.exit(1)

ROOT.gROOT.SetBatch(1)

inFileName = sys.argv[1]
print("Reading from "+str(inFileName))

inFile = ROOT.TFile.Open(inFileName,"READ")

tree = inFile.Get("ntuple0/objects")
ver = inFile.Get("ntuple0/objects/vz")

#Load variables based on inputs and initialize lists to be used later
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

#One-Hot Encoding for Particle Type
def scalePartType(a, n):
	if n==11:
		a.extend((1,0,0,0,0,0,0,0)) #Electron
	elif n==-11:
		a.extend((0,1,0,0,0,0,0,0)) #Positron
	elif n==13:
		a.extend((0,0,1,0,0,0,0,0)) #Muon
	elif n==-13:
		a.extend((0,0,0,1,0,0,0,0)) #Anti-Muon
	elif n==22:
		a.extend((0,0,0,0,1,0,0,0)) #Photon
	elif n==130:
		a.extend((0,0,0,0,0,1,0,0)) #Neutral Meson
	elif n==211:
		a.extend((0,0,0,0,0,0,1,0)) #Pion
	elif n==-211:
		a.extend((0,0,0,0,0,0,0,1)) #Anti-Pion
	else:
		a.extend((0,0,0,0,0,0,0,0)) #Case for unknown particle

#Scaling Phi for particles relative to their jet
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
for entryNum in range(eventNum):
	if entryNum%(int(eventNum/10))==0:
		print('Progress: '+str(entryNum)+" out of " + str(eventNum)+", approximately "+str(int(100*entryNum/eventNum))+"%"+" complete.")
		print('Current No. of Jets: '+str(len(jetPartsArray)))
		print('Current No. of Signal Jets: '+str(bQuarkCount))
	tree.GetEntry(entryNum)
	ver = tree.vz
	#Loading particle candidates based on PF or PUPPI input
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
	bannedParts = [] #List of indices of particles that have already been used by previous jets
	bannedbQuarks = [] #Same deal but with indices within the gen tree corresponding to b quarks

	#Loops through pf/pup candidates
	for i in range(len(obj)):
		jetPartList = []
		seedParticle = []
		if jetNum>=12: #Limited to 12 jets per event at maximum
			jetNum = 0
			break
		if i not in bannedParts: #Identifies highest avaiable pT particle to use as seed
			tempTLV = obj[i][0] #Takes TLorentzVector of seed particle to use for jet reconstruction
			scalePartType(seedParticle,abs(obj[i][1])) #One-Hot Encoding Seed Particle Type
			if obj[i][1]==22 or obj[i][1]==130:
				seedParticle.extend([0.0,verPfX[i],verPfY[i],obj[i][0].Pt(),obj[i][0].Eta(),obj[i][0].Phi()]) #Add in dZ, dX, dY, Particle Pt, Eta, & Phi, last 3 features to be scaled later
			else:
				seedParticle.extend([ver[0]-verPf[i],verPfX[i],verPfY[i],obj[i][0].Pt(),obj[i][0].Eta(),obj[i][0].Phi()]) #Add in dZ, dX, dY, Particle Pt, Eta, & Phi, last 3 features to be scaled later
			jetPartList.extend(seedParticle) #Add particle features to particle list
			bannedParts.append(i) #Mark this particle as unavailable for other jets
			for j in range(len(obj)):
				partFts = []
				if obj[i][0].DeltaR(obj[j][0])<=0.4 and i!=j and (j not in bannedParts): #Look for available particles within deltaR<0.4 of seed particle
					tempTLV=tempTLV+obj[j][0] #Add to tempTLV
					scalePartType(partFts,obj[j][1]) #One-Hot Encoding Particle Type
					if obj[j][1]==22 or obj[j][1]==130:
						partFts.extend([0.0,verPfX[j],verPfY[j],obj[j][0].Pt(),obj[j][0].Eta(),obj[j][0].Phi()]) #Add in dZ, dX, dY, Particle Pt, Eta, & Phi, last 3 features to be scaled later
					else:
						partFts.extend([ver[0]-verPf[j],verPfX[j],verPfY[j],obj[j][0].Pt(),obj[j][0].Eta(),obj[j][0].Phi()])
					jetPartList.extend(partFts)  #Add particle features to particle list
					bannedParts.append(j) #Mark this particle as unavailable for other jets
				if len(jetPartList)>=10*14: #If you reach 10 particles in one jet, break and move on
					break
			if abs(tempTLV.Pt())<pTCut: #Neglect to save jet if it falls below pT Cut
				break
			#Scaling particle pT, Eta, and Phi based on jet pT, Eta, and Phi
			c = 11
			while c<len(jetPartList)-2:
				jetPartList[c]=jetPartList[c]/tempTLV.Pt()
				jetPartList[c+1]=tempTLV.Eta()-jetPartList[c+1]
				tempPhi = jetPartList[c+2]
				jetPartList[c+2] = signedDeltaPhi(tempTLV.Phi(),tempPhi)
				c+=14
			#Ensure all inputs are same length			
			while len(jetPartList)<10*14:
				jetPartList.append(0)
			#Add in final value to indicate if particle is matched (1) or unmatched (0) to a gen b quark by looking for gen b quarks within deltaR<0.4 of jet
			jetPartList.append(0)
			for e in range(len(tree.gen)):
				if abs(tree.gen[e][1])==5 and (e not in bannedbQuarks) and abs(tree.gen[e][0].Eta())<2.3:
					if tree.gen[e][0].DeltaR(tempTLV)<=0.4:
						jetPartList[-1]=1
						bQuarkCount+=1
						bannedbQuarks.append(e)
						break
			#Store particle inputs and jet features in overall list
			jetPartsArray.append(jetPartList)
			jetDataArray.append((tempTLV.Pt(),tempTLV.Eta(),tempTLV.Phi(),tempTLV.M()))
			jetNum+=1

#Break dataset into training/testing data based on train/test split input
trainTestSplit = int(sys.argv[4])
splitIndex = int(float(trainTestSplit)/100*len(jetPartsArray))
trainArray = jetPartsArray[:splitIndex]
trainingFullData = jetDataArray[:splitIndex]

testArray = jetPartsArray[splitIndex:]
jetFullData = jetDataArray[splitIndex:]

print('Total Jets '+str(len(jetPartsArray)))
print('Total No. of Matched Jets: '+str(bQuarkCount))
print('No. of Jets in Training Data: '+str(len(trainArray)))
print('No. of Jets in Testing Data: '+str(len(testArray)))

#Final Check
print('Debug that everything matches up in length:')
print(len(testArray)==len(jetFullData) and len(trainArray)==len(trainingFullData))

#Save datasets as h5 files

#Testing Data: Particle Inputs for each jet of Shape [...,141]
with h5py.File('testingData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Testing Data", data=testArray)
#Jet Data: Jet Features (pT, Eta, Phi, Mass) of each testing data jet of shape [...,4]
with h5py.File('jetData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Jet Data", data=jetFullData)
#Training Data: Particle Inputs for each jet of Shape [...,141]
with h5py.File('trainingData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Training Data", data=trainArray)
#Sample Data: Jet Features (pT, Eta, Phi, Mass) of each training data jet of shape [...,4]
with h5py.File('sampleData'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Sample Data", data=trainingFullData)

end = time.time()
print(str(end-start))
