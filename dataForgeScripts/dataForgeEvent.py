import time

start = time.time()

import ROOT, sys
from ROOT import *
import numpy
import h5py

print(len(sys.argv))

if len(sys.argv)!=5 or int(sys.argv[3])<0 or (int(sys.argv[4])!=0 and int(sys.argv[4])!=1):
	print("USAGE: <input ROOT file> <data/file tag> <pT cut> <candidates (0 for PF, 1 for PUPPI)>")
	sys.exit(1)

ROOT.gROOT.SetBatch(1)

inFileName = sys.argv[1]
print("Reading from "+str(inFileName))

inFile = ROOT.TFile.Open(inFileName,"READ")

tree = inFile.Get("ntuple0/objects")
ver = inFile.Get("ntuple0/objects/vz")

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

def signedDeltaPhi(phi1, phi2):
    dPhi = phi1 - phi2
    if (dPhi < -numpy.pi):
        dPhi = 2 * numpy.pi + dPhi
    elif (dPhi > numpy.pi):
        dPhi = -2 * numpy.pi + dPhi
    return dPhi

eventPartList = []
eventDataList = []

jetPartsArray = []
jetDataArray = []
jetCount=0

finalPartList = []
finalDataList = []

print('Beginning Jet Construction')
for entryNum in range(eventNum):
	if entryNum%(int(eventNum/10))==0:
		print('Progress: '+str(entryNum)+" out of " + str(eventNum)+", approximately "+str(int(100*entryNum/eventNum))+"%"+" complete.")
		print('Current No. of Jets: '+str(jetCount))
		print('Current No. of Signal Jets: '+str(bQuarkCount))
	tree.GetEntry(entryNum)
	eventPartList = []
	eventDataList = []
	ver = tree.vz
	if int(sys.argv[4])==0:
		obj = tree.pf
		verPf = tree.pf_vz
		verPfX = tree.pf_vx
		verPfY = tree.pf_vy
	if int(sys.argv[4])==1:
		obj = tree.pup
		verPf = tree.pup_vz
		verPfX = tree.pup_vx
		verPfY = tree.pup_vy
	jetNum = 0
	bannedParts = [] #List of indices of particles that have already been used by previous jets
	bannedbQuarks = [] #Same deal but with indices within the gen tree corresponding to b quarks
	for i in range(len(obj)):
		jetPartList = []
		seedParticle = []
		if jetNum>=12:
			jetNum = 0
			break
		if i not in bannedParts:#obj[i][1] in chargedHadrons and (i not in bannedParts):
			tempTLV = obj[i][0]
			scalePartType(seedParticle,abs(obj[i][1]))
			seedParticle.extend([verPf[i]-ver[0],verPfX[i],verPfY[i],obj[i][0].Pt(),obj[i][0].Eta(),obj[i][0].Phi()])
			jetPartList.extend(seedParticle)
			bannedParts.append(i)
			for j in range(len(obj)):
				partFts = []
				if obj[i][0].DeltaR(obj[j][0])<=0.4 and i!=j and (j not in bannedParts):
					tempTLV=tempTLV+obj[j][0]
					scalePartType(partFts,obj[j][1])
					partFts.extend([verPf[j]-ver[0],verPfX[j],verPfY[j],obj[j][0].Pt(),obj[j][0].Eta(),obj[j][0].Phi()])
					jetPartList.extend(partFts)
					bannedParts.append(j)
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
				if abs(tree.gen[e][1])==5 and (e not in bannedbQuarks) and abs(tree.gen[e][0].Eta())<2.3:
					if tree.gen[e][0].DeltaR(tempTLV)<=0.4:
						jetPartList[-1]=1
						bQuarkCount+=1
						bannedbQuarks.append(e)
						break
			eventPartList.append(jetPartList)
			eventDataList.append((tempTLV.Pt(),tempTLV.Eta(),tempTLV.Phi(),tempTLV.M()))
			jetNum+=1
			jetCount+=1
	while len(eventPartList)<12:
		eventPartList.append(141*[0])
		eventDataList.append(4*[0])
	finalPartList.append(eventPartList)
	finalDataList.append(eventDataList)

print(numpy.shape(finalPartList))
print(numpy.shape(finalDataList))

print('Total Jets '+str(jetCount))
print('Total No. of Signal Jets: '+str(bQuarkCount))

print('Debug that everything matches up in length:')
print(len(eventPartList)==len(eventDataList))

with h5py.File('testingDataEvents'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Testing Data", data=finalPartList)
with h5py.File('jetDataEvents'+str(sys.argv[2])+'.h5','w') as hf:
	hf.create_dataset("Jet Data", data=finalDataList)

end = time.time()
print(str(end-start))
