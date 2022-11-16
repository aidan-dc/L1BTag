import ROOT, sys, os, re, string
from ROOT import *
import numpy
import h5py

ROOT.gROOT.SetBatch(1)

#Load in ROOT File
inFileName = sys.argv[1]
print("Reading from "+str(inFileName))

inFile = ROOT.TFile.Open(inFileName,"READ")

tree = inFile.Get("ntuple0/objects")
eventNum = tree.GetEntries()

#Establish output arrays
bScores = [] #b-tagging NN discriminant [0,1]
jetDataArray = [] #Store [pT,Eta,Phi,Mass,di-Higgs Mass] of jet/event


print('Beginning Event Reconstruction')
for entryNum in range(eventNum):
	if entryNum%(int(eventNum/10))==0:
		print('Progress: '+str(entryNum)+" out of " + str(eventNum)+", approximately "+str(int(100*entryNum/eventNum))+"%"+" complete.")
	tree.GetEntry(entryNum)
	
  #Apply selection filter for HH4b events
	bQuarks = []
	pTs = []
	for e in range(len(tree.gen)):
		if abs(tree.gen[e][1])==5:
			bQuarks.append(tree.gen[e][0])
			pTs.append(tree.gen[e][0].Pt())
	pTs = sorted(pTs,reverse=True)
	if len(bQuarks)!=4 or pTs[3]<30:
		continue
  #Obtain di-Higgs mass for entire event, stored as last entry in all jetData for this event
	mHH = (bQuarks[0]+bQuarks[1]+bQuarks[2]+bQuarks[3]).M()
	
  l1jets = tree.l1jet
	
  eventbS = []
	eventJet = []
	for j in range(min(len(l1jets),12)):
    #Jets w/ pT<20 or |Eta|>2.3 are given negative b scores, set these to zero otherwise store normal value
		if l1jets[j][1]>0: 
			eventbS.append(l1jets[j][1])
		else:
			eventbS.append(0)
    #Store jet pT, Eta, Phi, Mass as well as event di-Higgs Mass
		eventJet.append([l1jets[j][0].Pt(),l1jets[j][0].Eta(),l1jets[j][0].Phi(),l1jets[j][0].M(),mHH])
	while len(eventbS)<12:
		eventbS.append(0)
		eventJet.append([0.0,0.0,0.0,0.0,mHH])
	bScores.append(eventbS)
	jetDataArray.append(eventJet)

#Check manually that shapes of output arrays are equal
print(numpy.shape(bScores))
print(numpy.shape(jetDataArray))

with h5py.File('predProbsEventsHH_PUP_Off_CutXY_dXY_CMSSW.h5','w') as hf:
	hf.create_dataset("b Scores", data=numpy.array(bScores))
with h5py.File('jetDataEventsHH_PUP_Off_CutXY_dXY.h5','w') as hf:
	hf.create_dataset("Jet Data", data=numpy.array(jetDataArray))
