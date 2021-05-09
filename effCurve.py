from keras.models import Model, Sequential, load_model
import ROOT, numpy, sys, math
from ROOT import TCanvas, TGraph
from ROOT import gROOT
from array import array
from numpy import loadtxt, expand_dims
import h5py
import os

#Load in the network to be tested, the testing data, 1 minus the WP, the output image title, and the jet data
if len(sys.argv)>6 or len(sys.argv)<5:
	print("USAGE: <input model> <input testing data> <percent of bkg events to remove (1-99)> <desired image name> <jet full data list (optional)>")
	sys.exit(1)

#Load in testing data and check if jet data has been included
with h5py.File(str(sys.argv[2]), 'r') as hf:
    dataset = hf["Testing Data"][:]
hasJetData = False
if len(sys.argv)==5:
	print("No Jet Data File included, will be reconstructing jets")
if len(sys.argv)==6:
	print("Jet Data File included, will not reconstruct jets here")
	hasJetData = True
	#jetData = loadtxt(str(sys.argv[4]),delimiter=',')
	with h5py.File(str(sys.argv[5]), 'r') as hf:
		jetData = hf["Jet Data"][:]

#Separate testing data into inputs and outputs
X = dataset[:,0:len(dataset[0])-1]
y = dataset[:,len(dataset[0])-1]
X = expand_dims(X, axis=3)

#Load the network
model = load_model(str(sys.argv[1]))

#Sig Eff : A/(A+D)
#Bkg Eff : B/(B+C)
#Fake Rate : D/(B+C)

A = ROOT.TH1F("Jet Pt Histogram CS", "PtCS", 100, 0, 200) #correct signal, pt
B = ROOT.TH1F("Jet Pt Histogram CB", "PtCB", 100, 0, 200) #correct background, pt
C = ROOT.TH1F("Jet Pt Histogram IS", "PtIS", 100, 0, 200) #incorrect signal, pt
D = ROOT.TH1F("Jet Pt Histogram IB", "PtIB", 100, 0, 200) #incorrect background, pt
E = ROOT.TH1F("Jet Eta Histogram CS", "EtaCS", 100, -2.4, 2.4) #correct signal, eta
F = ROOT.TH1F("Jet Eta Histogram CB", "EtaCB", 100, -2.4, 2.4) #corrective background, eta
G = ROOT.TH1F("Jet Eta Histogram IS", "EtaIS", 100, -2.4, 2.4) #incorrect signal, eta
H = ROOT.TH1F("Jet Eta Histogram IB", "EtaIB", 100, -2.4, 2.4) #incorrect background, eta
I = ROOT.TH1F("Jet Mass Histogram CS", "MassCS", 100, 0, 30) #correct signal, mass
J = ROOT.TH1F("Jet Mass Histogram CB", "MassCB", 100, 0, 30) #correct background, mass
K = ROOT.TH1F("Jet Mass Histogram IS", "MassIS", 100, 0, 30) #incorrect signal, mass
L = ROOT.TH1F("Jet Mass Histogram IB", "MassIB", 100, 0, 30) #incorrect background, mass

#Calculate the NN outputs cutoff corresponding to the WP
predProbs = model.predict(X)
NNThreshold = 0
bkgCutoffArray = ROOT.TH1F("NN Outputs for All Bkg Events","NN Neg Outputs",100,0,1)
for i in range(len(y)):
	if y[i]==0:
		bkgCutoffArray.Fill(predProbs[i])

totalBkg = 0
for i in range(bkgCutoffArray.GetNbinsX()+1):
	totalBkg+=bkgCutoffArray.GetBinContent(i)
sumBkg = 0

for i in range(bkgCutoffArray.GetNbinsX()+1):
	sumBkg+=bkgCutoffArray.GetBinContent(i)
	if 100*(sumBkg/totalBkg)>=int(sys.argv[3]):
		NNThreshold = i
		break

print("Cutoff for NN Outputs given "+str(sys.argv[3])+"% working point: "+str(NNThreshold))

#Reconstruct the jets if jet data not provided
for j in range(len(y)):
	jet = ROOT.TLorentzVector()
	jetPt = 0
	jetMass = 0
	jetTLVs = []
	if hasJetData==False:
		for k in range(0,5):
			massList = X[j][k*8:k*8+4]
			partMass = 0
			for ind in range(len(massList)):
				if massList[ind]==1:
					if ind==0: #Electron
						partMass=0.510*(10**(-3))
						break
					elif ind==1: #Photon
						partMass=0
						break
					elif ind==2: #K-Long Meson
						partMass=497.648*(10**(-3))
						break
					elif ind==3: #Pion
						partMass=139.570*(10**(-3))
						break
					elif ind==4: #Muon
						partMass=105.658*(10**(-3))
			part = ROOT.TLorentzVector()
			part.SetPtEtaPhiM(X[j][k*8+5],X[j][k*8+6],X[j][k*8+7],partMass)
			jetTLVs.append(part)
		jet = jetTLVs[0]+jetTLVs[1]+jetTLVs[2]+jetTLVs[3]+jetTLVs[4]
	#Fill in the histograms with different jets
	predValue = 0
	if (predProbs[j]*100)>=NNThreshold:
		predValue=1
	trueValue = y[j]
	if hasJetData==True:
		if predValue==1 and trueValue==1:
			A.Fill(jetData[j][0]) #True signal classified as signal
			if jetData[j][0]>=30:
				E.Fill(jetData[j][1])
				I.Fill(jetData[j][3])
		elif predValue==0 and trueValue==0:
			B.Fill(jetData[j][0]) #True background classified as background
			if jetData[j][0]>=30:
				F.Fill(jetData[j][1])
				J.Fill(jetData[j][3])
		elif predValue==1 and trueValue==0:
			C.Fill(jetData[j][0]) #True background classified as signal
			if jetData[j][0]>=30:
				G.Fill(jetData[j][1])
				K.Fill(jetData[j][3])
		elif predValue==0 and trueValue==1:
			D.Fill(jetData[j][0]) #True signal classified as background
			if jetData[j][0]>=30:
				H.Fill(jetData[j][1])
				L.Fill(jetData[j][3])
	if hasJetData==False:
		if predValue==1 and trueValue==1:
			A.Fill(jet.Pt()) #True signal classified as signal
			if jetData[j][0]>=30:
				E.Fill(jet.Eta())
				I.Fill(jet.M())
		elif predValue==0 and trueValue==0:
			B.Fill(jet.Pt()) #True background classified as background
			if jetData[j][0]>=30:
				F.Fill(jet.Eta())
				J.Fill(jet.M())
		elif predValue==1 and trueValue==0:
			C.Fill(jet.Pt()) #True background classified as signal
			if jetData[j][0]>=30:			
				G.Fill(jet.Eta())
				K.Fill(jet.M())
		elif predValue==0 and trueValue==1:
			D.Fill(jet.Pt()) #True signal classified as background
			if jetData[j][0]>=30:
				H.Fill(jet.Eta())
				L.Fill(jet.M())

#Sig Eff : A/(A+D)
#Bkg Eff : B/(B+C)
#Fake Rate : D/(B+C)

#Initialize the TCanvas and draw the plots
c1 = TCanvas( 'Plots',  'Efficiency Curves', 2, 100, 2400, 600 )
c1.Divide(4,1)

c1.cd(1)
latex = ROOT.TLatex()
latex.DrawText(0,.9,"dZ+dXY Model w/ PUPPI Candidates")
latex.DrawText(0,0.7, "Working Point: "+str(100-int(sys.argv[3]))+"%")
latex.DrawText(0,0.6, "No. of jets used: "+str(len(y)))
sigCount = 0
for i in range(len(y)):
	if y[i]==1:
		sigCount+=1
latex.DrawText(0,0.5,"Signal Rate for Input Data: "+str(round(100*(float(sigCount)/float(len(y))),3))+"%")
c1.Update()


#gr = ROOT.TGraphAsymmErrors(hist_num, hist_den)
#Pt Sig/Bkg Eff
c1.cd(2)
s1 = A.Clone("Sig Eff")
s1.Sumw2()
denSigEffPt = A+D
s1.Divide(denSigEffPt)
denBkgEffPt = B+C
b1 = B.Clone("Bkg Eff")
b1.Divide(denBkgEffPt)
mgr1 = ROOT.TMultiGraph()
#grs1 = ROOT.TGraph(s1)
grs1 = ROOT.TGraphAsymmErrors(A,denSigEffPt)
grs1.SetMarkerStyle( 2 )
grs1.SetMarkerColor(4)
mgr1.Add(grs1)
#grb1 = ROOT.TGraph(b1)
grb1 = ROOT.TGraphAsymmErrors(C,denBkgEffPt)
grb1.SetMarkerStyle( 2 )
grb1.SetMarkerColor(1)
mgr1.Add(grb1)
mgr1.SetTitle("pT Sig/Bkg Eff Curve")
mgr1.GetXaxis().SetTitle( 'pT (GeV)' )
mgr1.GetYaxis().SetTitle( 'Efficiency' )
ROOT.gStyle.SetPalette(1)
mgr1.Draw( 'APPLC' )
leg = ROOT.TLegend(0.8,.9,.9,1.0)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.035)
leg.AddEntry(grs1,"Sig Eff","lep")
leg.AddEntry(grb1,"Bkg Eff","lep")
leg.Draw()
c1.Update()


#Eta Sig/Bkg Eff
c1.cd(3)
s2 = E.Clone("Sig Eff")
s2.Sumw2()
denSigEffEta = E+H
s2.Divide(denSigEffEta)
denBkgEffEta = F+G
b2 = F.Clone("Bkg Eff")
b2.Sumw2()
b2.Divide(denBkgEffEta)
for i in range(b2.GetNbinsX()):
	b2.SetBinContent(i,1-b2.GetBinContent(i))
mgr2 = ROOT.TMultiGraph()
#grs2 = ROOT.TGraph(s2)
grs2 = ROOT.TGraphAsymmErrors(E,denSigEffEta)
grs2.SetMarkerStyle( 2 )
grs2.SetMarkerColor(4)
mgr2.Add(grs2)
#grb2 = ROOT.TGraph(b2)
grb2 = ROOT.TGraphAsymmErrors(G,denBkgEffEta)
grb2.SetMarkerStyle( 2 )
grb2.SetMarkerColor(1)
mgr2.Add(grb2)
mgr2.SetTitle("Eta Sig/Bkg Eff Curve")
mgr2.GetXaxis().SetTitle( 'Eta' )
mgr2.GetYaxis().SetTitle( 'Efficiency' )
ROOT.gStyle.SetPalette(1)
mgr2.Draw( 'APPLC' )
leg1 = ROOT.TLegend(0.8,.9,.9,1.0)
leg1.SetBorderSize(0)
leg1.SetFillColor(0)
leg1.SetFillStyle(0)
leg1.SetTextFont(42)
leg1.SetTextSize(0.035)
leg1.AddEntry(grs2,"Sig Eff","lep")
leg1.AddEntry(grb2,"Bkg Eff","lep")
leg1.Draw()
c1.Update()

#Mass Sig/Bkg Eff
c1.cd(4)
s3 = I.Clone("Sig Eff")
s3.Sumw2()
denSigEffMass = I+L
s3.Divide(denSigEffMass)
denBkgEffMass = J+K
b3 = J.Clone("Bkg Eff")
b3.Sumw2()
b3.Divide(denBkgEffMass)
for i in range(b3.GetNbinsX()):
	b3.SetBinContent(i,1-b3.GetBinContent(i))
mgr3 = ROOT.TMultiGraph()
#grs3 = ROOT.TGraph(s3)
grs3 = ROOT.TGraphAsymmErrors(I,denSigEffMass)
grs3.SetMarkerStyle( 2 )
grs3.SetMarkerColor(4)
mgr3.Add(grs3)
#grb3 = ROOT.TGraph(b3)
grb3 = ROOT.TGraphAsymmErrors(K,denBkgEffMass)
grb3.SetMarkerStyle( 2 )
grb3.SetMarkerColor(1)
mgr3.Add(grb3)
mgr3.SetTitle("Mass Sig/Bkg Eff Curve")
mgr3.GetXaxis().SetTitle( 'Mass (GeV)' )
mgr3.GetYaxis().SetTitle( 'Efficiency' )
ROOT.gStyle.SetPalette(1)
mgr3.Draw( 'APPLC' )
leg2 = ROOT.TLegend(0.8,.9,.9,1.0)
leg2.SetBorderSize(0)
leg2.SetFillColor(0)
leg2.SetFillStyle(0)
leg2.SetTextFont(42)
leg2.SetTextSize(0.035)
leg2.AddEntry(grs3,"Sig Eff","lep")
leg2.AddEntry(grb3,"Bkg Eff","lep")
leg2.Draw()
c1.Update()

#Save the image
c1.SaveAs(sys.argv[4])
