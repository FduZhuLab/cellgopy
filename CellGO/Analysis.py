import sys
import os
import math
import copy
import numpy as np
import pandas as pd
import textwrap
import seaborn as sns
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import MinMaxScaler
from plotnine import *
import matplotlib.pyplot as plt

def PermutationTest(ModelingResults,PathwayInfor,Max=200,Times=100000):
	"""
	This function is used to prepare for the running of the "CellGO" function.
	Parameters:
		"ModelingResults":Imported modeling results with class "pandas.core.frame.DataFrame"
		"PathwayInfor":Imported pathway information with class "pandas.core.frame.DataFrame"
		"Max":The maximum number of genes for random sampling, default 200. Please ensure that this value is larger than the length of the analyzed gene list
		"Times":The number of random sampling, default 100,000
	"""
	TrainingResults=ModelingResults
	columns=list(TrainingResults.columns)
	CellType=list(TrainingResults.columns[list(range(0,len(TrainingResults.columns),3))])
	CellTypeNum=len(CellType)
	out={}
	out_mean={}
	out_std={}
	for i in range(CellTypeNum):
		out[CellType[i]]={}
		out_mean[CellType[i]]={}
		out_std[CellType[i]]={}
		SCORE=TrainingResults[columns[i*3]]
		GENE=TrainingResults[columns[i*3+1]]
		GO=TrainingResults[columns[i*3+2]]
		store={}
		for j in range(len(SCORE)):
			if GENE[j] not in store.keys():
				store[GENE[j]]=[SCORE[j]]
			else:
				store[GENE[j]].append(SCORE[j])
		SCORES=[]
		for GENE in store.keys():
			SCORES.append(np.mean(np.array(store[GENE])))
		SCORES=np.array(SCORES)
		len_array=len(SCORES)
		rand_index=np.random.randint(len_array,size=Max*Times)
		rand_sample=SCORES[rand_index].reshape(Times,Max)
		for j in range(1,Max+1):
			slice=np.sum(rand_sample[:,0:j],axis=1)
			linespace=np.float16(np.quantile(slice,np.linspace(0,1,10000)))
			out[CellType[i]][j]=ECDF(linespace)
			out_mean[CellType[i]][j]=np.mean(slice)
			out_std[CellType[i]][j]=np.std(slice)
	Go_Convert={}
	for i in range(PathwayInfor.shape[0]):
		Go_Convert[PathwayInfor['Name'][i]]=PathwayInfor['Go'][i]
	Go_Gene={}
	for i in range(PathwayInfor.shape[0]):
		Go_Gene[PathwayInfor[PathwayInfor.columns[0]][i]]=set()
	GENE=TrainingResults[columns[1]]
	GO=TrainingResults[columns[2]]
	for i in range(len(GO)):
		Go_Gene[Go_Convert[GO[i]]].update([GENE[i]])
	Go_bottom=[]
	Go_children={}
	for i in range(PathwayInfor.shape[0]):
		if pd.isnull(PathwayInfor[PathwayInfor.columns[3]][i]) == True:
			Go_bottom.append(PathwayInfor[PathwayInfor.columns[0]][i])
			Go_children[PathwayInfor[PathwayInfor.columns[0]][i]]=set()
		else:
			Go_children[PathwayInfor[PathwayInfor.columns[0]][i]]=set(PathwayInfor[PathwayInfor.columns[3]][i].split(' '))
	Go_leafnodes={}
	Go_geneNum={}
	for i in range(PathwayInfor.shape[0]):
		GoTerm=PathwayInfor[PathwayInfor.columns[0]][i]
		leafnodes=set()
		genes=set()
		DeepGO=set()
		DeepGO.add(GoTerm)
		while len(DeepGO)>0:
			GoChildren=set()
			for Go in DeepGO:
				leafnodes.add(Go)
				genes.update(Go_Gene[Go])
				GoChildren.update(Go_children[Go])
			DeepGO = GoChildren
		Go_leafnodes[GoTerm]=leafnodes
		Go_geneNum[GoTerm]=len(genes)
	Summary={}
	Summary['PermutationTestPrepare']=out
	Summary['Mean']=out_mean
	Summary['Std']=out_std
	Summary['Go_Convert']=Go_Convert
	Summary['Go_bottom']=Go_bottom
	Summary['Go_children']=Go_children
	Summary['Go_leafnodes']=Go_leafnodes
	Summary['Go_geneNum']=Go_geneNum
	return(Summary)

def SingleGeneAnnotation(Gene,ModelingResults,PathwayInfor,OutputHeatmap=True,OutputHeatmapName='Heatmap.pdf',SignificanceLevel=[0.05,0.01],annotation=True,annotation_fontsize=18,x_labelsize=12,y_labelsize=12,title='auto',titlesize=30,cmap='Reds',vmin=0,vmax=1.2,linewidths=2):
	"""
	This function is used to query CellGO annotation for a single gene.
	Parameters:
		"Gene":A single gene (e.g., "OLIG1")
		"ModelingResults":Imported modeling results with class "pandas.core.frame.DataFrame"
		"PathwayInfor":Imported pathway information with class "pandas.core.frame.DataFrame"
		"OutputHeatmap":Whether to output the heatmap, default True
		"OutputHeatmapName":The output file name of the heatmap, default "Heatmap.pdf"
		"SignificanceLevel":A list of numbers in the 0-1 range from large to small, indicating different significance levels, default [0.05,0.01]
		"annotation":Whether to show significance level markers, default True
		"annotation_fontsize":The font size of significance level markers, default 18
		"x_labelsize" and "y_labelsize":The font size of X and Y axis tick labels, default 12
		"title":If "auto" it uses the input gene as title, else it uses the input character string as title, default "auto"
		"titlesize":The font size of the title, default 30
		"cmap", "vmin", "vmax" and "linewidths":see the "seaborn.heatmap" function for details
	"""
	TrainingResults=ModelingResults
	if type(Gene) != str:
		raise TypeError('Error! Please input a single gene!')
	CellType=list(TrainingResults.columns[list(range(0,len(TrainingResults.columns),3))])
	CellTypeNum=len(CellType)
	columns=list(TrainingResults.columns)
	extract={}
	for i in range(CellTypeNum):
		loc=TrainingResults[TrainingResults[columns[i*3+1]]==Gene]
		locNum=TrainingResults[columns[i*3]].dropna().shape[0]
		if loc.shape[0]>0:
			extract[CellType[i]]={}
			extract[CellType[i]]['Pathway']=list(loc[columns[i*3+2]])
			extract[CellType[i]]['Score']=list(loc[columns[i*3]])
			extract[CellType[i]]['Rank']=list(np.array(loc.index)/locNum)
	if len(extract.keys()) == 0:
		raise ValueError('Error! The input gene is not in the annotated gene set!')
	Pathway=set()
	for i in range(len(CellType)):
		Pathway.update(extract[CellType[i]]['Pathway'])
	Pathway=sorted(list(Pathway))
	ylab=copy.deepcopy(Pathway)
	Go_Convert={}
	Go_type={}
	for i in range(PathwayInfor.shape[0]):
		Go_Convert[PathwayInfor['Name'][i]]=PathwayInfor['Go'][i]
		Go_type[PathwayInfor['Name'][i]]=PathwayInfor['Type'][i]
	Pathway_id=[]
	Pathway_type=[]
	for i in range(len(ylab)):
		ylab[i]=textwrap.fill(ylab[i],width=30)
		Pathway_id.append(Go_Convert[Pathway[i]])
		Pathway_type.append(Go_type[Pathway[i]])
	Matrix=pd.DataFrame(np.zeros((len(Pathway),CellTypeNum)),index=Pathway,columns=CellType)
	Matrix_marker=pd.DataFrame(np.zeros((len(Pathway),CellTypeNum)),index=Pathway,columns=CellType)
	summary={}
	summary['GO_id']=Pathway_id
	summary['GO_name']=Pathway
	summary['GO_type']=Pathway_type
	for i in range(len(CellType)):
		subPathway=extract[CellType[i]]['Pathway']
		subRank=extract[CellType[i]]['Rank']
		subScore=extract[CellType[i]]['Score']
		summary['Pvalue_'+CellType[i]]=[1 for path in Pathway]
		summary['Score_'+CellType[i]]=[0 for path in Pathway]
		for j in range(len(subPathway)):
			Value=0
			Marker=0
			for z in range(len(SignificanceLevel)):
				if subRank[j] <= SignificanceLevel[z]:
					Value=(z+1)/len(SignificanceLevel)
					Marker=z+1
			Matrix[CellType[i]][subPathway[j]]=Value
			Matrix_marker[CellType[i]][subPathway[j]]=Marker
			summary['Score_'+CellType[i]][Pathway.index(subPathway[j])]=-float('%.3g' % subScore[j])
			summary['Pvalue_'+CellType[i]][Pathway.index(subPathway[j])]=float('%.3g' % subRank[j])
	if OutputHeatmap == True:
		Matrix.index=ylab
		plt.clf()
		f,ax=plt.subplots(figsize=(len(CellType)+6,len(ylab)))
		if annotation == False:
			Matrix_marker=False
		sns.heatmap(Matrix,cmap=cmap,annot=Matrix_marker,linewidths=linewidths,vmax=vmax,vmin=vmin,annot_kws={'fontsize':annotation_fontsize},cbar=False,square=True)
		plt.tick_params(axis='x',labelsize=x_labelsize)
		plt.tick_params(axis='y',labelsize=y_labelsize)
		if title=='auto':
			title=Gene
		plt.title(Gene,fontsize=titlesize)
		plt.savefig(OutputHeatmapName)
	summary=pd.DataFrame(summary)
	return(summary)

def CellTypeEnrichment(GeneList,ModelingResults,method='KsTest',FisherExactTestCut=0.05):
	"""
	This function is used to perform cell type enrichment for a gene list.
	Parameters:
		"GeneList":A list of genes
		"ModelingResults":Imported modeling results with class "pandas.core.frame.DataFrame"
		"method":The statistical method for cell type enrichment, default "KsTest". This function provides 2 methods: "KsTest" and "FisherExactTest"
		"FisherExactTestCut":The parameter of the Fisher's exact test, default 0.05
	"""
	TrainingResults=ModelingResults
	if type(GeneList) != list:
		raise TypeError('Error! Please input a gene list!')
	if method not in ['KsTest','FisherExactTest']:
		raise ValueError('Error! Please input one of the two test methods: KsTest or FisherExactTest!')
	CellType=list(TrainingResults.columns[list(range(0,len(TrainingResults.columns),3))])
	CellTypeNum=len(CellType)
	columns=list(TrainingResults.columns)
	extract={}
	for i in range(CellTypeNum):
		Genes=list(TrainingResults[columns[i*3+1]].dropna())
		Scores=list(TrainingResults[columns[i*3]].dropna())
		locNum=len(Genes)
		rank=[]
		score=[]
		scoreKeep=[]
		for Gene in enumerate(Genes):
			if Gene[1] in GeneList:
				rank.append(Gene[0])
				score.append(Scores[Gene[0]])
			else:
				scoreKeep.append(Scores[Gene[0]])
		if len(rank) > 0:
			extract[CellType[i]]={}
			extract[CellType[i]]['rank']=np.array(rank)/locNum
			extract[CellType[i]]['score']=np.array(score)
			extract[CellType[i]]['scoreKeep']=np.array(scoreKeep)
			extract[CellType[i]]['locNum']=locNum
	if len(extract.keys()) == 0:
		raise ValueError('Error! The input gene list is not in the annotated gene set!')
	Type=list(extract.keys())
	P=np.ones(CellTypeNum)
	for i in range(len(Type)):
		if method == 'KsTest':
			p=stats.kstest(extract[Type[i]]['score'],extract[Type[i]]['scoreKeep'],alternative='greater').pvalue
			P[CellType.index(Type[i])]=float('%.3g' % p)
		if method == 'FisherExactTest':
			num1=sum(extract[Type[i]]['rank'] <= FisherExactTestCut)
			num2=int(extract[Type[i]]['locNum']*FisherExactTestCut)-num1
			num3=sum(extract[Type[i]]['rank'] > FisherExactTestCut)
			num4=int(extract[Type[i]]['locNum']*(1-FisherExactTestCut))-num3
			p=stats.fisher_exact([[num1,num2],[num3,num4]],alternative='greater')[1]
			P[CellType.index(Type[i])]=float('%.3g' % p)
	out=pd.DataFrame(P,index=CellType,columns=[method])
	return(out)

def CellGO(GeneList,ModelingResults,PathwayInfor,PermutationTestPrepare,SkipGeneNumber=1000):
	"""
	This function is used to perform basic CellGO analysis for a gene list.
	Parameters:
		"GeneList":A list of genes
		"ModelingResults":Imported modeling results with class "pandas.core.frame.DataFrame"
		"PathwayInfor":Imported pathway information with class "pandas.core.frame.DataFrame"
		"PermutationTestPrepare":The output of the function "PermutationTest"
		"SkipGeneNumber":If there are more than N (default 1000) genes included in both the input gene list and a pathway, P-values for that pathway will not be calculated
	"""
	TrainingResults=ModelingResults
	Go_Convert=PermutationTestPrepare['Go_Convert']
	Go_bottom=PermutationTestPrepare['Go_bottom']
	Go_children=PermutationTestPrepare['Go_children']
	Go_leafnodes=PermutationTestPrepare['Go_leafnodes']
	Go_geneNum=PermutationTestPrepare['Go_geneNum']
	PermutationTest=PermutationTestPrepare['PermutationTestPrepare']
	Mean=PermutationTestPrepare['Mean']
	Std=PermutationTestPrepare['Std']
	if type(GeneList) != list:
		raise TypeError('Error! Please input a gene list!')
	CellType=list(TrainingResults.columns[list(range(0,len(TrainingResults.columns),3))])
	CellTypeNum=len(CellType)
	columns=list(TrainingResults.columns)
	extract={}
	for i in range(CellTypeNum):
		extract[CellType[i]]={}
		SCORE=TrainingResults[columns[i*3]]
		GENE=TrainingResults[columns[i*3+1]]
		GO=TrainingResults[columns[i*3+2]]
		for j in range(len(GENE)):
			if GENE[j] in GeneList:
				GO_ID=Go_Convert[GO[j]]
				if GO_ID not in extract[CellType[i]].keys():
					extract[CellType[i]][GO_ID]={}
				extract[CellType[i]][GO_ID][GENE[j]]=[SCORE[j]]
	if len(extract[CellType[0]].keys()) == 0:
		raise ValueError('Error! The input gene list is not in the annotated gene set!')
	CellGOOutput={}
	CellGOOutput['GO_id']=[]
	CellGOOutput['GO_name']=[]
	CellGOOutput['Type']=[]
	CellGOOutput['Genes']=[]
	CellGOOutput['Leafnodes']=[]
	GOOutput={}
	GOOutput['Number of genes_total']=[]
	GOOutput['Number of genes_appear']=[]
	GOOutput['Pvalue_fisher exact test']=[]
	for i in range(len(CellType)):
		CellGOOutput['Active score_'+CellType[i]]=[]
		CellGOOutput['Pvalue_'+CellType[i]]=[]
		CellGOOutput['Gene contribution score_'+CellType[i]]=[]
	Startnodes=[]
	BackgroundNum=len(set(TrainingResults[columns[1]]))
	AppearNum=len(set(GeneList)&set(TrainingResults[columns[1]]))
	for i in range(PathwayInfor.shape[0]):
		GoID=PathwayInfor['Go'][i]
		GoName=PathwayInfor['Name'][i]
		GoGeneNum=Go_geneNum[GoID]
		GoType=PathwayInfor['Type'][i]
		interGO=Go_leafnodes[GoID]&extract[CellType[0]].keys()
		if len(interGO) > 0:
			for j in range(len(CellType)):
				combined={}
				mean=[]
				for GO in interGO:
					for key in extract[CellType[j]][GO].keys():
						if key not in combined.keys():
							combined[key]=extract[CellType[j]][GO][key]
						else:
							combined[key]=combined[key]+extract[CellType[j]][GO][key]
				for key in sorted(combined.keys()):
					mean.append(float('%.3g' % np.mean(np.array(combined[key]))))
				CellGOOutput['Gene contribution score_'+CellType[j]].append(list(-np.array(mean)))
				if len(combined.keys()) > SkipGeneNumber:
					CellGOOutput['Pvalue_'+CellType[j]].append(1)
					CellGOOutput['Active score_'+CellType[j]].append(0)
				else:
					Ac=-(np.sum(mean)-Mean[CellType[j]][len(combined.keys())])/(Std[CellType[j]][len(combined.keys())])
					CellGOOutput['Active score_'+CellType[j]].append(float('%.3g' % Ac))
					CellGOOutput['Pvalue_'+CellType[j]].append(float('%.3g' % PermutationTest[CellType[j]][len(combined.keys())](np.sum(mean))))
			CellGOOutput['GO_id'].append(GoID)
			CellGOOutput['GO_name'].append(GoName)
			CellGOOutput['Type'].append(GoType)
			CellGOOutput['Genes'].append(list(sorted(combined.keys())))
			CellGOOutput['Leafnodes'].append(GoID in Go_bottom)
			Num1=len(combined.keys())
			Num2=GoGeneNum
			Num3=AppearNum-Num1
			Num4=BackgroundNum-Num2
			GOOutput['Number of genes_total'].append(GoGeneNum)
			GOOutput['Number of genes_appear'].append(len(combined.keys()))
			GOOutput['Pvalue_fisher exact test'].append(float('%.3g' % stats.fisher_exact([[Num1,Num2],[Num3,Num4]],alternative='greater')[1]))
			Startnodes.append(GoID in extract[CellType[0]].keys())
	summary={}
	summary['CellGO']=pd.DataFrame(CellGOOutput)
	summary['GO']=pd.DataFrame(GOOutput)
	summary=pd.concat(summary,axis=1)
	NetworkPlotPrepare={}
	for i in range(len(CellType)):
		vector=np.array(CellGOOutput['Pvalue_'+CellType[i]])
		vector[vector<0.0001]=0.0001
		vector=list(-np.log10(vector))
		NetworkPlotPrepare[CellType[i]]=vector
	vector=np.array(GOOutput['Pvalue_fisher exact test'])
	vector[vector<0.0000000001]=0.0000000001
	vector=list(-np.log10(vector))
	NetworkPlotPrepare['GO']=vector
	NetworkPlotPrepare['Start']=Startnodes
	NetworkPlotPrepare['Type']=CellGOOutput['Type']
	NetworkPlotPrepare=pd.DataFrame(NetworkPlotPrepare,index=CellGOOutput['GO_id'])
	out={}
	out['Results']=summary
	out['NetworkPlotPrepare']=NetworkPlotPrepare
	return(out)

def PvalueDensityPlot(CellGOresults):
	"""
	This function is used to plot density distribution of P-values of pathways for different cell types.
	Parameters:
		"CellGOresults":Value of key "Results" in the dictionary-like output of the function "CellGO"
	This function returns a "ggplot" object in the "plotnine" package.
	"""
	CellTypeNum=int((CellGOresults.shape[1]-8)/3)
	gap=3
	x=[]
	y=[]
	panel_break=[]
	for i in range(CellTypeNum):
		CellType=CellGOresults.columns[5+gap*i][1].split('_')[1]
		panel_break.append(CellType)
		pvalue=list(CellGOresults[CellGOresults.columns[6+gap*i]].dropna())
		x=x+[str(i) for _ in range(len(pvalue))]
		y=y+pvalue
	data=pd.DataFrame({'CellType':x,'Pvalue':y})
	p=(
	ggplot(data,aes(x='Pvalue',fill='CellType',color='CellType'))+
	geom_density(alpha=.10,size=0.5,adjust=.20)+
	labs(x ='P-value',y='Density',fill='Cell type',color='Cell type',title="Density distribution of P-values of pathways")+
	geom_vline(xintercept=0.05,colour="#990000", linetype="dashed")+
	scale_x_continuous(expand =(0, 0),breaks=(0,0.05,0.25,0.5,0.75,1),labels=['0','0.05','0.25','0.5','0.75','1'])+
	scale_y_continuous(expand =(0, 0))+
	scale_colour_discrete(labels=panel_break)+
	scale_fill_discrete(labels=panel_break)+
	theme_classic()+
	theme(legend_background=element_rect(fill='white'))
	)
	return(p)

def PvalueBarPlot(CellGOresults,SignificanceLevel=[0.05,0.01]):
	"""
	This function is used to plot the numbers of pathways that meet different significance levels for different cell types.
	Parameters:
		"CellGOresults":Value of key "Results" in the dictionary-like output of the function "CellGO"
		"SignificanceLevel":A list of numbers in the 0-1 range from large to small, indicating different significance levels, default [0.05,0.01]
	This function returns a "ggplot" object in the "plotnine" package.
	"""
	CellTypeNum=int((CellGOresults.shape[1]-8)/3)
	gap=3
	x=[]
	y=[]
	z=[]
	panel_break=[]
	for i in range(CellTypeNum):
		CellType=CellGOresults.columns[5+gap*i][1].split('_')[1]
		panel_break.append(CellType)
		pvalue=np.array(CellGOresults[CellGOresults.columns[6+gap*i]].dropna())
		for j in SignificanceLevel:
			x=x+[str(i)]
			y=y+[sum(pvalue<j)]
			z=z+[str(j)]
	data=pd.DataFrame({'CellType':x,'Pathway number':y,'Pvalue cutoff':z})
	p=(
	ggplot(data,aes(x='CellType',y='Pathway number',group='Pvalue cutoff',fill='Pvalue cutoff'))+
	geom_bar(stat="identity",width=0.5,position=position_dodge(width=0.7)) +
	scale_x_discrete(labels=panel_break)+
	scale_y_continuous(expand =(0, 0))+
	labs(x ='Cell type',y='Pathway number',fill='P-value cut-off',title="The numbers of pathways that meet different P-value cut-offs")+
	theme_classic()+
	theme(legend_background=element_rect(fill='white'))
	)
	return(p)
