#The related files involved in tutorials are available at: http://www.cellgo.world
from CellGO import Analysis
import pandas as pd
import numpy as np

#####Step1 Importing modeling results and pathway information
#Use the modeling results from an adult human PFC dataset
ModelingResults=pd.read_csv('Processed_1.csv.gz')
#Import pathway information of the human
PathwayInfor=pd.read_csv('allGO_infor_Human.csv.gz')

#####Step2 Query CellGO annotation for a single gene
#help(Analysis.SingleGeneAnnotation)
GeneAnnotation=Analysis.SingleGeneAnnotation('MBP',ModelingResults,PathwayInfor,OutputHeatmap=True,OutputHeatmapName='Heatmap.pdf',SignificanceLevel=[0.05,0.01])
GeneAnnotation.to_csv('GeneAnnotation_MBP.csv',index=False,sep=',')

#####Step3 Cell type enrichment for a gene list
#Analyze 116 ASD risk genes from GWAS
GeneList=list(pd.read_csv('GENES_GWAS_ASD.csv.gz').iloc[:,0])

#help(Analysis.CellTypeEnrichment)
pvalue_ks=Analysis.CellTypeEnrichment(GeneList,ModelingResults,method='KsTest')

#####Step4 Basic CellGO analysis for a gene list
#Prepare for the running of the "CellGO" function
#help(Analysis.PermutationTest)
PermutationTestPrepare=Analysis.PermutationTest(ModelingResults,PathwayInfor,Max=200)
#Note! Please ensure "Max" is larger than the length of the analyzed gene list

#Run basic CellGO analysis
#help(Analysis.CellGO)
CellGOresults=Analysis.CellGO(GeneList,ModelingResults,PathwayInfor,PermutationTestPrepare)
print(CellGOresults.keys())

#Output results
CellGOresults['Results'].to_csv('Results.csv',index=False,sep=',')
#Keep it for pathway network analysis
NetworkPlotPrepare=CellGOresults['NetworkPlotPrepare']

from plotnine import *
#Plot density distribution of P-values of pathways for different cell types
#help(Analysis.PvalueDensityPlot)
p=Analysis.PvalueDensityPlot(CellGOresults['Results'])
#You can change plotting parameters as following
p=(p+
labs(title="Density distribution of P-values of pathways")
)
ggplot.save(p,filename='DensityPlot.pdf')

#Plot the numbers of pathways that meet different significance levels for different cell types
#help(Analysis.PvalueBarPlot)
p=Analysis.PvalueBarPlot(CellGOresults['Results'],SignificanceLevel=[0.05,0.01])
ggplot.save(p,filename='BarPlot.pdf')

#####Step5 Cell type-specific pathway network analysis
from CellGO import Network

#Select a cell type and a pathway type
AllCellType=list(ModelingResults.columns[list(range(0,len(ModelingResults.columns),3))])
AllGoType=['biological_process','molecular_function']
print(AllCellType)
print(AllGoType)
CellType=AllCellType[0]
GoType=AllGoType[0]

#Generate the ExN-specific pathway network
#help(Network.GetNetwork)
G=Network.GetNetwork(CellType,GoType,NetworkPlotPrepare,PathwayInfor)

#Determine pathway communities
#help(Network.CommunityPartition)
PartitionResults=Network.CommunityPartition(G,PathwayInfor,CellGO_pvalue_cutoff=0.01,GO_pvalue_cutoff=1,PlotPathwayNetwork=True,PlotCommunityNetwork=True,OutputFileName=['PathwayNetwork.pdf','CommunityNetwork.pdf'])
print(PartitionResults.keys())

#Output community partition results
PartitionResults['Partition_results'].to_csv('PathwayCommunityDetails.csv',index=False,sep=',')
PartitionResults['Partition_summary'].to_csv('PathwayCommunitySummary.csv',index=False,sep=',')

#Plot the pathway network within a community
#help(Network.PlotsubNetwork)
print(set(PartitionResults['Partition_results']['Cluster']))
ClusterID=0
Network.PlotsubNetwork(PartitionResults['Partition_results'],ClusterID,PathwayInfor,OutputFileName='subNetwork.pdf')

#Congratulations! You have mastered CellGO!
