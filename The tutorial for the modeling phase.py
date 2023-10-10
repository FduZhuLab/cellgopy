#The related files involved in tutorials are available at: http://www.cellgo.world
from CellGO import Modeling
import pandas as pd
import numpy as np
import scanpy as sc

#####Step1 Importing the single-cell expression matrix and the cell identity list
GeneExpMatrix=sc.read('Example_ExpressionMatrix.csv.gz')
CellType=list(pd.read_csv('Example_CellTypeList.csv.gz').iloc[:,0])

#View the size of the example matrix
print(GeneExpMatrix.X.shape)

#View identities of cells. Here, 0 represents excitatory neurons, 1 represents oligodendrocytes, and 2 represents microglia
print(CellType)

#####Step2 Generating necessary files
#Prepare an empty folder for storing files
OutputDir='/home/user/TOPOLOGY_human/'
import os
os.makedirs(OutputDir)

#Download the ontology file from "http://purl.obolibrary.org/obo/go/go-basic.obo"
GO_OBO_FILE='2022-7-1-go-basic.obo'

#Download the species-specific GO annotation file from "http://current.geneontology.org/products/pages/downloads.html"
#Here we combined "goa_human.gaf" and "goa_human_isoform.gaf" for more annotations
GOA_GAF_FILE='2022-7-1-goa_human_combined.gaf.gz'

#Generate necessary files
#help(Modeling.ExportTopology)
Modeling.ExportTopology(GO_OBO_FILE,GOA_GAF_FILE,OutputDir)

#Note! "allGO_infor.csv" is necessary for the analysis phase, which provides details about each GO term (i.e., pathway)
#####Step3 Converting the single-cell expression matrix and the cell identity list to tensor
import torch

#Extract genes in the single-cell expression matrix
Genes=list(GeneExpMatrix.obs.index)
print(len(Genes))

#Extract GO-annotated genes
AnnotatedGenes=list(pd.read_csv(OutputDir+'/allGene.csv').iloc[:,0])
print(len(AnnotatedGenes))

#Keep genes in the single-cell expression matrix consistent with GO-annotated genes
GeneExpMatrix_annotated=np.zeros((len(AnnotatedGenes),GeneExpMatrix.X.shape[1]))
index1=[]
index2=[]
for i in range(len(AnnotatedGenes)):
	if AnnotatedGenes[i] in Genes:
		index1.append(Genes.index(AnnotatedGenes[i]))
		index2.append(i)
GeneExpMatrix_annotated[index2,]=GeneExpMatrix.X[index1,]

#Converting them to tensor
GeneExpMatrix_training=torch.from_numpy(np.float32(GeneExpMatrix_annotated))
GeneExpMatrix_training=torch.transpose(GeneExpMatrix_training,0,1)
CellTypeLabel_training=torch.LongTensor(CellType)

#####Step4 Using the "visible" neural network (VNN) to score the cell type-specific activity of each gene-term pair
#help(Modeling.CellGOModeling)
#Set parameters
config={}
#The name of the output file containing raw cell type-specific active scores of all gene-term pairs, default "RawScores.csv"
config['OutputFileName']='RawScores.csv'
#This function trains each subtree-matched VNN separately and infers cell type-specific active scores of all gene-term pairs within the subtree, therefore, this function outputs every N (default 100) subtrees
config['ResultOutputInterval']=100
#Whether to use cuda, default False
config['Use_Cuda']=True
#The identifier of the used GPU, default 0
config['Cuda_Device']=0
#Whether to randomly sample cells. It is suitable for data with uneven distribution of cell identities, default False
config['Sampler']=False
#A list of numbers. It represents the number of randomly sampled cells of each identity during VNN training, default []
config['Sample_Num']=[]
#The batch size, default 100
config['BatchSize']=30
#The epoch number of VNN training, default 15
config['Epoch']=15
#The learning rate, default 0.001
config['lr']=0.001
#The exponential decay rates for the 1st and 2nd moment estimates, default (0.9, 0.99)
config['betas']=(0.9, 0.99)
#The L2 regularized coefficient, default 0.001
config['weight_decay']=0.001
#If the number of genes annotated in the leaf-node-level pathway is greater than N (default 100), the VNN matched with this subtree will be skipped
config['SkipGeneNum']=100
#The loss weight (default 0.3) of the non-top-level pathway compared with the top-level pathway
config['RelativeLossWeight']=0.3

Modeling.CellGOModeling(GeneExpMatrix_training,CellTypeLabel_training,OutputDir,config)

#####Step5 Processing raw cell type-specific gene-term scores
#A list of file names. If you run the "CellGOModeling" function with the same gene expression matrix and the cell identity list for multiple times and get multiple output files, you can use these files as input to increase reliability
File_List=['RawScores.csv']
#A list of cell type names. Names and orders of these cell types must be corresponding to cell identites
CellType_Name=['ExN','Oligo','Microglia']

#help(Modeling.ProcessRawData)
Processed=Modeling.ProcessRawData(File_List,CellType_Name)
Processed.to_csv('Processed.csv',index=False,sep=',')

#Finally, please save "Processed.csv" and "allGO_infor.csv" (see above) for the analysis phase
