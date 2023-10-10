import sys
import os
import math
import random
from copy import deepcopy
import numpy as np
import pandas as pd
from goatools import obo_parser
from goatools.rpt.rpt_lev_depth import RptLevDepth
import torch
from torch import nn
import torch.optim as optim

def CellGOModeling(GeneExpMatrix,CellTypeLabel,topoDir,config={'OutputFileName':'RawScores.csv','ResultOutputInterval':100,'Use_Cuda':False,'Cuda_Device':0,'Sampler':False,'Sample_Num':[],'BatchSize':100,'Epoch':15,'lr':0.001,'betas':(0.9, 0.99),'weight_decay':0.001,'SkipGeneNum':100,'RelativeLossWeight':0.3}):
	"""
	This function uses the "visible" neural network (VNN) to score the cell type-specific activity of each gene-term pair.
	Parameters:
		"GeneExpMatrix":The 2D-tensor of a gene expression matrix with rows representing cells and columns representing genes
		"CellTypeLabel":The 1D-tensor of identities of cells. Each identity must be represented by an integer starting from 0. For example: 0 represents excitatory neurons, 1 represents inhibitory neurons and 2 represents oligodendrocytes
		"topoDir":The path where contains necessary files. These files are generated from the "ExportTopology" function
		"config":A dictionary contains parameters of the VNN
			config['OutputFileName']:The name of the output file containing raw cell type-specific active scores of all gene-term pairs, default "RawScores.csv"
			config['ResultOutputInterval']:This function trains each subtree-matched VNN separately and infers cell type-specific active scores of all gene-term pairs within the subtree, therefore, this function outputs every N (default 100) subtrees
			config['Use_Cuda']:Whether to use cuda, default False
			config['Cuda_Device']:The identifier of the used GPU, default 0
			config['Sampler']:Whether to randomly sample cells. It is suitable for data with uneven distribution of cell identities, default False
			config['Sample_Num']:A list of numbers. It represents the number of randomly sampled cells of each identity during VNN training, default []
			config['BatchSize']:The batch size, default 100
			config['Epoch']:The epoch number of VNN training, default 15
			config['lr']:The learning rate, default 0.001
			config['betas']:The exponential decay rates for the 1st and 2nd moment estimates, default (0.9, 0.99)
			config['weight_decay']:The L2 regularized coefficient, default 0.001
			config['SkipGeneNum']:If the number of genes annotated in the leaf-node-level pathway is greater than N (default 100), the VNN matched with this subtree will be skipped
			config['RelativeLossWeight']:The loss weight (default 0.3) of the non-top-level pathway compared with the top-level pathway
	"""
	topoDir=topoDir.strip()
	if topoDir[-1] != '/':
		topoDir=topoDir+'/'
	if os.path.exists(topoDir) == False:
		raise ValueError('Error! The input path does not exist!')
	Data=GeneExpMatrix
	Label_ori=CellTypeLabel
	if(Label_ori.min() > 0):
		Label_ori=Label_ori-Label_ori.min()
	type_num=len(set(Label_ori.numpy()))
	random_right=np.max(np.bincount(Label_ori.numpy()))/np.sum(np.bincount(Label_ori.numpy()))
	Record={}
	Record['Start']=[]
	Record['Gene']=[]
	for i in range(type_num):
		Record[i]=[]
	Record['Random_RightRate']=[]
	Record['Real_RightRate']=[]
	Record['SizeFactor']=[]
	subtree_infor=pd.read_csv(topoDir+"bottomGO_infor.csv")
	all_infor=pd.read_csv(topoDir+"allGO_infor.csv")
	GO_name={}
	for i in range(all_infor.shape[0]):
		GO_name[all_infor['Go'][i]]=all_infor['Name'][i]
	gene_all=list(pd.read_csv(topoDir+"allGene.csv").iloc[:,0])
	leafnodeAllNumber=len(subtree_infor['go'])
	if config['Use_Cuda']==True:
		torch.cuda.set_device(config['Cuda_Device'])
	for leafnodeNumber in range(leafnodeAllNumber):
		topoFile=topoDir+subtree_infor['go'][leafnodeNumber]
		root=subtree_infor['root'][leafnodeNumber]
		start=subtree_infor['go'][leafnodeNumber]
		print('Start the leaf-node-level pathway: '+start)
		bottomsize=subtree_infor['bottomsize'][leafnodeNumber]
		if bottomsize > config['SkipGeneNum']:
			continue
		term_list, gene_list, children_term_map, children_gene_map, term_size_map=load_ontology_file(topoFile)
		gene2id = cal_gene2id(gene_all)
		select=[]
		for i in range(len(gene_list)):
			select.append(gene2id[gene_list[i]])
		DataUsed=deepcopy(torch.index_select(Data,1,torch.LongTensor(select)))
		Label=deepcopy(Label_ori)
		gene2id = cal_gene2id(gene_list)
		neuron_size_map,add_neuron_size_map,term_state = get_neuron_num(term_list,term_size_map)
		add_neuron_size_map[start] = 50
		neuron_size_map[start] = 50
		neuron_size_map[root] = 50
		term_node_layers=None
		net=None
		criterion=None
		optimizer=None
		term_node_layers = make_node_layers(term_state,children_term_map,children_gene_map,neuron_size_map,add_neuron_size_map,type_num,root)
		net = CombineNode(term_node_layers,term_state,children_term_map,children_gene_map)
		criterion = LossCompute(root,config['RelativeLossWeight'])
		if config['Use_Cuda'] == True:
			for term_name,term_layer in zip(term_node_layers.keys(),term_node_layers.values()):
				term_node_layers[term_name] = term_layer.cuda()
			net=net.cuda()
		optimizer = optim.Adam([{'params': model.parameters()} for model in term_node_layers.values()],lr=config['lr'],betas=config['betas'],weight_decay=config['weight_decay'])
		for model in term_node_layers.values():
			model.train()
		net.train()
		for i in range(config['Epoch']):
			torch.cuda.empty_cache()
			if config['Sampler']==True:
				randomNum=[]
				for j in range(type_num):
					randomNum=randomNum+list(np.random.choice(np.argwhere(np.array(Label)==j).reshape(-1),size=config['Sample_Num'][j]))
				random.shuffle(randomNum)
				randomNum=torch.LongTensor(randomNum)
			else:
				randomNum=torch.randperm(DataUsed.size(0))
			if config['Use_Cuda'] == True:
				TrainDataUsed=torch.index_select(DataUsed,0,randomNum).cuda()
				TrainLabel=torch.index_select(Label,0,randomNum).cuda()
			else:
				TrainDataUsed=torch.index_select(DataUsed,0,randomNum)
				TrainLabel=torch.index_select(Label,0,randomNum)
			train_predict = []
			true_labels=[]
			lossVal = []
			NumSamples=TrainDataUsed.size(0)
			for j in range(0,NumSamples,config['BatchSize']):
				z = min(j+config['BatchSize']-1,NumSamples)
				inputs=TrainDataUsed[j:z,:]
				labels=TrainLabel[j:z]
				NN_input=get_NN_input(inputs,term_list,children_gene_map,gene2id,config['Use_Cuda'])
				optimizer.zero_grad()
				outputs = net(NN_input)
				loss = criterion.finalLoss(outputs, labels)
				loss.backward()
				optimizer.step()
				train_predict.append(outputs[root].data)
				true_labels.append(labels)
				lossVal.append(loss)
			CEloss = sum(lossVal)
			train_predict=torch.cat(train_predict,0)
			train_result=torch.max(train_predict,1).indices.cpu()
			true_labels=torch.cat(true_labels,0).cpu()
			print('Epoch:',i+1)
			print('Right rate:',torch.sum(torch.eq(train_result,true_labels))/true_labels.shape[0])
			real_right=torch.sum(torch.eq(train_result,true_labels))/true_labels.shape[0]
			train_result_true=train_result[torch.eq(train_result,true_labels)]
			for z in range(type_num):
				num1=torch.sum(torch.eq(train_result_true,z*torch.ones(train_result_true.shape)))
				num2=torch.sum(torch.eq(true_labels,z*torch.ones(true_labels.shape)))
				print('Right rate for CellType '+str(z)+':',num1/num2)
			print('Loss:',CEloss)
		real_right=float(real_right.numpy())
		if real_right <= random_right:
			SizeFactor = 0
		else:
			SizeFactor=(real_right-random_right)/(1-random_right)
		for model in term_node_layers.values():
			model.eval()
		net.eval()
		if config['Use_Cuda'] == True:
			randomNum=torch.randperm(DataUsed.size(0))
			TrainDataUsed=torch.index_select(DataUsed,0,randomNum).cuda()
		else:
			randomNum=torch.randperm(DataUsed.size(0))
			TrainDataUsed=torch.index_select(DataUsed,0,randomNum)
		subRecord={}
		for i in range(type_num):
			subRecord[i]=[]
		for TREM in term_list:
			for i in range(len(children_gene_map[TREM])):
				print('Perturbated: '+TREM+' '+children_gene_map[TREM][i])
				NN_input=get_NN_KO_input(TrainDataUsed,term_list,children_gene_map,gene2id,TREM,i,config['Use_Cuda'])
				outputs = net(NN_input)
				train_result=torch.max(outputs[root].data,1).indices.cpu()
				Record['Start'].append(GO_name[TREM])
				Record['Gene'].append(children_gene_map[TREM][i])
				Record['Random_RightRate'].append(random_right)
				Record['Real_RightRate'].append(real_right)
				Record['SizeFactor'].append(SizeFactor)
				for j in range(type_num):
					num1=torch.sum(torch.eq(train_result,j*torch.ones(train_result.shape)))
					subRecord[j].append(int(num1))
		for i in range(type_num):
			subRecord[i]=list(np.array(subRecord[i])-np.median(subRecord[i]))
			Record[i]=Record[i]+subRecord[i]
		if leafnodeNumber % config['ResultOutputInterval'] == 0:
			out_csv=pd.DataFrame(Record)
			out_csv.to_csv(config['OutputFileName'],index=False,sep=',')
	out_csv=pd.DataFrame(Record)
	out_csv.to_csv(config['OutputFileName'],index=False,sep=',')

def ExportTopology(GO_OBO_FILE,GOA_GAF_FILE,OutputDir,MinSize=6,InterGoNewGeneCutOff=100):
	"""
	This function generates necessary files of the "CellGOModeling" function.
	Parameters:
		"GO_OBO_FILE":The ontology file from "http://purl.obolibrary.org/obo/go/go-basic.obo"
		"GOA_GAF_FILE":The species-specific GO annotation file from "http://current.geneontology.org/products/pages/downloads.html"
		"OutputDir":The output path
		"MinSize":The minimum number of genes annotated in the leaf-node-level pathway, default 6
		"InterGoNewGeneCutOff":If the number of newly annotated genes in a non-leaf-node-level pathway exceeds N (default 100), these newly annotated genes in that pathway will be removed
	"""
	OutputDir=OutputDir.strip()
	if OutputDir[-1] != '/':
		OutputDir=OutputDir+'/'
	if os.path.exists(OutputDir) == False:
		raise ValueError('Error! The output path does not exist!')
	GO=obo_parser.GODag(GO_OBO_FILE)
	goTree = {}
	for go in GO.keys():
		depth = (GO[go]).depth
		ns = GO[go].namespace
		parents = []
		children = []
		for pGo in GO[go].parents:
			parents.append(pGo.id)
		for cGo in GO[go].children:
			children.append(cGo.id)
		leafNodes = set()
		childTerms1 = GO[go].children
		childTerms2 = set()
		if len(childTerms1) == 0:      
			leafNodes.add(go)
		else:
			while len(childTerms1) !=0: 
				for cTerm in childTerms1:
					if len(GO[cTerm.id].children) == 0:
						leafNodes.add(cTerm.id)
					else:
						childTerms2.update(GO[cTerm.id].children)
				childTerms1 = childTerms2
				childTerms2 = set()
		if ns in goTree.keys():
			if depth in goTree[ns].keys():
				goTree[ns][depth].update({GO[go].id: {"name": GO[go].name, "parents": parents, "children": children, "leafNodes": leafNodes}})
			else:
				goTree[ns][depth] = {GO[go].id: {"name": GO[go].name, "parents": parents, "children": children, "leafNodes": leafNodes}}
		else:
			goTree[ns] = {depth: {GO[go].id: {"name": GO[go].name, "parents": parents, "children": children, "leafNodes": leafNodes}}}
	del goTree['cellular_component']
	goa=pd.read_csv(GOA_GAF_FILE,sep="\t",comment="!",header=None)
	allGOs=set()
	for ns in goTree.keys():
		for depth in goTree[ns].keys():
			for go in goTree[ns][depth].keys():
				allGOs.add(go)
	humanGOA_geneList = {}
	for index, row in goa.iterrows():
		if row[4] in allGOs:
			if pd.isnull(row[2]) == False:                      
				if row[4] in humanGOA_geneList.keys():
					humanGOA_geneList[row[4]].add(row[2])
				else:
					humanGOA_geneList[row[4]] = set([row[2]])
	def giveGene(inputTree,humanGOA_geneList):
		outTree = {}
		for ns in inputTree.keys():
			for depth in inputTree[ns].keys():
				for go in inputTree[ns][depth].keys():
					goGene = set();
					if ns not in outTree.keys():
						outTree[ns] = {}
					if depth not in outTree[ns].keys():
						outTree[ns][depth] = {}
					outTree[ns][depth][go] = inputTree[ns][depth][go]
					if go in humanGOA_geneList.keys():
						goGene = humanGOA_geneList[go]
					outTree[ns][depth][go].update({"gene": goGene})
		return outTree
	trimmedTree = giveGene(goTree,humanGOA_geneList)
	allGOs=[]
	for ns in trimmedTree.keys():
		for depth in trimmedTree[ns].keys():
			for go in trimmedTree[ns][depth]:
				allGOs.append(go)
	for ns in trimmedTree.keys():
		for depth in trimmedTree[ns].keys():
			for go in trimmedTree[ns][depth].keys():
				trimmedTree[ns][depth][go]['children']=list(set(allGOs)&set(trimmedTree[ns][depth][go]['children']))
	def trimTree(cutBranchTree, bottomGOs, minSize):
		outTree = deepcopy(cutBranchTree)
		needRepeat = True
		while needRepeat:
			needRepeat = False
			toTrimGOs = set()
			for go in bottomGOs:
				ns = GO[go].namespace
				depth = GO[go].depth
				if  len(outTree[ns][depth][go]['gene']) < minSize and len(outTree[ns][depth][go]['children']) == 0:
					pGOs = outTree[ns][depth][go]['parents']
					for pGO in pGOs:
						pdepth = GO[pGO].depth
						outTree[ns][pdepth][pGO]['gene'] = outTree[ns][depth][go]['gene'].union(outTree[ns][pdepth][pGO]['gene'])
						outTree[ns][pdepth][pGO]['children'].remove(go)
						if len(outTree[ns][pdepth][pGO]['children']) == 0 and len(outTree[ns][pdepth][pGO]['gene']) < minSize: 
							toTrimGOs.add(pGO)
							needRepeat = True	
					del outTree[ns][depth][go]
			bottomGOs = toTrimGOs
		return outTree
	trimmedBottomGOs = set()
	for ns in trimmedTree.keys():
		for depth in trimmedTree[ns].keys():
			for go in trimmedTree[ns][depth].keys():
				if len(trimmedTree[ns][depth][go]['children']) == 0:
					trimmedBottomGOs.add(go)
	filteredTree=trimTree(trimmedTree,trimmedBottomGOs,minSize=MinSize)
	GObottomgo=list()
	GObottomsize=list()
	GObottomtype=list()
	GObottomdepth=list()
	root=list()
	for ns in filteredTree.keys():
		for depth in filteredTree[ns].keys():
			for go in filteredTree[ns][depth].keys():
				if len(filteredTree[ns][depth][go]['children']) == 0:
					GObottomgo.append(go)
					GObottomtype.append(ns)
					GObottomdepth.append(depth)
					if ns == 'biological_process':
						root.append('GO:0008150')
					if ns == 'molecular_function':
						root.append('GO:0003674')
					GObottomsize.append(len(filteredTree[ns][depth][go]['gene']))
	filtertree=deepcopy(filteredTree)
	for ns in filteredTree.keys():
		for depth in filteredTree[ns].keys():
			for go in filteredTree[ns][depth].keys():
				children = filteredTree[ns][depth][go]['children']
				gene1 = filteredTree[ns][depth][go]['gene']
				gene2 = set()
				while len(children) > 0:
					children2 = set()
					for child in children:
						childrenDepth = GO[child].depth
						gene2.update(filteredTree[ns][childrenDepth][child]['gene'])
						if len(filteredTree[ns][childrenDepth][child]['children']) > 0:
							children2.update(filteredTree[ns][childrenDepth][child]['children'])
					children = children2
				if len(gene2&gene1) > 0:
					filtertree[ns][depth][go]['gene'] = filteredTree[ns][depth][go]['gene'].difference(gene2&gene1)
	filteredTree = deepcopy(filtertree)
	for ns in filteredTree.keys():
		for depth in filteredTree[ns].keys():
			for go in filteredTree[ns][depth].keys():
				if len(filteredTree[ns][depth][go]['children'])>0:
					if len(filteredTree[ns][depth][go]['gene'])>InterGoNewGeneCutOff:
						filteredTree[ns][depth][go]['gene']=set()
	filteredTree['biological_process'][0]['GO:0008150']['gene'] = set()
	filteredTree['molecular_function'][0]['GO:0003674']['gene'] = set()
	allgene=set()
	for ns in filteredTree.keys():
		for depth in filteredTree[ns].keys():
			for go in filteredTree[ns][depth].keys():
				allgene.update(filteredTree[ns][depth][go]['gene'])
	allgene=list(allgene)
	dataframe=pd.DataFrame({'x':allgene})
	dataframe.to_csv(OutputDir+"allGene.csv",index=False,sep=',')
	st = ' '
	tt1 = '\n'
	termsize=[]
	genesize=[]
	for i in range(len(GObottomgo)):
		goterm = GObottomgo[i]
		gotype = GObottomtype[i]
		deepGO=set()
		subtree={}
		genenumb={}
		deepGO.add(goterm)
		while len(deepGO)>0:
			goparents=set()
			for go in deepGO:
				godeep=GO[go].depth
				subtree[go] = deepcopy(filteredTree[gotype][godeep][go])
				goparents.update(filteredTree[gotype][godeep][go]['parents'])
			deepGO = goparents
		for go in subtree.keys():
			godeep=GO[go].depth
			genes=set()
			children=set()
			subtree[go]['children'] = list(set(subtree[go]['children'])&set(subtree.keys()))
			genes.update(subtree[go]['gene'])
			children.update(subtree[go]['children'])
			while len(children)>0:
				childrens=set()
				for childGO in children:
					godeep=GO[childGO].depth
					genes.update(subtree[childGO]['gene'])
					childrens.update(subtree[childGO]['children'])
				children=list(set(childrens)&set(subtree.keys()))
			genenumb[go]=len(genes)
		genes=set()
		for go in subtree.keys():
			genes.update(subtree[go]['gene'])
		genesize.append(len(genes))
		termsize.append(len(subtree.keys()))
		with open(OutputDir+goterm,'a') as file_handle:
			for go in subtree.keys():
				file_handle.writelines('ROOT: '+go+' '+str(genenumb[go])+tt1)
				file_handle.writelines('GENES: '+st.join(subtree[go]['gene'])+tt1)
				file_handle.writelines('TERMS: '+st.join(subtree[go]['children'])+tt1)
		file_handle.close()
	dataframe = pd.DataFrame({'go':GObottomgo,'root':root,'termsize':termsize,'genesize':genesize,'bottomsize':GObottomsize})
	dataframe.to_csv(OutputDir+'bottomGO_infor'+'.csv',index=False,sep=',')
	AllGOID=list()
	AllGODepth=list()
	AllGOGene=list()
	AllGOChildren=list()
	AllGOParents=list()
	AllGOName=list()
	AllGOType=list()
	AllGOGenesize=list()
	for ns in filteredTree.keys():
		for depth in filteredTree[ns].keys():
			for go in filteredTree[ns][depth].keys():
				AllGOID.append(go)
				AllGODepth.append(str(GO[go].depth))
				AllGOGene.append(st.join(list(filteredTree[ns][depth][go]['gene'])))
				AllGOChildren.append(st.join(list(filteredTree[ns][depth][go]['children'])))
				AllGOParents.append(st.join(list(filteredTree[ns][depth][go]['parents'])))
				AllGOName.append(GO[go].name)
				AllGOType.append(GO[go].namespace)
				genes=set()
				children=set()
				children.update(filteredTree[ns][depth][go]['children'])
				genes.update(filteredTree[ns][depth][go]['gene'])
				while len(children)>0:
					childrens=set()
					for childGO in children:
						godeep=GO[childGO].depth
						genes.update(filteredTree[ns][godeep][childGO]['gene'])
						childrens.update(filteredTree[ns][godeep][childGO]['children'])
					children=list(childrens)
				AllGOGenesize.append(len(genes))
	dataframe = pd.DataFrame({'Go':AllGOID,'depth':AllGODepth,'Gene':AllGOGene,'Children':AllGOChildren,'Parents':AllGOParents,'Name':AllGOName,'Type':AllGOType,'GeneSize':AllGOGenesize})
	dataframe.to_csv(OutputDir+'allGO_infor'+'.csv',index=False,sep=',')

def ProcessRawData(File_List,CellType_Name,remove_noise=False,noise_cutoff=16,adjusted_by_accuracy=True,log2transform=False):
	"""
	This function processes the output file from the "CellGOModeling" function and generates processed cell type-specific active scores of all gene-term pairs.
	Parameters:
		"File_List":A list of file names. If you run the "CellGOModeling" function with the same gene expression matrix and cell identities for multiple times and get multiple output files, you can use these files as input to increase reliability
		"CellType_Name":A list of cell type names. Names and orders of these cell types must be corresponding to cell identites
		"remove_noise":Whether to remove noise, default False
		"noise_cutoff":If a raw gene-term score is less than N (default 16), this score will be set to 0
		"adjusted_by_accuracy":Whether to adjust gene-term scores by classification accuracy, default True
		"log2transform":Whether to log2-transform gene-term scores, default False
	"""
	File_Num=len(File_List)
	Raw={}
	for i in range(File_Num):
		Raw[i]=pd.read_csv(File_List[i])
	combined=pd.concat(Raw,axis=0,ignore_index=True)
	File_Shape=combined.shape
	CellType_Num=len(CellType_Name)
	Processed={}
	for i in range(CellType_Num):
		Processed[CellType_Name[i]]={}
	if remove_noise == True:
		for j in range(CellType_Num):
			L=np.abs(combined.iloc[:,2+j]) <= noise_cutoff
			combined.loc[L,combined.columns[2+j]]=0
	if adjusted_by_accuracy == True:
		for j in range(CellType_Num):
			combined.iloc[:,2+j]=combined.iloc[:,2+j]*combined.iloc[:,-1]
	if log2transform == True:
		for j in range(CellType_Num):
			L=combined.iloc[:,2+j] >= 0
			combined.loc[L,combined.columns[2+j]]=np.log2(combined.loc[L,combined.columns[2+j]]+1)
			L=combined.iloc[:,2+j] < 0
			combined.loc[L,combined.columns[2+j]]=-np.log2(np.abs(combined.loc[L,combined.columns[2+j]])+1)
	for i in range(File_Shape[0]):
		GO=combined.iloc[i,0]
		GENE=combined.iloc[i,1]
		for j in range(CellType_Num):
			if GO not in Processed[CellType_Name[j]].keys():
				Processed[CellType_Name[j]][GO]={}
			if GENE not in Processed[CellType_Name[j]][GO].keys():
				Processed[CellType_Name[j]][GO][GENE]=[combined.iloc[i,2+j]]
			else:
				Processed[CellType_Name[j]][GO][GENE].append(combined.iloc[i,2+j])
	output={}
	for i in range(CellType_Num):
		GO_list=[]
		GENE_list=[]
		SCORE_list=[]
		for GO in Processed[CellType_Name[i]].keys():
			for GENE in Processed[CellType_Name[i]][GO].keys():
				GO_list.append(GO)
				GENE_list.append(GENE)
				SCORE_list.append(np.mean(np.array(Processed[CellType_Name[i]][GO][GENE])))
		subProcessed=pd.DataFrame({CellType_Name[i]:SCORE_list,'Gene':GENE_list,'Go':GO_list})
		subProcessed=subProcessed.sort_values(by=CellType_Name[i],ascending=[True])
		subProcessed=subProcessed.reset_index(drop=True)
		Normalized=np.array(subProcessed[CellType_Name[i]])
		Normalized[Normalized>0]=Normalized[Normalized>0]*np.sum(np.abs(Normalized[Normalized<0]))/np.sum(np.abs(Normalized[Normalized>0]))
		Normalized=Normalized/np.max(np.abs(Normalized))
		subProcessed[CellType_Name[i]]=Normalized
		output[i]=subProcessed
	colname=[]
	for i in range(CellType_Num):
		colname=colname+list(output[i].columns)
	out=pd.concat(output,axis=1)
	out.columns=colname
	return(out)

def load_ontology_file(topo_file):
    assert os.path.exists(topo_file), 'Can not find ontology file!'
    children_term_map = {}
    children_gene_map = {}
    term_size_map = {}
    term_list = []
    gene_list = []
    rfile = open(topo_file, 'r')
    for line in rfile.readlines():
        if line.startswith('ROOT'):
            terms = line.strip().split()
            term_name = terms[1]
            term_list.append(term_name)
            term_size_map[term_name] = int(terms[2])
        if line.startswith('GENES'):
            terms = line.strip().split()
            children_gene_list = terms[1:]
            gene_list += children_gene_list
            children_gene_map[term_name] = children_gene_list
        if line.startswith('TERMS'):
            terms = line.strip().split()
            children_term_list = terms[1:]
            children_term_map[term_name] = children_term_list
    term_list = list(set(term_list))
    gene_list = list(set(gene_list)) 
    rfile.close()
    return term_list, gene_list, children_term_map, children_gene_map, term_size_map

def cal_gene2id(gene_list):
    gene2id = {}
    for id,gene in enumerate(gene_list):
        gene2id[gene] = id
    return gene2id

def get_neuron_num(term_list,term_size_map):
    add_neuron_size_map = {}
    neuron_size_map = {}
    term_state = {}
    for i,term_name in enumerate(term_list):
        term_state[term_name] = 0
        neuron_size_map[term_name] = min(50,max(20,math.floor(0.3*term_size_map[term_name])))
        add_neuron_size_map[term_name] = 10
    return neuron_size_map,add_neuron_size_map,term_state

def filter_terms(term_state,children_term_map):
    result_list = []
    for term_name,state in zip(term_state.keys(), term_state.values()):        
        if state == 0:
            children_term_list = children_term_map[term_name]
            child_all_ready = True
            for i,child in enumerate(children_term_list):
                if term_state[child] == 0:
                    child_all_ready = False
                    break
            if child_all_ready == True:
                result_list.append(term_name)
    return result_list

class BranchNode(nn.Module):
    def __init__(self, feature_dim, hidden_num1, hidden_num2, out_num, type_num):
        super(BranchNode,self).__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(feature_dim, hidden_num1),
                nn.Tanh()
        )
        self.layer2 = nn.Sequential(
                    nn.Linear(hidden_num2, out_num),
                    nn.Tanh(),
                    nn.BatchNorm1d(out_num,eps=1e-5,momentum=0.1,affine=False)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(out_num,type_num),
            nn.Tanh()
        )
    def forward(self, x, previous_out = None):
        x = self.layer1(x)
        if previous_out != None:
            x = torch.cat((x, previous_out),1)
        else:
            x = x
        trans_out = self.layer2(x)
        predict_out = self.layer3(trans_out)
        return trans_out, predict_out

class BranchNode_NoGeneLayer(nn.Module):
    def __init__(self,hidden_num2, out_num,type_num):
        super(BranchNode_NoGeneLayer,self).__init__()
        self.layer1 = nn.Sequential(
                    nn.Linear(hidden_num2, out_num),
                    nn.Tanh(),
                    nn.BatchNorm1d(out_num,eps=1e-5,momentum=0.1,affine=False)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(out_num,type_num),
            nn.Tanh()
        )
    def forward(self,previous_out):
        trans_out = self.layer1(previous_out)
        predict_out = self.layer2(trans_out)
        return trans_out, predict_out

class RootNode(nn.Module):
    def __init__(self,hidden_num2, out_num,type_num):
        super(RootNode,self).__init__()
        self.layer1 = nn.Sequential(
                    nn.Linear(hidden_num2, out_num), 
                    nn.Tanh(),
                    nn.BatchNorm1d(out_num,eps=1e-5,momentum=0.1,affine=False),
                    nn.Linear(out_num,type_num),
                    nn.Tanh()
        )
    def forward(self,previous_out): 
        trans_out = None
        predict_out = self.layer1(previous_out)
        return trans_out, predict_out

def make_node_layers(term_state,children_term_map,children_gene_map,neuron_size_map,add_neuron_size_map,type_num,root):
    term_states = deepcopy(term_state)
    term_node_layers = {}
    ready_term = filter_terms(term_states,children_term_map)
    while len(ready_term) != 0:
        ready_term = filter_terms(term_states,children_term_map)
        print("New round selects",len(ready_term),"terms")
        for i,term_name in enumerate(ready_term):
            children_term_list = children_term_map[term_name]
            children_gene_list = children_gene_map[term_name]
            feature_num=len(children_gene_list)
            hidden_num1 = add_neuron_size_map[term_name]
            hidden_num2 = 0
            if len(children_term_list) > 0:
                for j,child in enumerate(children_term_list):
                    hidden_num2 = hidden_num2 + neuron_size_map[child]
                if len(children_gene_list) != 0:
                    hidden_num2 = hidden_num2 + hidden_num1
                else:
                    hidden_num2 = hidden_num2
            else:
                hidden_num2 = hidden_num1
            out_num = neuron_size_map[term_name]
            if term_name != root:
                if len(children_gene_list) > 0:
                    term_node_layers[term_name] = BranchNode(feature_num,hidden_num1,hidden_num2,out_num,type_num)
                else:
                    term_node_layers[term_name] = BranchNode_NoGeneLayer(hidden_num2,out_num,type_num)
            else:
                term_node_layers[term_name] = RootNode(hidden_num2,out_num,type_num)
            term_states[term_name] = 1
    return term_node_layers

class CombineNode(nn.Module):
    def __init__(self,term_node_layers,term_state,children_term_map,children_gene_map):
        super(CombineNode,self).__init__()
        self.term_node_layers = term_node_layers
        self.children_term_map = children_term_map
        self.children_gene_map = children_gene_map
        self.term_state = term_state
    def forward(self, x): 
        trans_out = {}
        predict_out = {}
        term_states = deepcopy(self.term_state)
        ready_term = filter_terms(term_states, self.children_term_map)
        while len(ready_term) != 0:
            ready_term = filter_terms(term_states, self.children_term_map)
            for i,term_name in enumerate(ready_term):
                children_term_list = self.children_term_map[term_name]
                children_gene_list = self.children_gene_map[term_name]
                node_layer = self.term_node_layers[term_name]
                previous_out_list = []
                if len(children_term_list) > 0:
                    for j,child in enumerate(children_term_list):
                        previous_out_list.append(trans_out[child])
                    previous_out = torch.cat(previous_out_list, 1)
                    if len(children_gene_list) > 0: 
                        trans_out[term_name], predict_out[term_name] = node_layer(x[term_name],previous_out)
                    else:
                        trans_out[term_name], predict_out[term_name] = node_layer(previous_out)
                else:
                    previous_out=None
                    trans_out[term_name], predict_out[term_name] = node_layer(x[term_name],previous_out)
                term_states[term_name] = 1
        return predict_out

class LossCompute():
    def __init__(self,root_term,relative_weight):
        self.root_term = root_term
        self.relative_weight=relative_weight
    def finalLoss(self, outputs, labels):
        CEL = nn.CrossEntropyLoss()
        Loss_list = []
        for term_name,predict in zip(outputs.keys(),outputs.values()):
            if term_name == self.root_term:
                Loss_list.append(CEL(predict, labels))
            else:
                Loss_list.append(CEL(predict, labels)*self.relative_weight)
        return sum(Loss_list)

def get_NN_input(inputs,term_list,children_gene_map,gene2id,Use_Cuda):
    NN_input={}
    for i,term_name in enumerate(term_list):
        children_gene_list = children_gene_map[term_name]
        if len(children_gene_list) >0:
            Select=[]
            for j in range(len(children_gene_list)):
                Select.append(gene2id[children_gene_list[j]])
            if Use_Cuda == True:
                Select=torch.LongTensor(Select).cuda()
            else:
                Select=torch.LongTensor(Select)
            NN_input[term_name]=torch.index_select(inputs,1,Select)
    return NN_input

def get_NN_KO_input(inputs,term_list,children_gene_map,gene2id,start,startgeneid,Use_Cuda):
    NN_input={}
    for i,term_name in enumerate(term_list):
        children_gene_list = children_gene_map[term_name]
        if len(children_gene_list) >0:
            Select=[]
            for j in range(len(children_gene_list)):
                 Select.append(gene2id[children_gene_list[j]])
            if Use_Cuda == True:
                Select=torch.LongTensor(Select).cuda()
            else:
                Select=torch.LongTensor(Select)
            NN_input[term_name]=torch.index_select(inputs,1,Select)
    NN_input[start][:,startgeneid]=0
    return NN_input
