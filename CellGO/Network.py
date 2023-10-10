import sys
import os
import math
import copy
import numpy as np
import pandas as pd
import textwrap
import heapq
import matplotlib
from graph_tool.all import *
import graphviz as gz

def GetNetwork(CellType,GoType,NetworkPlotPrepare,PathwayInfor):
	"""
	This function generates a pathway network for a specific cell type and a specific pathway type and returns a "graph" object in the "graph-tools" package.
	Parameters:
		"CellType":A cell type in the modeling results
		"GoType":"biological_process" or "molecular_function"
		"NetworkPlotPrepare":Value of key "NetworkPlotPrepare" in the dictionary-like output of the function "CellGO"
		"PathwayInfor":Imported pathway information with class "pandas.core.frame.DataFrame"
	"""
	if CellType not in NetworkPlotPrepare.columns:
		raise ValueError("Error! The input cell type does not exist!")
	if GoType not in ['biological_process','molecular_function']:
		raise ValueError("Error! Please input one of the two pathway types: biological_process or molecular_function!")
	keep=NetworkPlotPrepare[NetworkPlotPrepare['Start']==True]
	keep=keep[keep['Type']==GoType]
	Bottom=list(set(list(keep.index)))
	if len(Bottom) == 0:
		raise ValueError('Error! The input gene list is not in the annotated gene set!')
	Go_parents={}
	Go_name={}
	for i in range(PathwayInfor.shape[0]):
		Go_name[PathwayInfor[PathwayInfor.columns[0]][i]] = PathwayInfor[PathwayInfor.columns[5]][i]
		if pd.isnull(PathwayInfor[PathwayInfor.columns[4]][i]) == False:
			Go_parents[PathwayInfor[PathwayInfor.columns[0]][i]] = set(PathwayInfor[PathwayInfor.columns[4]][i].split(' '))
	Go_parents['GO:0008150']=set()
	Go_parents['GO:0003674']=set()
	G=Graph(directed=True)
	Subtree_Go=set()
	for k in range(len(Bottom)):
		GoTerm = Bottom[k]
		DeepGO=set()
		DeepGO.add(GoTerm)
		while len(DeepGO)>0:
			GoParents=set()
			for Go in DeepGO:
				Subtree_Go.add(Go)
				GoParents.update(Go_parents[Go])
			DeepGO = GoParents
	Subtree_Go=list(Subtree_Go)
	for k in range(len(Subtree_Go)):
		G.add_vertex()
	Tmp_edge=[]
	for k in range(len(Bottom)):
		GoTerm = Bottom[k]
		DeepGO=set()
		DeepGO.add(GoTerm)
		while len(DeepGO)>0:
			GoParents=set()
			for Go in DeepGO:
				GoParents.update(Go_parents[Go])
				if len(Go_parents[Go]) >0:
					Len = len(Go_parents[Go])
					subParents = list(Go_parents[Go])
					for z in range(Len):
						if subParents[z]+Go not in Tmp_edge:
							G.add_edge(G.vertex(Subtree_Go.index(Go)),G.vertex(Subtree_Go.index(subParents[z])),add_missing=True)
						Tmp_edge.append(subParents[z]+Go)
			DeepGO = GoParents
	CellGO_score=np.array(NetworkPlotPrepare.loc[Subtree_Go,CellType],dtype=np.float64)
	GO_score=np.array(NetworkPlotPrepare.loc[Subtree_Go,'GO'],dtype=np.float64)
	vprop_score=G.new_vertex_property("double")
	vprop_GO=G.new_vertex_property("double")
	vprop_label=G.new_vertex_property("string")
	vprop_id=G.new_vertex_property("string")
	for z in range(len(CellGO_score)):
		vprop_score[G.vertex(z)]=CellGO_score[z]
		vprop_GO[G.vertex(z)]=GO_score[z]
		vprop_label[G.vertex(z)]=Go_name[Subtree_Go[z]]
		vprop_id[G.vertex(z)]=Subtree_Go[z]
	G.vp['score']=vprop_score
	G.vp['GO']=vprop_GO
	G.vp['label']=vprop_label
	G.vp['id']=vprop_id
	return(G)

def CommunityPartition(G,PathwayInfor,CellGO_pvalue_cutoff=0.01,GO_pvalue_cutoff=1,PlotPathwayNetwork=True,PlotCommunityNetwork=True,OutputFileName=['PathwayNetwork.pdf','CommunityNetwork.pdf'],Step=10,Epoch=25,P_Return=0.1,Proportion=1.1,Resolution=20,MC=100,NodeShape_Pnet='circle',NodeColormap_Pnet=matplotlib.pyplot.cm.tab20,NodeTransparent_Pnet=0.9,NodeSize_Pnet=12,ShowNodeText=True,EdgeColor_Pnet=[150,178,178],EdgeTransparent_Pnet=0.7,EdgeWidth_Pnet=0.1,NodeShape_Cnet='circle',NodeColor_Cnet=[[178,178,178],[0,0,178]],NodeTransparent_Cnet=0.9,NodeSizeScaler_Cnet=[8,25],EdgeColor_Cnet=[150,178,178],EdgeTransparent_Cnet=0.8,EdgeWidthScaler_Cnet=[1,5],adjust_aspect=True):
	"""
	This function uses random walk with restart (RWR) to obtain high-affinity pathways of seed pathways, and performs community partition of these high-affinity pathways according to Newman's modularity. 
	Parameters:
		"G":The output of the "GetNetwork" function
		"PathwayInfor":Imported pathway information with class "pandas.core.frame.DataFrame"
		"CellGO_pvalue_cutoff" and "GO_pvalue_cutoff":Pathways meeting the two P-value cut-offs will be seed pathways of RWR. If no pathway meets the P-value cut-offs, then 0 will be returned
		"PlotPathwayNetwork","PlotCommunityNetwork":Whether to draw the pathway network and community network, default True
		"OutputFileName":The output file names, default ["PathwayNetwork.pdf","CommunityNetwork.pdf"]
		"Step":The number of steps per travel of RWR, default 10
		"Epoch":The number of travels per seed pathway of RWR, default 25
		"P_Return":The probability of returning to the starting per step, default 0.1
		"Proportion":This parameter (default 1.1) determines the number of high-affinity pathways, defined as int("proportion" * "number of seed pathways")
		"Resolution":This parameter (default 20) determines the minimum number of communities, defined as math.ceil("number of seed pathways" / "Resolution")
		"MC":The number (default 100) of Monte Carlo (MC) community partitions
		"NodeShape_Pnet":The node shape of the pathway network, default "circle"
		"NodeColormap_Pnet":The node color map of the pathway network, default "matplotlib.pyplot.cm.tab20"
		"NodeTransparent_Pnet":The node transparency of the pathway network, default 0.9, and 0 means full transparency
		"NodeSize_Pnet":The node size of the pathway network, default 12
		"ShowNodeText":Whether to show community numbers, default True
		"EdgeColor_Pnet":The edge color of the pathway network, default [150,178,178]
		"EdgeTransparent_Pnet":The edge transparency of the pathway network, default 0.7
		"EdgeWidth_Pnet":The edge width of the pathway network, default 0.1
		"NodeShape_Cnet":The node shape of the community network, default "circle"
		"NodeColor_Cnet":The node color of the community network, default [[178,178,178],[0,0,178]]. It consists of two RGB vectors, the former RGB vector corresponds to the color when the community-level significance is 0, and the latter corresponds to the color when the community-level significance is 4
		"NodeTransparent_Cnet":The node transparency of the community network, default 0.9, and 0 means full transparency
		"NodeSizeScaler_Cnet":The node size of the community network, default [8,25]. The more pathways it contains, the larger the node
		"EdgeColor_Cnet":The edge color of the community network, default [150,178,178]
		"EdgeTransparent_Cnet":The edge transparency of the community network, default 0.8
		"EdgeWidthScaler_Cnet":The edge width of the community network, default [1,5]. The more connections between communities, the thicker the edge
		"adjust_aspect":If Ture the output size of the pathway network and the community network will be decreased in the width or height to remove empty spaces
	"""
	seed_nodes1=np.argwhere(G.vp.score.a >= -np.log10(CellGO_pvalue_cutoff))
	seed_nodes2=np.argwhere(G.vp.GO.a >= -np.log10(GO_pvalue_cutoff))
	seed_nodes=np.intersect1d(seed_nodes1,seed_nodes2)
	if len(seed_nodes) == 0:
		print("Warning! No pathway meets the P-value cut-offs and 0 is returned!")	
		return(0)
	Go_bottom=[]
	for i in range(PathwayInfor.shape[0]):
		if pd.isnull(PathwayInfor[PathwayInfor.columns[3]][i]) == True:
			Go_bottom.append(PathwayInfor[PathwayInfor.columns[0]][i])
	appear_times = np.zeros(G.num_vertices())
	for seed_id in range(len(seed_nodes)):
		seed=G.vertex(seed_nodes[seed_id])
		for epoch in range(Epoch):
			travel_now=seed
			for step in range(Step):
				appear_times[G.vertex_index[travel_now]]+=1
				neibor=list(travel_now.all_neighbours())
				neibor_index=set()
				for one_neibor in neibor:
					neibor_index.add(G.vertex_index[one_neibor])
				neibor_index=list(neibor_index)
				neibor_len=len(neibor_index)
				if np.random.rand(1)[0] > P_Return:
					travel_now=G.vertex(neibor_index[np.random.randint(neibor_len)])
				else:
					travel_now=seed
	Cut=heapq.nlargest(int(len(seed_nodes)*Proportion),appear_times)[-1]
	Keep_nodes=appear_times >= Cut
	Gf = GraphView(G, vfilt=Keep_nodes)
	Gf.purge_vertices()
	Gf.purge_edges()
	Gf.clear_filters()
	block_num=math.ceil(len(seed_nodes)/Resolution)
	bs = []
	for mc in range(MC):
		state = minimize_blockmodel_dl(Gf, state=ModularityState,multilevel_mcmc_args=dict(B_min=block_num))
		bs.append(state.b.a.copy())
	pmode = PartitionModeState(bs,converge=True)
	block = contiguous_map(pmode.get_max(Gf))
	if ShowNodeText==True:
		vertex_text=block
	else:
		vertex_text=None
	np.random.seed(111)
	seed_rng(111)
	if PlotPathwayNetwork==True:
		graph_draw(Gf,vertex_size=NodeSize_Pnet,vertex_text=vertex_text,vertex_fill_color=block,vcmap=(NodeColormap_Pnet,NodeTransparent_Pnet),vertex_shape=NodeShape_Pnet,edge_pen_width=EdgeWidth_Pnet,edge_color=[EdgeColor_Pnet[0]/255,EdgeColor_Pnet[1]/255,EdgeColor_Pnet[2]/255,EdgeTransparent_Pnet],vertex_pen_width=0,output=OutputFileName[0],adjust_aspect=adjust_aspect)
	Partition=pd.DataFrame({'Cluster':list(block),'GO_ID':list(Gf.vp.id),'GO_Name':list(Gf.vp.label),'(-Log10Pvalue_CellGO)':list(np.round(list(Gf.vp.score),3)),'(-Log10Pvalue_GO)':list(np.round(list(Gf.vp.GO),3))})
	C_leafnodesNum=[]
	C_pathwayNum=[]
	C_meanScore=[]
	C_meanScore_GO=[]
	C_id=[]
	C_Num=list(set(Partition['Cluster']))
	for C_Id in C_Num:
		C_Infor=Partition.loc[Partition['Cluster']==C_Id]
		C_id.append(C_Id)
		C_pathwayNum.append(C_Infor.shape[0])
		C_meanScore.append(np.round(np.mean(C_Infor['(-Log10Pvalue_CellGO)']),3))
		C_meanScore_GO.append(np.round(np.mean(C_Infor['(-Log10Pvalue_GO)']),3))
		C_leafnodesNum.append(len(list(set(C_Infor['GO_ID']).intersection(set(Go_bottom)))))
	Partition_rank = pd.DataFrame({'Cluster':C_id,'Mean(-Log10Pvalue_CellGO)':C_meanScore,'Mean(-Log10Pvalue_GO)':C_meanScore_GO,'PathwayNum':C_pathwayNum,'LeafnodeNum':C_leafnodesNum})
	Partition_rank = Partition_rank.sort_values(by=['Mean(-Log10Pvalue_CellGO)','LeafnodeNum'],ascending=[False,False])
	Partition=Partition.sort_values(by=['Cluster','(-Log10Pvalue_CellGO)'],ascending=[True,False])
	Partition=Partition.reset_index(drop=True)
	Partition_rank=Partition_rank.reset_index(drop=True)
	state = minimize_blockmodel_dl(Gf)
	state = state.copy(b=block)
	BG=state.get_bg()
	vprop_label=BG.new_vertex_property("string")
	vprop_shape=BG.new_vertex_property("string")
	vprop_color=BG.new_vertex_property("vector<double>")
	num1=((NodeColor_Cnet[0][0]-NodeColor_Cnet[1][0])/255)/-math.log10(0.0001)
	num2=((NodeColor_Cnet[0][1]-NodeColor_Cnet[1][1])/255)/-math.log10(0.0001)
	num3=((NodeColor_Cnet[0][2]-NodeColor_Cnet[1][2])/255)/-math.log10(0.0001)
	for z in range(state.get_nonempty_B()):
		vprop_label[BG.vertex(z)]=str(z)
		vprop_shape[BG.vertex(z)]=NodeShape_Cnet
		vprop_color[BG.vertex(z)]=[(NodeColor_Cnet[0][0]/255-C_meanScore[z]*num1),(NodeColor_Cnet[0][1]/255-C_meanScore[z]*num2),(NodeColor_Cnet[0][2]/255-C_meanScore[z]*num3),NodeTransparent_Cnet]
	if PlotCommunityNetwork==True:
		graph_draw(BG,vertex_text=vprop_label,vertex_fill_color=vprop_color,vertex_shape=vprop_shape,vertex_size=prop_to_size(state.wr,mi=NodeSizeScaler_Cnet[0],ma=NodeSizeScaler_Cnet[1]),edge_pen_width=prop_to_size(state.mrs,mi=EdgeWidthScaler_Cnet[0],ma=EdgeWidthScaler_Cnet[1]),edge_color=[EdgeColor_Cnet[0]/255,EdgeColor_Cnet[1]/255,EdgeColor_Cnet[2]/255,EdgeTransparent_Cnet],output=OutputFileName[1],adjust_aspect=adjust_aspect)
	out={}
	out['Partition_results']=Partition
	out['Partition_summary']=Partition_rank
	return(out)

def PlotsubNetwork(PartitionResults,ClusterID,PathwayInfor,OutputFileName='subNetwork.pdf',showLogPvalue=True,NodeShape=['egg','plaintext'],NodeColor=[[181,181,181],[58,95,205]],NodeFontsize='30',NodeFontcolor='white',NodePeripheries='2',GraphRankdir='BT'):
	"""
	This function is used to draw the pathway network within a community using the "graphviz" package.
	Parameters:
		"PartitionResults":Value of key "Partition_results" in the dictionary-like output of the function "CommunityPartition"
		"ClusterID":A community number in the output of the function "CommunityPartition". If "ALL" then it draws the pathway network of all communities
		"PathwayInfor":Imported pathway information with class "pandas.core.frame.DataFrame"
		"OutputFileName":The output file name, default "subNetwork.pdf"
		"showLogPvalue":Whether to show "-log10(P-value_CellGO)", default True
		"NodeShape":The shape of leaf-node-level pathways and non-leaf-node-level pathways, default ["egg","plaintext"]
		"NodeColor":The node color, default [[181,181,181],[58,95,205]]. It consists of two RGB vectors, the former RGB vector corresponds to the color when "-log10(P-value_CellGO)" is 0, and the latter corresponds to the color when "-log10(P-value_CellGO)" is 4
		"NodeFontsize":The font size, default "30"
		"NodeFontcolor":The font color, default "white"
		"NodePeripheries":The boundary number of nodes, default "2"
		"GraphRankdir":The direction of the network, default "BT"
	"""
	if ClusterID == 'ALL':
		Cluster_Infor=PartitionResults
	else:
		Cluster_Infor=PartitionResults.loc[PartitionResults['Cluster']==ClusterID]
	if Cluster_Infor.shape[0]==0:
		raise ValueError("The input community number dose not exist!")
	Go_bottom=[]
	Go_parents={}
	for i in range(PathwayInfor.shape[0]):
		if pd.isnull(PathwayInfor[PathwayInfor.columns[3]][i]) == True:
			Go_bottom.append(PathwayInfor[PathwayInfor.columns[0]][i])
		if pd.isnull(PathwayInfor[PathwayInfor.columns[4]][i]) == False:
			Go_parents[PathwayInfor[PathwayInfor.columns[0]][i]] = set(PathwayInfor[PathwayInfor.columns[4]][i].split(' '))
	Go_parents['GO:0008150']=set()
	Go_parents['GO:0003674']=set()
	dot=gz.Digraph(strict=True,filename=OutputFileName)
	dot.graph_attr['rankdir']=GraphRankdir
	dot.node_attr['peripheries']=NodePeripheries
	Top=list(['GO:0008150','GO:0003674'])
	Bottom=list(Cluster_Infor['GO_ID'])
	Name=list(Cluster_Infor['GO_Name'])
	Score=list(Cluster_Infor['(-Log10Pvalue_CellGO)'])
	Subtree_Go=Bottom
	Chara=[]
	for k in range(10000):
		Chara.append('A'+str(k))
	Effects_colors=[]
	Effects_bottom=[]
	num1=(NodeColor[0][0]-NodeColor[1][0])/-math.log10(0.0001)
	num2=(NodeColor[0][1]-NodeColor[1][1])/-math.log10(0.0001)
	num3=(NodeColor[0][2]-NodeColor[1][2])/-math.log10(0.0001)
	for z in range(len(Score)):
		if Subtree_Go[z] in Go_bottom:
			Effects_bottom.append(NodeShape[1])
		if Subtree_Go[z] not in Go_bottom:
			Effects_bottom.append(NodeShape[0])
		Effects_colors.append(color((int(NodeColor[0][0]-Score[z]*num1),int(NodeColor[0][1]-Score[z]*num2),int(NodeColor[0][2]-Score[z]*num3))))
	if showLogPvalue == True:
		for k in range(len(Subtree_Go)):
			dot.node(Chara[k],textwrap.fill(Name[k],width=30)+'\\n'+str(round(Score[k],2)),shape=Effects_bottom[k],color=Effects_colors[k],fontcolor=NodeFontcolor,style='filled',width='1.5',height='1.5',fontsize=NodeFontsize)
	else:
		for k in range(len(Subtree_Go)):
			dot.node(Chara[k],textwrap.fill(Name[k],width=30),shape=Effects_bottom[k],color=Effects_colors[k],fontcolor=NodeFontcolor,style='filled',width='1.5',height='1.5',fontsize=NodeFontsize)
	for k in range(len(Bottom)):
		GoTerm = Bottom[k]
		DeepGO=set()
		DeepGO.add(GoTerm)
		while len(DeepGO)>0:
			GoParents=set()
			for Go in DeepGO:
				GoParents.update(Go_parents[Go])
				if len(Go_parents[Go]) >0:
					Len = len(Go_parents[Go])
					subParents = list(Go_parents[Go])
					for z in range(Len):
						if Go in Subtree_Go and subParents[z] in Subtree_Go:
							dot.edge(Chara[Subtree_Go.index(Go)],Chara[Subtree_Go.index(subParents[z])])
			DeepGO = GoParents
	dot.view(cleanup=True, quiet=True, quiet_view=True)

def color(value):
	digit = list(map(str, range(10))) + list("ABCDEF")
	if isinstance(value, tuple):
		string = '#'
		for i in value:
			a1 = i // 16
			a2 = i % 16
			string += digit[a1] + digit[a2]
		return string
	elif isinstance(value, str):
		a1 = digit.index(value[1]) * 16 + digit.index(value[2])
		a2 = digit.index(value[3]) * 16 + digit.index(value[4])
		a3 = digit.index(value[5]) * 16 + digit.index(value[6])
		return (a1, a2, a3)
