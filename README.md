## **CellGO: A novel deep learning-based framework and webserver for cell type-specific gene function interpretation**

## **What is CellGO?**

CellGO is a method designed to discover cell type-specific active pathways of a single gene or a gene set by modeling the propagation of cell type-specific signals through the hierarchical structure of the Gene Ontology tree.

## **What can CellGO do?**

1)CellGO is able to predict cell type-specific pathways activated by a single gene.

2)CellGO is able to generate a four-level hierarchy of biological insights for a gene set:

1.CellGO can select cell type-specific active pathways.<br>
2.CellGO can identify the cell type most activated by the enquiry gene set.<br>
3.CellGO can construct the network of cell type-specific active pathways and report top communities enriched with active pathways.<br>
4.CellGO can screen essential genes responsible for pathway activation.<br>

#### **For more information about CellGO, please refer to our preprint version: https://doi.org/10.1101/2023.08.02.551654**

#### **For interactive web-based CellGO analysis, please login: http://www.cellgo.world**

## **Installation and dependencies**

### **1) System requirements**
#### **Requirements for software dependencies:**
Python3 (tested versions: v3.6.13-v3.9.15)<br>
Python3 packages:<br>
numpy (tested versions: v1.19.5-v1.24.0)<br>
pandas (tested versions: v1.1.5-v1.5.2)<br>
goatools (tested versions: v1.0.15-v1.2.4)<br>
torch (tested versions: v1.9.0-v1.13.1)<br>
seaborn (tested versions: v0.11.1-v0.12.1)<br>
scipy (tested versions: v1.5.4-v1.9.3)<br>
statsmodels (tested versions: v0.12.2-v0.13.5)<br>
scikit-learn (tested versions: v0.24.2-v1.2.0)<br>
plotnine (tested versions: v0.8.0-v0.10.1)<br>
matplotlib (tested versions: v3.3.4-v3.6.2)<br>
graphviz (tested versions: v0.19.1-v0.20.1)<br>
graph-tool (tested versions: v2.43-v2.45)<br>

#### **Requirements for hardware:**
RAM: 8+ GB<br>
CPU: 2+ cores, 3.0+ GHz/core<br>
(optional) GPU specifications are equal to or higher than NVIDIA Geforce GTX 1080<br>

### **2) Installation guide**

CellGO has been implemented in Python3 and can be installed using git clone to download this git repository file:

git clone https://github.com/FduZhuLab/cellgopy.git<br>

CellGO depends on a number of Python3 packages, and these dependencies can be installed using the Conda package manager:

conda create -n cellgo<br>
conda activate cellgo<br>
conda install -c conda-forge graph-tool #(https://graph-tool.skewed.de/)<br>
pip install numpy #(https://pypi.org/project/numpy/)<br>
pip install pandas #(https://pypi.org/project/pandas/)<br>
pip install goatools #(https://pypi.org/project/goatools/)<br>
pip install torch #(https://pypi.org/project/torch/)<br>
pip install seaborn #(https://pypi.org/project/seaborn/)<br>
pip install scipy #(https://pypi.org/project/scipy/)<br>
pip install statsmodels #(https://pypi.org/project/statsmodels/)<br>
pip install scikit-learn #(https://pypi.org/project/scikit-learn/)<br>
pip install plotnine #(https://pypi.org/project/plotnine/)<br>
pip install matplotlib #(https://pypi.org/project/matplotlib/)<br>
pip install graphviz #(https://pypi.org/project/graphviz/)<br>

Maybe the simplest way to install CellGO is using the Conda package manager:

conda install -c conda-forge cellgopy<br>

## **The tutorial for the python package cellgopy**

Please refer to "The tutorial for the modeling phase.py" and "The tutorial for the analysis phase.py"

**The related files involved in tutorials are available at: http://www.cellgo.world**
