# Use this code to download dataset, and use local machine (linux) Thanks to this, dirs cut with 2 times, and data folder saved on the desired path
# CATH4.2 dataset 
!wget -r -np -nH --cut-dirs=2 --reject "index.html*" http://people.csail.mit.edu/ingraham/graph-protein-design/data/
