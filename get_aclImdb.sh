#! /bin/bash
dir=./data/aclImdb
if [[ ! -d "$dir" ]]; then
	echo "Downloading aclImdb."
   	wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    	tar zxf aclImdb_v1.tar.gz --directory ./data/ 
    	rm -f aclImdb_v1.tar.gz
    	echo "aclImdb downloaded and extracted into $dir."  
else
    	echo "aclImdb already downloaded."
fi