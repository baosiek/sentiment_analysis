#! /bin/bash

# Directory vocab.txt
model_dir=./model/bert/assets
if [[ ! -d "$model_dir" ]]; then
    mkdir -p ./$model_dir
	echo "$model_dir created!."  
else
    echo "$model_dir already exists."
fi

# Downloads aclImdb_v1.tar.gz
dir=./data/
if [[ ! -d "$dir" ]]; then
    mkdir -p ./$dir
	echo "Downloading aclImdb."
   	wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar zxf aclImdb_v1.tar.gz --directory ./$dir 
    rm -f aclImdb_v1.tar.gz
    echo "aclImdb downloaded and extracted into $dir."  
else
    echo "aclImdb already downloaded."
fi

# Downloads Bert
bert_dir=./bert_uncased
if [[ ! -d "$bert_dir" ]]; then
    mkdir -p ./$bert_dir
	echo "Downloading bert_uncased_L-12_H-768_A-12_1."
    wget -O bert_uncased.tar.gz https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1?tf-hub-format=compressed
    tar zxf bert_uncased.tar.gz --directory ./$bert_dir 
    rm -f bert_uncased.tar.gz
    cp $bert_dir/assets/vocab.txt $model_dir/vocab.txt
    echo "Bert Uncased downloaded and extracted into $bert_dir."  
else
    echo "Bert Uncased already downloaded."
fi
