#! /bin/bash

conda activate twc 

for level in  'medium' 'hard'
do 
	for split in 'train' 'test' 'valid'
	do 
		tw-extract vocab --merge ./$level/$split/*.json
	done
done
