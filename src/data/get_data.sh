## prepare data
mkdir data && cd data

# download Lyft Dataset
# train + val + aerial map + semantic map 
# wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/train.tar  # training set part 1/2 (8.4G)
# wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/train_full.tar  # training set part 2/2 (70G)
# wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/validate.tar 
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/aerial_map.tar
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/semantic_map.tar

# tar -xvf train.tar
# tar -xvf train_full.tar
# tar -xvf validate.tar
tar -xvf aerial_map.tar
tar semantic_map.tar


