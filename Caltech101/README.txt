How to Run Code:


Step 1:

First we have to extract deep features.(Give path to dataset)
Caltech101.m/Caltech256.m generate Csv files of deep features.
If you want use direct use below link with lakehead id.
https://drive.google.com/drive/folders/1qy0QBeAUQxYvcmp1UZObOaiyKqmnoliA?usp=sharing
-- new_F_train.csv , train_labels.csv , new_F_test.csv , test_labels.csv --- For Caltech101

-- Cnew_F_train.csv , Ctrain_labels.csv , Cnew_F_test.csv , Ctest_labels.csv --- For Caltech256



Step 2:

To compile cuda files use below cammands.

- nvcc Caltech101ELM.cu MatAdd.cu MatMul.cu Matrixprint.cu RandomGPU.cu ReadCSV.cu Pinv.cu -lcublas -lcurand -o Caltech101ELM

- nvcc Caltech256ELM.cu MatAdd.cu MatMul.cu Matrixprint.cu RandomGPU.cu ReadCSV.cu Pinv.cu -lcublas -lcurand -o Caltech256ELM



Note :
I also used double precions.

I have 5048 features for each samples and I used 10000 Hidden Nodes  It will required Big Gpu memory I ran it into RTX 8000 48GB .

If code Not gives output You can try less no of hidden node But It might not give TOP-1 Accuracy.

ELM is working Proper Tu test I also put ELMtesting on small Dataset(iris)