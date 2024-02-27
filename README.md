# MotifMDA
This is a public code for predicting miRNA-disease associations

## run MotifMDA
 - python main.py

## Options
See help for the other available options to use with *MotifMDA*
  - python main.py --help

## Experimental Results' directory
Contain the prediction results of each baseline models is training on HMDD V2.0 and miR2Disease
Please refer the code of ABMDA [here](https:// github.com/githubcode007/ABMDA/); please refer the code of NIMCGCN [here](https://github.com/ljatynu/NIMCGCN/); please refer the code of MINIMDA [here](https://github.com/chengxu123/MINIMDA/), please refer the code of SAEMDA [here](https://github.com/xpnbs/SAEMDA/).
For the sake of conducting a fair comparison, we initially obtain the source codes of the comparison methods from the repositories provided in their original works. This allows us to run the standalone version of these comparison methods on the same machine, which is equipped with an 18-core Intel Xeon 6154 Gold processor and 128GB of RAM. Concerning the parameter settings of the comparison methods, it is explicitly mentioned that we adopt the parameter values recommended in their original works for each method. The performance of these comparison methods is then evaluated under the same 5-fold cross-validation setup as MotifMDA to ensure consistency and avoid potential biases in the experimental scenarios. Furthermore, in our project's repository on GitHub, we not only furnish a comprehensive description of the parameter settings for each comparison method but also upload all experimental results. This is aimed at facilitating the reproducibility of our work and providing transparency in the evaluation process.
