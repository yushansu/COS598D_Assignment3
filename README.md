# Neural Architecture Search (NAS)
In this assignment, you are required to use an open source AutoML toolkit (NNI) for neural architecture search. You are ask to compare three different NAS methods:
Efficient Neural Architecture Search via Parameter Sharing (ENAS)[1], DARTS: Differentiable Architecture Search[2], and Progressive DARTS (PDARTS)): Bridging the Optimization Gap for NAS in the Wild[3].

### References 
[1] Pham, H., Guan, M., Zoph, B., Le, Q. and Dean, J., 2018, July. Efficient neural architecture search via parameters sharing. In International Conference on Machine Learning (pp. 4095-4104). PMLR.

[2] Liu, H., Simonyan, K. and Yang, Y., 2018. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055.

[3] Chen, X., Xie, L., Wu, J. and Tian, Q., 2019. Progressive differentiable architecture search: Bridging the depth gap between search and evaluation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1294-1303).

# Your tasks
- [ ] Installing toolkit NNI following the [instruction](https://nni.readthedocs.io/en/stable/Tutorial/InstallationLinux.html#installation) 
- [ ] Perform CNN architecture search for Cifar10 (training and testing datasets are automatically split). You need to compare ENAS, DARTS, and PDARTS. Please record the top1 testing accuracy per GPU hours. You can present the results using a figure (referring to Figure 3 of [2]). Please use the same GPU types for comparison. Read the document about how to use NNI for NAS: [Document](https://nni.readthedocs.io/en/stable/nas.html).
- [ ] Show the final architecture for each method. ###Bonus###: check the visualization method on NNI, [document](https://nni.readthedocs.io/en/stable/NAS/Visualization.html?highlight=visualizationhttps://nni.readthedocs.io/en/stable/NAS/Visualization.html?highlight=visualization)
- [ ] You are required to submit a report to show the comparison results, discuss the results, and analyze the limitation of the three NAS methods.

# How to run?
