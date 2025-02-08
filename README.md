## Seismic Traveltime Tomography With Label-Free Learning
Deep learning techniques have been used to build velocity models (VMs) for seismic traveltime tomography and have shown encouraging performance in recent years. However, they need to generate labeled samples (i.e., pairs of input and label) to train the deep neural network (NN) with end-to-end learning, and the real labels for field data inversion are usually missing or very expensive. Some traditional tomographic methods can be implemented quickly, but their effectiveness is often limited by prior assumptions. To avoid generating and/or collecting labeled samples, we propose a novel method by integrating deep learning and dictionary learning to enhance the VMs with low resolution by using the traditional tomography-least square method (LSQR). We first design a type of shallow and simple NN to reduce computational cost followed by proposing a two-step strategy to enhance the VMs with low resolution: 1) warming up: an initial dictionary is trained from the estimation by LSQR through the dictionary learning method; 2) dictionary optimization: the initial dictionary obtained in the warming-up step will be optimized by the NN, and then it will be used to reconstruct high-resolution VMs with the reference slowness and the estimation by LSQR. Furthermore, we design a loss function to minimize traveltime misfit to ensure that NN training is label-free, and the optimized dictionary can be obtained after each epoch of NN training. We demonstrate the effectiveness of the proposed method through the numerical tests on both synthetic and field data.
### Usage
```
Refer to the test_sd.ipynb and test_marmousi.ipynb.

```
### If you find this code helpful, please cite
```
@article{wang_seismic_2024,
	title = {Seismic {Traveltime} {Tomography} {With} {Label}-{Free} {Learning}},
	volume = {62},
	issn = {1558-0644},
	doi = {10.1109/TGRS.2024.3386783},
	urldate = {2024-04-18},
	journal = {IEEE Transactions on Geoscience and Remote Sensing},
	author = {Wang, Feng and Yang, Bo and Wang, Renfang and Qiu, Hong},
	year = {2024}
}

```
