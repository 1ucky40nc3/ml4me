# E2Style: Improve the Efficiency and Effectiveness of StyleGAN Inversion
---

## Resources

- 📃 [Paper](https://wty-ustc.github.io/inversion/paper/E2Style.pdf)
- 📚 [Project Page](https://wty-ustc.github.io/inversion)
- 🎬 [Examples](https://youtu.be/gJwFgdRHK0M)
- 💻 [Code](https://github.com/wty-ustc/e2style)

## Abstract

[Abstract](https://wty-ustc.github.io/inversion/paper/E2Style.pdf)—*This paper studies the problem of StyleGAN inversion, which plays an essential role in enabling the pretrained
StyleGAN to be used for real image editing tasks. The goal of
StyleGAN inversion is to find the exact latent code of the given
image in the latent space of StyleGAN. This problem has a high
demand for quality and efficiency. Existing optimization-based
methods can produce high-quality results, but the optimization
often takes a long time. On the contrary, forward-based methods
are usually faster but the quality of their results is inferior. In
this paper, we present a new feed-forward network “E2Style”
for StyleGAN inversion, with significant improvement in terms
of efficiency and effectiveness. In our inversion network, we
introduce: 1) a shallower backbone with multiple efficient heads
across scales; 2) multi-layer identity loss and multi-layer face
parsing loss to the loss function; and 3) multi-stage refinement.
Combining these designs together forms an effective and efficient method that exploits all benefits of optimization-based
and forward-based methods. Quantitative and qualitative results
show that our E2Style performs better than existing forwardbased methods and comparably to state-of-the-art optimizationbased methods while maintaining the high efficiency as well
as forward-based methods. Moreover, a number of real image
editing applications demonstrate the efficacy of our E2Style. Our
code is available at* https://github.com/wty-ustc/e2style

## Authors
Tianyi Wei<sup>1</sup>,
Dongdong Chen<sup>2</sup>,
Wenbo Zhou<sup>1</sup>,
Jing Liao<sup>3</sup>,
Weiming Zhang<sup>1</sup>, 
Lu Yuan<sup>2</sup>, 
Gang Hua<sup>4</sup>, 
Nenghai Yu<sup>1</sup> <br>
<sup>1</sup>*University of Science and Technology of China,*<br>
<sup>2</sup>*Microsoft Cloud AI*<br>
<sup>3</sup>*City University of Hong Kong,*<br>
<sup>4</sup>*Wormpex AI Research*

## Citation

### Plain Text

```
T. Wei et al., "E2Style: Improve the Efficiency and Effectiveness of StyleGAN Inversion," in IEEE Transactions on Image Processing, vol. 31, pp. 3267-3280, 2022, doi: 10.1109/TIP.2022.3167305.
```

### BibTex

```
@ARTICLE{9760266,
  author={Wei, Tianyi and Chen, Dongdong and Zhou, Wenbo and Liao, Jing and Zhang, Weiming and Yuan, Lu and Hua, Gang and Yu, Nenghai},
  journal={IEEE Transactions on Image Processing}, 
  title={E2Style: Improve the Efficiency and Effectiveness of StyleGAN Inversion}, 
  year={2022},
  volume={31},
  number={},
  pages={3267-3280},
  doi={10.1109/TIP.2022.3167305}}
```