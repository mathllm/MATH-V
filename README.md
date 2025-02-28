# Measuring Multimodal Mathematical Reasoning with the MATH-Vision🔥 Dataset

![MathQA](https://img.shields.io/badge/Task-MathQA-red) 
![Mathematical Reasoning](https://img.shields.io/badge/Task-Mathematical_Reasoning-red) 
![Multimodal Reasoning](https://img.shields.io/badge/Task-Multi--Modal-red) 

![ChatGPT](https://img.shields.io/badge/Model-ChatGPT-green) 
![GPT-4](https://img.shields.io/badge/Model-GPT--4-green) 
![GPT-4V](https://img.shields.io/badge/Model-GPT--4V-green)
![GPT-4V](https://img.shields.io/badge/Model-GPT--4o-green)
![Gemini](https://img.shields.io/badge/Model-Gemini-green)

🌟  This is the official repository for the paper "[Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset](https://arxiv.org/pdf/2402.14804.pdf)", which contains both evaluation code and data for the **MATH-V** benchmark.

[[🌐 Homepage](https://mathllm.github.io/mathvision/)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/MathLLMs/MathVision)] [[📊 Leaderboard ](https://mathllm.github.io/mathvision/#leaderboard )] [[🔍 Visualization](https://mathllm.github.io/mathvision/#visualization)] [[📖 ArXiv Paper](https://arxiv.org/pdf/2402.14804.pdf)]

## 💥 News
- **[2025.02.28]** 💥 **Doubao-1.5-pro 🥇 Sets New SOTA on MATH-V with 48.62%!** Read more on the [Doubao blog](https://team.doubao.com/zh/special/doubao_1_5_pro).
- **[2025.01.26]** 🚀 [Qwen2.5-VL-72B](http://qwenlm.github.io/blog/qwen2.5-vl/) achieves **38.1%**, establishing itself as the best-performing one in open-sourced models. 🎉 Congratulations!
- **[2025.01.22]** 💥 **Kimi k1.5  achieves new SOTA** on MATH-Vision with **38.6%**! Learn more at the [Kimi k1.5 report](https://arxiv.org/pdf/2501.12599).
- **[2024-09-27]** **MATH-V** is accepted by NeurIPS DB Track, 2024! 🎉🎉🎉
- **[2024-08-29]** 🔥🔥🔥 Qwen2-VL-72B achieves new open-sourced SOTA on MATH-Vision with 25.9! 🎉 Congratulations! Learn more at the [Qwen2-VL blog](https://qwenlm.github.io/blog/qwen2-vl/).
- **[2024-07-19]** [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) now supports **MATH-V**, utilizing LLMs for more accurate answer extraction!🔥🔥
- **[2024-05-19]** OpenAI's **GPT-4o** scores **30.39%** on **MATH-V**, considerable advancement in short time! 💥
- **[2024-03-01]** **InternVL-Chat-V1-2-Plus** achieves **16.97%**, establishing itself as the new best-performing open-sourced model. 🎉 Congratulations!
- **[2024-02-23]** Our dataset is now accessible at [huggingface](https://huggingface.co/datasets/MathLLMs/MathVision).
- **[2024-02-22]** The top-performing model, **GPT-4V** only scores **23.98%** on **MATH-V**, while human performance is around **70%**.
- **[2024-02-22]** Our paper is now accessible at [ArXiv Paper](https://arxiv.org/abs/2402.14804).

## 👀 Introduction

Recent advancements in Large Multimodal Models (LMMs) have shown promising results in mathematical reasoning within visual contexts, with models approaching human-level performance on existing benchmarks such as MathVista. However, we observe significant limitations in the diversity of questions and breadth of subjects covered by these benchmarks. To address this issue, we present the MATH-Vision (MATH-V) dataset, a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty, our dataset provides a comprehensive and diverse set of challenges for evaluating the mathematical reasoning abilities of LMMs.


<p align="center">
    <img src="assets/figures/figure1_new.png" width="100%"> <br>
  The accuracies of four prominent Large Multimodal Models (LMMs), random chance, and human
performance are evaluated on our proposed <b>MATH-Vision (MATH-V)</b> across 16 subjects.
</p>

Through extensive experimentation, we unveil a notable performance gap between current LMMs and human performance on MATH-V, underscoring the imperative for further advancements in LMMs.





You can refer to our [project homepage](https://mathllm.github.io/mathvision/) and [the paper](https://arxiv.org/pdf/2402.14804.pdf) for more details.

## 📐 Dataset Examples

Some examples of MATH-V on three subjects: analytic geometry, topology, and graph theory.

<details>
<summary>Analytic geometry</summary><p align="center">
    <img src="assets/examples/exam_analytic_geo.png" width="50%"> <br>
</p></details>

<details>
<summary>Topology</summary><p align="center">
    <img src="assets/examples/exam_topology.png" width="50%"> <br>
</p></details>

<details>
<summary>Graph Geometry</summary><p align="center">
    <img src="assets/examples/exam_graph.png" width="50%"> <br>
</p></details>

You can refer to the Appendix D.3 of [the paper](https://arxiv.org/pdf/2402.14804.pdf) for example images of 16 subjects.

## 🏆 Leaderboard

The leaderboard is available [here](https://mathllm.github.io/mathvision/#leaderboard).



## 📈 Evaluation

### Generating Outputs of Different Models

#### Gemini

`python models/Gemini.py --in_path ./data/test.jsonl --save_path ./Gemini.jsonl`

This will run the Gemini API and save the outputs to `./Gemini.jsonl` path. You can modify the system prompt, max tokens, etc. in the `benchmark_gemini` function.

#### GPT_with_caption

Generate image captions using GPT-4V:

`python models/GPT_with_caption.py --model gpt-4-vision-preview --in_path ./data/test.jsonl --save_path ./data/gpt4v-captions.jsonl`

Generate outputs using ChatGPT-3.5 or GPT-4 with image captions:

`python models/GPT_with_caption.py --model gpt-3.5-turbo-0125 (gpt-4-turbo-preview) --in_path ./data/test.jsonl --save_path ./gpt3.5_caption.jsonl (./gpt4_caption.jsonl)`



### Evaluation of Model Outputs

Once all the model outputs have been generated, execute the `python evaluation/evaluate.py` function to assess these outputs. This script will examine all outputs located in the `outputs/` directory, computing overall accuracy as well as accuracy for each subject and level.

You can refer to the Appendix E and F of [the paper](https://arxiv.org/pdf/2402.14804.pdf) for some evaluation results of the above models.

## 📝 Citation

If you find this benchmark useful in your research, please consider citing this BibTex:

```
@inproceedings{
    wang2024measuring,
    title={Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset},
    author={Ke Wang and Junting Pan and Weikang Shi and Zimu Lu and Houxing Ren and Aojun Zhou and Mingjie Zhan and Hongsheng Li},
    booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2024},
    url={https://openreview.net/forum?id=QWTCcxMpPA}
}
```

## 🧠 Related Work

- **[CSV🔥]** [Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification](https://wangk.org/publications/1_iclr2024_csv/)
- **[MathGenie]** [MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs](https://github.com/MathGenie/MathGenie)
- **[MathCoder🔥]** [MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning](https://github.com/mathllm/MathCoder)
- **[MathVerse]** [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](https://github.com/ZrrSkywalker/MathVerse)
- **[MathVista]** [MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts](https://github.com/lupantech/MathVista)
- **[SPHINX]** [The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal LLMs](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)
- **[SPHINX-X]** [Scaling Data and Parameters for a Family of Multi-modal Large Language Models](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)
