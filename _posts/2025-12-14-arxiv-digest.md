---
layout: post
title: "Daily arXiv digest (2025-12-14, UTC)"
---

Model: `mistralai/mistral-small-3.1-24b-instruct:free`

## Code Plagiarism

### Bin2Vec: Interpretable and Auditable Multi-View Binary Analysis for Code Plagiarism Detection

**Authors:** Moussa Moussaoui, Tarik Houichime, Abdelalim Sadiq  
**Published:** 2025-12-01  
**arXiv:** http://arxiv.org/abs/2512.02197v1  
**PDF:** http://arxiv.org/pdf/2512.02197v1.pdf

**Summary:**
- **Problem**: The challenge of detecting code plagiarism and comparing software programs in a clear, interpretable, and auditable manner.

- **Method**: Bin2Vec framework combines multiple views of software programs, including structural (functions, imports, exports) and behavioral (instructions, memory usage) information, to generate a comprehensive similarity score.

- **Data/Benchmarks**: The framework was tested on multiple versions of PuTTY and 7-Zip, two well-known Windows programs.

- **Key Results**: Bin2Vec successfully computed an optimal and visualization-friendly representation of the analyzed software. PuTTY versions exhibited more complex behavior and memory activity, while 7-Zip versions showed performance-related patterns.

- **Limitations**: The abstract does not mention any specific limitations of the Bin2Vec framework.

- **Why It Matters**: Bin2Vec provides reliable and explainable decisions, making it valuable for tasks such as auditing, verifying software origins, and screening large numbers of programs in cybersecurity and reverse-engineering.

- **Resources**: arXiv: http://arxiv.org/abs/2512.02197v1
PDF: http://arxiv.org/pdf/2512.02197v1.pdf

---

## Software Plagiarism

_No matching papers in this interval._

## Code Models

### UniCoR: Modality Collaboration for Robust Cross-Language Hybrid Code Retrieval

**Authors:** Yang Yang, Li Kuang, Jiakun Liu, Zhongxin Liu, Yingjie Xia, David Lo  
**Published:** 2025-12-11  
**arXiv:** http://arxiv.org/abs/2512.10452v1  
**PDF:** http://arxiv.org/pdf/2512.10452v1.pdf

**Summary:**
- **Problem**: Existing code retrieval methods struggle with hybrid queries (combining natural language and code snippets) and cross-language contexts, facing challenges in semantic understanding, modality fusion, and generalization across languages.

- **Method**: The authors propose UniCoR, a self-supervised framework designed to learn unified and robust code representations. It includes:
  - A multi-perspective supervised contrastive learning module to enhance semantic understanding and modality fusion.
  - A representation distribution consistency learning module to improve cross-language generalization by aligning feature distributions of different programming languages.

- **Data/Benchmarks**: The study uses both empirical benchmarks and large-scale benchmarks to evaluate the performance of UniCoR.

- **Key Results**: UniCoR outperforms all baseline models, achieving an average improvement of 8.64% in Mean Reciprocal Rank (MRR) and 11.54% in Mean Average Precision (MAP) over the best-performing baseline. It also shows stability in hybrid code retrieval and strong generalization in cross-language scenarios.

- **Limitations**: The abstract does not explicitly mention any limitations of the study or the UniCoR framework.

- **Why It Matters**: UniCoR addresses critical challenges in code retrieval, making it more effective in real-world scenarios where hybrid queries and cross-language contexts are common. This can significantly enhance developer productivity and code reuse.

- **Resources**: arXiv: http://arxiv.org/abs/2512.10452v1
- **Resources**: PDF: http://arxiv.org/pdf/2512.10452v1.pdf

---

### Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance

**Authors:** Ruihang Chu, Yefei He, Zhekai Chen, Shiwei Zhang, Xiaogang Xu, Bin Xia, Dingdong Wang, Hongwei Yi, Xihui Liu, Hengshuang Zhao, Yu Liu, Yingya Zhang, Yujiu Yang  
**Published:** 2025-12-09  
**arXiv:** http://arxiv.org/abs/2512.08765v1  
**PDF:** http://arxiv.org/pdf/2512.08765v1.pdf

**Summary:**
- **Problem**: Existing motion-controllable video generation methods often lack fine control granularity and scalability, making them impractical for high-quality video synthesis.

- **Method**: Wan-Move introduces a framework that makes condition features motion-aware to guide video synthesis. It uses dense point trajectories to represent object motions, allowing for precise control. These trajectories are projected into latent space, and the first frame's features are propagated along each trajectory to create an aligned spatiotemporal feature map.

- **Data/Benchmarks**: The authors designed MoveBench, a comprehensive benchmark with diverse content categories and high-quality motion annotations. It features larger data volumes, longer video durations, and hybrid-verified annotations. Experiments were also conducted on public datasets.

- **Key Results**: Wan-Move generates 5-second, 480p videos with superior motion controllability, rivaling commercial tools like Kling 1.5 Pro's Motion Brush. Extensive experiments on MoveBench and public datasets consistently demonstrate Wan-Move's superior motion quality.

- **Limitations**: The abstract does not specify any limitations of the method.

- **Why It Matters**: Wan-Move addresses the need for precise and scalable motion control in video generation, making it a significant advancement in the field. It integrates seamlessly with existing image-to-video models without requiring architecture changes, enhancing practical applicability.

- **Resources**: Code, models, and benchmark data are made publicly available.

---

### Understanding Privacy Risks in Code Models Through Training Dynamics: A Causal Approach

**Authors:** Hua Yang, Alejandro Velasco, Sen Fang, Bowen Xu, Denys Poshyvanyk  
**Published:** 2025-12-08  
**arXiv:** http://arxiv.org/abs/2512.07814v2  
**PDF:** http://arxiv.org/pdf/2512.07814v2.pdf

**Summary:**
- **Problem**: Large language models for code (LLM4Code) improve developer productivity but pose privacy risks by potentially leaking personally identifiable information (PII) from open-source repositories.

- **Method**: The study investigates whether different types of PII vary in their likelihood of being learned and leaked by LLM4Code, and whether this relationship is causal. The methodology includes fine-tuning representative models of different scales, computing training dynamics on real PII data, and formulating a structural causal model to estimate the causal effect of learnability on leakage.

- **Data/Benchmarks**: The researchers built a dataset with diverse PII types to analyze the training dynamics and leakage risks.

- **Key Results**: The results indicate that leakage risks differ significantly across PII types and correlate with their training dynamics. Easy-to-learn instances, such as IP addresses, exhibit higher leakage, while harder types, such as keys and passwords, leak less frequently. Ambiguous types show mixed behaviors.

- **Limitations**: The study does not provide details on the specific models used or the extent of the dataset, which may limit the generalizability of the findings.

- **Why It Matters**: This work provides the first causal evidence that leakage risks are type-dependent, offering guidance for developing type-aware and learnability-aware defenses for LLM4Code.

- **Resources**: arXiv: http://arxiv.org/abs/2512.07814v2
PDF: http://arxiv.org/pdf/2512.07814v2.pdf

---

### What Happens When: Learning Temporal Orders of Events in Videos

**Authors:** Daechul Ahn, Yura Choi, Hyeonbeom Choi, Seongwon Cho, San Kim, Jonghyun Choi  
**Published:** 2025-12-05  
**arXiv:** http://arxiv.org/abs/2512.08979v1  
**PDF:** http://arxiv.org/pdf/2512.08979v1.pdf

**Summary:**
- **Problem**: Video Large Multimodal Models (VLMMs) struggle to accurately capture the temporal order of events in videos, despite performing well on existing benchmarks even when video frames are scrambled.

- **Method**: The authors propose VECTOR, a benchmark designed to explicitly assess a model's ability to identify the temporal order of events in videos.

- **Data/Benchmarks**: The study uses VECTOR to evaluate various VLMMs and finds that they often fail to understand the correct sequence of events.

- **Proposed Solution**: To improve temporal understanding, the authors introduce MECOT (Multi-Event instruction fine-tuning with Chain-of-Thought). This method involves training models on detailed, event-by-event video descriptions and using chain-of-thought prompts during inference.

- **Key Results**: MECOT outperforms previous methods on the VECTOR benchmark and also improves performance on existing video benchmarks, demonstrating the effectiveness of enhanced temporal understanding.

- **Limitations**: The abstract does not discuss specific limitations of the study or the proposed methods.

- **Why It Matters**: This research highlights the importance of temporal understanding in video analysis and provides a new benchmark and method to improve the performance of VLMMs in capturing the sequence of events.

- **Resources**: The code, model, and datasets used in this study are released by the authors.

---

### LAMP: Language-Assisted Motion Planning for Controllable Video Generation

**Authors:** Muhammed Burak Kizil, Enes Sanli, Niloy J. Mitra, Erkut Erdem, Aykut Erdem, Duygu Ceylan  
**Published:** 2025-12-03  
**arXiv:** http://arxiv.org/abs/2512.03619v2  
**PDF:** http://arxiv.org/pdf/2512.03619v2.pdf

**Summary:**
- **Problem**: Existing video generation techniques struggle with motion control, particularly in specifying object dynamics and camera trajectories for complex, cinematic scenes.

- **Method**: The authors introduce LAMP, a framework that uses large language models (LLMs) as motion planners. LAMP translates natural language descriptions into explicit 3D trajectories for dynamic objects and cameras.

- **Data/Benchmarks**: A large-scale procedural dataset was created, pairing natural text descriptions with corresponding motion programs and 3D trajectories.

- **Key Results**: LAMP demonstrates improved performance in motion controllability and better alignment with user intent compared to state-of-the-art alternatives.

- **Why It Matters**: LAMP is the first framework to generate both object and camera motions directly from natural language specifications, enhancing the controllability and complexity of video generation.

- **Limitations**: The abstract does not specify any limitations of the LAMP framework.

- **Resources**: Code, models, and data are available on the project page. The arXiv link is http://arxiv.org/abs/2512.03619v2 and the PDF link is http://arxiv.org/pdf/2512.03619v2.pdf.

---

### OneThinker: All-in-one Reasoning Model for Image and Video

**Authors:** Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, Yan Feng, Peng Pei, Xunliang Cai, Xiangyu Yue  
**Published:** 2025-12-02  
**arXiv:** http://arxiv.org/abs/2512.03043v2  
**PDF:** http://arxiv.org/pdf/2512.03043v2.pdf

**Summary:**
- **Problem**: Current reinforcement learning (RL) approaches for multimodal large language models (MLLMs) train separate models for different tasks, treating image and video reasoning as distinct domains. This limits scalability and knowledge sharing across tasks and modalities.

- **Method**: The authors propose OneThinker, an all-in-one reasoning model that unifies image and video understanding across various fundamental visual tasks, including question answering, captioning, spatial and temporal grounding, tracking, and segmentation.

- **Data/Benchmarks**: The OneThinker-600k training corpus was constructed to cover all these tasks. The model was evaluated on 31 benchmarks across 10 fundamental visual understanding tasks.

- **Key Results**: OneThinker demonstrated strong performance across the benchmarks and showed effective knowledge transfer between certain tasks. It also exhibited preliminary zero-shot generalization ability.

- **Innovations**: The study introduces the OneThinker-SFT-340k for supervised fine-tuning (SFT) cold start and EMA-GRPO to handle reward heterogeneity in multi-task RL.

- **Limitations**: The abstract does not specify the limitations of the study, but potential areas for improvement could include the scalability of the model and the diversity of the training corpus.

- **Why It Matters**: OneThinker represents a significant step toward creating a unified multimodal reasoning generalist, which could enhance the versatility and practicality of visual reasoning models.

- **Resources**: Code, model, and data are released. The arXiv identifier is 2512.03043v2.

---

### CauSight: Learning to Supersense for Visual Causal Discovery

**Authors:** Yize Zhang, Meiqi Chen, Sirui Chen, Bo Peng, Yanxi Zhang, Tianyu Li, Chaochao Lu  
**Published:** 2025-12-01  
**arXiv:** http://arxiv.org/abs/2512.01827v1  
**PDF:** http://arxiv.org/pdf/2512.01827v1.pdf

**Resources:**
- https://github.com/OpenCausaLab/CauSight

**Summary:**
- **Problem**: The study addresses the challenge of enabling AI systems to understand cause-and-effect relationships in visual data, moving beyond mere perception of entities.

- **Method**: The authors introduce the task of visual causal discovery and develop a model called CauSight. The approach integrates three key components:
  - Training data curation from the Visual Causal Graph dataset (VCG-32K).
  - Tree-of-Causal-Thought (ToCT) for generating reasoning trajectories.
  - Reinforcement learning with a causal reward to refine the reasoning policy.

- **Data/Benchmarks**: The Visual Causal Graph dataset (VCG-32K) is constructed, consisting of over 32,000 images annotated with entity-level causal graphs.

- **Key Results**: Experiments demonstrate that CauSight outperforms GPT-4.1 on visual causal discovery tasks, achieving over a threefold performance improvement (21% absolute gain).

- **Why It Matters**: This research advances the field of AI by enabling models to understand and reason about causal relationships in visual data, which is crucial for applications requiring deeper understanding and decision-making.

- **Resources**: The code, model, and dataset are fully open-sourced at https://github.com/OpenCausaLab/CauSight.

---

## Code Pre-Trained Models

_No matching papers in this interval._

## Pre-Trained Code Models

_No matching papers in this interval._
