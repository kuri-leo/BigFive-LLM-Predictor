# Predicting the Big Five Personality Traits in Chinese Counselling Dialogues Using Large Language Models

This repository contains the code and instructions for the paper ["Predicting the Big Five Personality Traits in Chinese Counselling Dialogues Using Large Language Models"](https://arxiv.org/abs/2406.17287).

## Research Overview

Our research investigates whether LLMs can reason about personality traits from conversational text. The starting point is the BigFive prediction framework ([Yan et al., 2024](https://arxiv.org/abs/2406.17287)), which applies role-play and questionnaire-based prompting to predict Big Five traits from real-world counselling sessions.

During the early baseline experiments, we observed some interesting reasoning failures and traced them to two different bottlenecks:

1. **The Knowledge Dimension ("Knowing That").** Does the LLM encode sufficient domain knowledge but fail to recall it during multi-step reasoning? This investigation led to the knowledge-deconstruction mechanism in **SocraticAgent** (AAMAS 2026).

2. **The Method Dimension ("Knowing How").** Does the LLM generalize abstract computational rules, or rely on surface pattern matching? This led to our parallel work, **"Do Large Language Models Truly Grasp Addition?"** (EMNLP 2025).

In turn, insights from these two studies fed back into the applied psych-assessment setting:

3. **Calibration.** The knowledge-synthesis idea behind SocraticAgent was adapted to calibrate chain-of-thought reasoning in psych-assessment agents, producing **"Chain-of-Thought Calibration via Collaborative Knowledge Synthesis for Psych-Assessment Agents"** (CSCWD 2026).

4. **Questionnaire Design.** We also stepped back to question the assessment instrument itself — can LLMs collaborate with domain experts to design better psychological questionnaires? This led to **PsyScale** (CSCWD 2026).

## Publications

| Paper | Venue | Code |
|:------|:------|:-----|
| Predicting the Big Five Personality Traits in Chinese Counselling Dialogues Using Large Language Models | arXiv 2024 | this repo |
| SocraticAgent: An Autonomous Agent for Unlocking Latent Knowledge in LLMs | AAMAS 2026 | [kuri-leo/SocraticAgent](https://github.com/kuri-leo/SocraticAgent) |
| Do Large Language Models Truly Grasp Addition? A Rule-Focused Diagnostic Using Two-Integer Arithmetic | EMNLP 2025 | [kuri-leo/llm-arithmetic-diagnostic](https://github.com/kuri-leo/llm-arithmetic-diagnostic) |
| Chain-of-Thought Calibration via Collaborative Knowledge Synthesis for Psych-Assessment Agents | CSCWD 2026 | — |
| PsyScale: A Collaborative Human-AI Framework for Theory-Driven Psych Questionnaire Design | CSCWD 2026 | — |

## News

- [2026/02] Two papers accepted at CSCWD 2026: "Chain-of-Thought Calibration via Collaborative Knowledge Synthesis for Psych-Assessment Agents" and "PsyScale: A Collaborative Human-AI Framework for Theory-Driven Psych Questionnaire Design".
- [2025/12] "SocraticAgent: An Autonomous Agent for Unlocking Latent Knowledge in LLMs" accepted at AAMAS 2026.
- [2025/09] "Do Large Language Models Truly Grasp Addition? A Rule-Focused Diagnostic Using Two-Integer Arithmetic" accepted at EMNLP 2025.
- [2024/08] Released fine-tuned checkpoint [`Gemma-2-2b-it-BFI-Anonymous`](https://huggingface.co/kurileo/Gemma-2-2b-it-BFI-Anonymous), outperforming `Qwen1.5-110B-Chat` by 18.82% with only 1.8% of the parameters.

---

## BigFive-LLM-Predictor

Paper: [Predicting the Big Five Personality Traits in Chinese Counselling Dialogues Using Large Language Models](https://arxiv.org/abs/2406.17287)

### Abstract

Accurate assessment of personality traits is crucial for effective psycho-counseling, yet traditional methods like self-report questionnaires are time-consuming and biased. This study examines whether Large Language Models (LLMs) can predict the Big Five personality traits directly from counseling dialogues and introduces an innovative framework to perform the task. Our framework applies role-play and questionnaire-based prompting to condition LLMs on counseling sessions, simulating client responses to the Big Five Inventory. We evaluated our framework on 853 real-world counseling sessions, finding a significant correlation between LLM-predicted and actual Big Five traits, proving the framework's validity. Moreover, ablation studies highlight the importance of role-play simulations and task simplification via questionnaires in enhancing prediction accuracy. Our fine-tuned Llama3-8B model, utilizing Direct Preference Optimization with Supervised Fine-Tuning, achieves a 130.95% improvement, surpassing the state-of-the-art Qwen1.5-110B by 36.94% in personality prediction validity. In conclusion, LLMs can predict personality based on counseling dialogues. Our code and model are publicly available at [GitHub](https://github.com/kuri-leo/BigFive-LLM-Predictor), providing a valuable tool for future research in computational psychometrics.

### File Structure

- `generate_bfi_requests.py` - Generates Big Five Inventory (BFI) requests for the counseling dialogues following our role-play and questionnaire-based prompting framework.
- `process_results.py` - Processes the results of the LLM predictions and calculates the OCEAN scores.

_Note: The generated requests can be processed with [api_request_parallel_processor.py](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) from the OpenAI Cookbook._

### Data Schema

Each counseling dialogue session is stored in a separate TXT file. Each utterance has a prefix indicating the speaker, either "咨询师" (Counselor) or "来访者" (Client). The dialogues are in Chinese.

Each file is named using the format `{Client_ID}_chat_{Chatround_ID}_{Timestamp}.txt`, where:

- `Client_ID` is the unique identifier of the client.
- `Chatround_ID` is the number of chat rounds in the counseling session.
- `Timestamp` is the timestamp of the counseling session.

### Model Checkpoints

Currently, we provide the following model checkpoints:

- `Gemma-2-2b-it-BFI-Anonymous` - [Hugging Face Model Hub](https://huggingface.co/kurileo/Gemma-2-2b-it-BFI-Anonymous)
- `Llama-3-8b-BFI-Anonymous` - [Hugging Face Model Hub](https://huggingface.co/loeol/Llama-3-8b-BFI-Anonymous), Note: This checkpoint is only fine-tuned on 242 counseling dialogues.

*Other models will be publicly available once a dual-blind review is completed. You may also request the model checkpoints by contacting us :-)*

For each model, you can use `transformers` from Hugging Face or serving tools like [`vLLM`](https://github.com/vllm-project/vllm) or [`sglang`](https://github.com/sgl-project/sglang) for inference or hosting.

#### Benchmark Results

| Model                       | Open Mindedness | Conscientiousness | Extraversion | Agreeableness | Negative Emotionality |  Avg. |
| :-------------------------- | :-------------- | :---------------- | :----------- | :------------ | :-------------------- | ----: |
| Qwen1.5-110B-Chat           | 0.455\*\*\*     | 0.463\*\*\*       | 0.521\*\*\*  | 0.334\*\*     | 0.354\*\*             | 0.425 |
| gemma-2-2b-it               | 0.276\*         | 0.046             | 0.212        | 0.176         | -0.117                | 0.119 |
| Meta-Llama-3-8B-Instruct    | 0.177           | 0.434\*\*\*       | 0.233        | 0.111         | 0.303\*               | 0.252 |
| Gemma-2-2b-it-BFI-Anonymous | 0.588\*\*\*     | 0.492\*\*\*       | 0.541\*\*\*  | 0.402\*\*\*   | 0.501\*\*\*           | 0.505 |
| Llama-3-8b-BFI              | 0.692\*\*\*     | 0.554\*\*\*       | 0.569\*\*\*  | 0.448\*\*\*   | 0.648\*\*\*           | 0.582 |

\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001 for statistical significance.

For full benchmark results, please refer to the paper.

### Getting Started

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/kuri-leo/BigFive-LLM-Predictor.git
cd BigFive-LLM-Predictor
pip install -r requirements.txt
```

#### Generating BFI Requests

To generate the Big Five Inventory requests for your counseling dialogues, run:

```bash
python generate_bfi_requests.py --model_name MODEL_NAME --source_path path_to_your_dialogues --output_path path_to_output_requests
```

#### Processing Results

To process the results of the LLM predictions and calculate the OCEAN scores, run:

```bash
python process_results.py
```

Note: modify the `process_results.py` script to load the LLM predictions and define the output path for the OCEAN scores.

## Citation

If you find this work useful for your research, please consider citing the papers:

```bibtex
@article{yan2024predicting,
  title={Predicting the Big Five Personality Traits in Chinese Counselling Dialogues Using Large Language Models},
  author={Yan, Yang and Ma, Lizhi and Li, Anqi and Ma, Jingsong and Lan, Zhenzhong},
  journal={arXiv preprint arXiv:2406.17287},
  year={2024}
}

@inproceedings{yan2026socraticagent,
  title={SocraticAgent: An Autonomous Agent for Unlocking Latent Knowledge in LLMs},
  author={Yan, Yang and Lu, Yu and Xu, Renjun and Lan, Zhenzhong},
  booktitle={Proc. of the 25th International Conference on Autonomous Agents and Multiagent Systems},
  pages={1977--1985},
  year={2026}
}

@inproceedings{yan2025large,
  title={Do Large Language Models Truly Grasp Addition? A Rule-Focused Diagnostic Using Two-Integer Arithmetic},
  author={Yan, Yang and Lu, Yu and Xu, Renjun and Lan, Zhenzhong},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={13478--13494},
  year={2025}
}

@inproceedings{yan2026chain,
  title={Chain-of-Thought Calibration via Collaborative Knowledge Synthesis for Psych-Assessment Agents},
  author={Yan, Yang and Ma, Lizhi and Lu, Yu and Xu, Renjun and Lan, Zhenzhong},
  booktitle={Proceedings of the 2026 29th International Conference on Computer Supported Cooperative Work in Design (CSCWD)},
  year={2026}
}

@inproceedings{yan2026psyscale,
  title={PsyScale: A Collaborative Human-AI Framework for Theory-Driven Psych Questionnaire Design},
  author={Yan, Yang and Ma, Lizhi and Lu, Yu and Xu, Renjun and Lan, Zhenzhong},
  booktitle={Proceedings of the 2026 29th International Conference on Computer Supported Cooperative Work in Design (CSCWD)},
  year={2026}
}
```

We hope this repository will be helpful for your research. If you have any questions, please feel free to open an issue or contact us at [this email](mailto:yanyang@westlake.edu.cn).
