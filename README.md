# 🥂 Awesome-Graphs-Meet-Agents

<div align="center">
<img src="fig/graph_meets_agent_toy.jpg" border="0" width=320px/>
</div>

<div align="center">
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg"/></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"/></a>
</div>

---

📚 This repository contains a curated list of related papers on graphs & agents, based on the survey paper "[**Graphs Meet AI Agents: Taxonomy, Progress, and Future Opportunities**](https://arxiv.org/pdf/2506.18019)".

🤗 Welcome to update this repository by launching an issue or a pull request if you find any missing resources or want to include some new resources.

- [🥂 Awesome-Graphs-Meet-Agents](#-awesome-graphs-meet-agents)
  - [🪴 Overall Description](#-overall-description)
  - [🚀 Taxonomy](#-taxonomy)
    - [Graph for Agent Planning](#graph-for-agent-planning)
      - [Task Reasoning](#task-reasoning)
      - [Task Decomposition](#task-decomposition)
      - [Task Decision Searching](#task-decision-searching)
    - [Graph for Agent Execution](#graph-for-agent-execution)
      - [Tool Usage](#tool-usage)
      - [Environment Interaction](#environment-interaction)
    - [Graph for Agent Memory](#graph-for-agent-memory)
      - [Memory Organization](#memory-organization)
      - [Memory Retrieval](#memory-retrieval)
      - [Memory Maintenance](#memory-maintenance)
    - [Graphs for Multi-Agent Coordination](#graphs-for-multi-agent-coordination)
      - [Coordination Message Passing](#coordination-message-passing)
      - [Coordination Topology Optimization](#coordination-topology-optimization)
    - [Agents for Graph Learning](#agents-for-graph-learning)
      - [Graph Annotation and Synthesis](#graph-annotation-and-synthesis)
      - [Graph Understanding](#graph-understanding)
  - [💻 Benchmarks and Open-Source Toolkits](#-benchmarks-and-open-source-toolkits)
    - [General](#general)
    - [Graph-Related](#graph-related)
  - [📃 Citation](#-citation)


---
## 🪴 Overall Description
<div align="center">
<img src="fig/fig_overall_git.jpg" border="0" width=800px/>
</div>


---
## 🚀 Taxonomy

### Graph for Agent Planning
[[Back to Top](#-awesome-graphs-meet-agents)]
#### Task Reasoning

***Knowledge Graph-Auxiliary Reasoning***

- (NAACL 2021) QA-GNN: Reasoning with language models and knowledge graphs for question answering [[Paper]](https://arxiv.org/pdf/2104.06378) [[Code]](https://github.com/michiyasunaga/qagnn)

- (ICLR 2024) Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph [[Paper]](https://arxiv.org/pdf/2307.07697) [[Code]](https://github.com/IDEA-FinAI/ToG)

- (ICLR 2024) Reasoning on graphs: Faithful and interpretable large language model reasoning [[Paper]](https://arxiv.org/pdf/2310.01061) [[Code]](https://github.com/RManLuo/reasoning-on-graphs)

- (IJCAI 2024) Kg-cot: Chain-of-thought prompting of large language models over knowledge graphs for knowledge-aware question answering [[Paper]](https://www.ijcai.org/proceedings/2024/0734.pdf) 

- (ACL 2024) Mindmap: Knowledge graph prompting sparks graph of thoughts in large language models [[Paper]](https://arxiv.org/pdf/2308.09729) [[Code]](https://github.com/wyl-willing/MindMap)

- (WWW 2025) Paths-over-graph: Knowledge graph empowered large language model reasoning [[Paper]](https://arxiv.org/pdf/2410.14211?) 

- (ICML 2025) GIVE: Structured Reasoning of Large Language Models with Knowledge-Graph-Inspired Veracity Extrapolation [[Paper]](https://arxiv.org/pdf/2410.08475) [[Code]](https://github.com/Jason-Tree/GIVE)

- (Arxiv 2025) Youtu-GraphRAG: Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning [[Paper]](https://arxiv.org/pdf/2508.19855)

***Structure-Organized Reasoning***

- (NeurIPS 2023) Tree of thoughts: Deliberate problem solving with large language models [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf) [[Code]](https://github.com/princeton-nlp/tree-of-thought-llm)

- (NAACL 2024) GoT: Effective Graph-of-Thought Reasoning in Language Models [[Paper]](https://aclanthology.org/2024.findings-naacl.183.pdf) [[Code]](https://github.com/Zoeyyao27/Graph-of-Thought)

- (AAAI 2024) Graph of thoughts: Solving elaborate problems with large language models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29720/31236) [[Code]](https://github.com/spcl/graph-of-thoughts)

- (AAAI 2025) Ratt: A thought structure for coherent and correct llm reasoning [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34876/37031) [[Code]](https://github.com/jinghanzhang1998/RATT)

- (Arxiv 2025) Reasoning with Graphs: Structuring Implicit Knowledge to Enhance LLMs Reasoning [[Paper]](https://arxiv.org/pdf/2501.07845)

#### Task Decomposition
- (CoLM 2024) Agentkit: Structured LLM reasoning with dynamic graphs [[Paper]](https://arxiv.org/pdf/2404.11483) [[Code]](https://github.com/holmeswww/AgentKit)

- (TMLR 2024) Feudal Graph Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2304.05099)

- (NeurIPS 2024)  Can Graph Learning Improve Planning in LLM-based Agents? [[Paper]](https://openreview.net/pdf?id=bmoS6Ggw4j) [[Code]](https://github.com/WxxShirley/GNN4TaskPlan)

- (ACL 2024) Villageragent: A graph-based multi-agent framework for coordinating complex task dependencies in minecraft [[Paper]](https://arxiv.org/pdf/2406.05720) [[Code]](https://github.com/cnsdqd-dyb/VillagerAgent-Minecraft-multiagent-framework)

- (Arxiv 2024) DAG-Plan: Generating Directed Acyclic Dependency Graphs for Dual-Arm Cooperative Planning [[Paper]](https://arxiv.org/pdf/2406.09953?) [[Demo]](https://sites.google.com/view/dag-plan)

- (ICRA 2025) Enhancing Multi-Agent Systems via Reinforcement Learning with LLM-based Planner and Graph-based Policy [[Paper]](https://arxiv.org/pdf/2503.10049)

- (Arxiv 2025) DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems [[Paper]](https://arxiv.org/pdf/2503.07675?)

- (Arxiv 2025) Plan-over-Graph: Towards Parallelable LLM Agent Schedule [[Paper]](https://arxiv.org/pdf/2502.14563) [[Code]](https://github.com/zsq259/Plan-over-Graph)


#### Task Decision Searching

- (NeurIPS 2014) Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning [[Paper]](https://papers.nips.cc/paper_files/paper/2014/file/88bf0c64edabeeb913c378227beef8f9-Paper.pdf) 

- (AAAI 2018, Best Paper) Memory-augmented monte carlo tree search [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/download/11531/11390) 

- (ACML 2020) Monte-Carlo Graph Search: the Value of Merging Similar States [[Paper]](https://proceedings.mlr.press/v129/leurent20a/leurent20a.pdf) 

- (ICAPS 2021) Improving alphazero using monte-carlo graph search [[Paper]](https://ojs.aaai.org/index.php/ICAPS/article/download/15952/15763)

- (AI 2024) Evolving interpretable decision trees for reinforcement learning [[Paper]](https://www.sciencedirect.com/science/article/pii/S0004370223002035)

- (AAMAS 2024) Continuous monte carlo graph search [[Paper]](https://arxiv.org/pdf/2210.01426) [[Code]](https://github.com/kallekku/cmcgs)

- (ICLR 2024) Promptagent: Strategic planning with language models enables expert-level prompt optimization [[Paper]](https://openreview.net/pdf?id=22pyNMuIoa) [[Code]](https://github.com/XinyuanWangCS/PromptAgent)

- (ICML 2024) Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models [[Paper]](https://openreview.net/forum?id=njwv9BsGHF) [[Code]](https://github.com/lapisrocks/LanguageAgentTreeSearch)

---

### Graph for Agent Execution
[[Back to Top](#-awesome-graphs-meet-agents)]
#### Tool Usage
- (ACL 2024) Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs [[Paper]](https://arxiv.org/pdf/2404.07103) [[Code]](https://github.com/PeterGriffinJin/Graph-CoT)

- (ECCV 2024) ControlLLM: Augment Language Models with Tools by Searching on Graphs [[Paper]](https://arxiv.org/pdf/2310.17796) [[Code]](https://github.com/OpenGVLab/ControlLLM)

- (ICML 2024) GPTSwarm: Language Agents as Optimizable Graphs [[Paper]](https://openreview.net/pdf?id=uTC9AFXIhg) [[Code]](https://github.com/metauto-ai/gptswarm)

- (Arxiv 2024) ToolNet: Connecting Large Language Models with Massive Tools via Tool Graph [[Paper]](https://arxiv.org/pdf/2403.00839)

- (Arxiv 2025) Graph RAG-Tool Fusion [[Paper]](https://arxiv.org/pdf/2502.07223) [[Code]](https://github.com/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS)

#### Environment Interaction
***Heuristic-Based Relationship***
- (ICRA 2023) Multiagent Reinforcement Learning for Autonomous Routing and Pickup Problem with Adaptation to Variable Demand [[Paper]](https://arxiv.org/pdf/2211.14983) 

- (AAMAS 2024) Towards Generalizability of Multi-Agent Reinforcement Learning in Graphs with Recurrent Message Passing [[Paper]](https://arxiv.org/pdf/2402.05027) [[Code]](https://github.com/jw3il/graph-marl)

- (Arxiv 2024) PlanAgent: A Multi-modal Large Language Agent for Closed-loop Vehicle Motion Planning [[Paper]](https://arxiv.org/pdf/2406.01587)

- (CVPR 2025) GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_GUI-Xplore_Empowering_Generalizable_GUI_Agents_with_One_Exploration_CVPR_2025_paper.pdf) [[Code]](https://github.com/921112343/GUI-Xplore)

- (Arxiv 2025) Multi-agent Auto-Bidding with Latent Graph Diffusion Models [[Paper]](https://arxiv.org/pdf/2503.05805)

- (Arxiv 2025) A Schema-Guided Reason-while-Retrieve framework for Reasoning on Scene Graphs with Large-Language-Models (LLMs) [[Paper]](https://arxiv.org/pdf/2502.03450)


***Learning-Based Relationship***
- (NeurIPS Workshop 2018) Deep Multi-Agent Reinforcement Learning with Relevance Graphs [[Paper]](https://arxiv.org/pdf/1811.12557) [[Code]](https://github.com/tegg89/magnet)

- (CoRL 2023) Learning Control Admissibility Models with Graph Neural Networks for Multi-Agent Navigation [[Paper]](https://proceedings.mlr.press/v205/yu23a/yu23a.pdf)

- (AAMAS 2023) TransfQMix: Transformers for Leveraging the Graph Structure of Multi-Agent Reinforcement Learning Problems [[Paper]](https://arxiv.org/pdf/2301.05334) [[Code]](https://github.com/mttga/pymarl_transformers)

- (TAI 2024) Reinforcement Learned Multi–Agent Cooperative Navigation in Hybrid Environment with Relational Graph Learning [[Paper]](https://ieeexplore.ieee.org/abstract/document/10636265/)


---

### Graph for Agent Memory
[[Back to Top](#-awesome-graphs-meet-agents)]
#### Memory Organization
- (Arxiv 2024) AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents [[Paper]](https://arxiv.org/pdf/2407.04363)

- (Arxiv 2024) On the Structural Memory of LLM Agents [[Paper]](https://arxiv.org/pdf/2412.15266) [[Code]](https://github.com/zengrh3/StructuralMemory)

- (Arxiv 2024) From Local to Global: A GraphRAG Approach to Query-Focused Summarization [[Paper]](https://arxiv.org/pdf/2404.16130) [[Code]](https://github.com/microsoft/graphrag)

- (Arxiv 2024) KG-Retriever: Efficient Knowledge Indexing for Retrieval-Augmented Large Language Models [[Paper]](https://arxiv.org/pdf/2412.05547) [[Code]](https://github.com/BAI-LAB/KG-Retriever)

- (SIGIR 2025) Enhancing the Patent Matching Capability of Large Language Models via the Memory Graph [[Paper]](https://arxiv.org/pdf/2504.14845) [[Code]](https://github.com/NEUIR/MemGraph)

- (AAAI 2025) LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning [[Paper]](](https://arxiv.org/pdf/2502.05453)) [[Code]](https://github.com/HappyEureka/mcrafter)

- (WWW 2025) Graphusion: A RAG Framework for Scientific Knowledge Graph Construction with a Global Perspective [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3701716.3717821) [[Code]](https://github.com/IreneZihuiLi/Graphusion)

- (Arxiv 2025) Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research [[Paper]](https://arxiv.org/pdf/2502.04644) [[Code]](https://github.com/theworldofagents/Agentic-Reasoning)

- (Arxiv 2025) G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems [[Paper]](https://arxiv.org/pdf/2506.07398) [[Code]](https://github.com/bingreeky/GMemory)

#### Memory Retrieval
- (NeurIPS 2024) G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/efaf1c9726648c8ba363a5c927440529-Paper-Conference.pdf) [[Code]](https://github.com/XiaoxinHe/G-Retriever)

- (Arxiv 2024) LightRAG: Simple and Fast Retrieval-Augmented Generation [[Paper]](https://arxiv.org/pdf/2410.05779) [[Code]](https://github.com/HKUDS/LightRAG)

- (ICLR 2025) Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation [[Paper]](https://openreview.net/pdf?id=JvkuZZ04O7) [[Code]](https://github.com/Graph-COM/SubgraphRAG)

- (NAACL 2025) GRAG: Graph Retrieval-Augmented Generation [[Paper]](https://aclanthology.org/2025.findings-naacl.232.pdf) [[Code]](https://github.com/HuieL/GRAG)

- (Arxiv 2025) GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation [[Paper]](https://arxiv.org/pdf/2502.01113) [[Code]](https://github.com/RManLuo/gfm-rag)

- (Arxiv 2025) PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths [[Paper]](https://arxiv.org/pdf/2502.14902?) [[Code]](https://github.com/BUPT-GAMMA/PathRAG)


#### Memory Maintenance
- (NeurIPS 2024) HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models [[Paper]](https://openreview.net/pdf?id=hkujvAPVsg) [[Code]](https://github.com/OSU-NLP-Group/HippoRAG)

- (Arxiv 2024) LightRAG: Simple and Fast Retrieval-Augmented Generation [[Paper]](https://arxiv.org/pdf/2410.05779) [[Code]](https://github.com/HKUDS/LightRAG)

- (Arxiv 2024) KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Graph [[Paper]](https://arxiv.org/pdf/2402.11163)

- (Arxiv 2024) AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents [[Paper]](https://arxiv.org/pdf/2407.04363)  [[Code]](https://github.com/AIRI-Institute/AriGraph)

- (AAAI 2025) LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning [[Paper]](](https://arxiv.org/pdf/2502.05453)) [[Code]](https://github.com/HappyEureka/mcrafter)

- (Arxiv 2025) Zep: A Temporal Knowledge Graph Architecture for Agent Memory [[Paper]](https://arxiv.org/pdf/2501.13956) [[Code]](https://github.com/getzep/graphiti)

- (Arxiv 2025) A-Mem: Agentic Memory for LLM Agents [[Paper]](https://arxiv.org/pdf/2502.12110) [[Code]](https://github.com/agiresearch/A-mem)

- (Arxiv 2025) InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning [[Paper]](https://arxiv.org/pdf/2504.13032)

---

### Graphs for Multi-Agent Coordination
[[Back to Top](#-awesome-graphs-meet-agents)]
#### Coordination Message Passing
***Task-Specific Relationship***
- (NeurIPS 2022) Learning NP-Hard Multi-Agent Assignment Planning using GNN: Inference on a Random Graph and Provable Auction-Fitted Q-learning [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/66ad22a4a1d2e6fe6f6f6581fadeedbc-Paper-Conference.pdf) 

- (ICRA 2025) Enhancing Multi-Agent Systems via Reinforcement Learning with LLM-based Planner and Graph-based Policy [[Paper]](https://arxiv.org/pdf/2503.10049)

- (ICLR 2025) Scaling Large Language Model-based Multi-Agent Collaboration [[Paper]](https://arxiv.org/pdf/2406.07155) [[Code]](https://github.com/OpenBMB/ChatDev/tree/macnet)

- (Arxiv 2025) DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems [[Paper]](https://arxiv.org/pdf/2503.07675)

- (Arxiv 2025) MAGNNET: Multi-Agent Graph Neural Network-based Efficient Task Allocation for Autonomous Vehicles with Deep Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2502.02311)

- (Arxiv 2025) GNNs as Predictors of Agentic Workflow Performances [[Paper]](https://arxiv.org/pdf/2503.11301) [[Code]](https://github.com/youngsoul0731/Flora-Bench)

***Environment-Specific Relationship***
- (ICASSP 2021) Graphcomm: A Graph Neural Network Based Method for Multi-Agent Reinforcement Learning [[Paper]](https://ieeexplore.ieee.org/document/9413716)

- (TITS 2022) Multi-Agent Trajectory Prediction with Heterogeneous Edge-Enhanced Graph Attention Network [[Paper]](https://ieeexplore.ieee.org/abstract/document/9700483)

- (TPAMI 2023) Robust Multi-Agent Communication With Graph Information Bottleneck Optimization [[Paper]](https://ieeexplore.ieee.org/abstract/document/10334015) 

- (IROS 2024) Transformer-based Multi-Agent Reinforcement Learning for Generalization of Heterogeneous Multi-Robot Cooperation [[Paper]](https://ieeexplore.ieee.org/abstract/document/10802580)

- (ICLR 2025) Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning [[Paper]](https://openreview.net/pdf?id=CL3U0GxFRD) [[Code]](https://github.com/LXXXXR/ExpoComm)

- (Arxiv 2025) AGENTSNET: Coordination and Collaborative Reasoning in Multi-Agent LLMs [[Paper]](https://arxiv.org/pdf/2507.08616) [[Code]](https://github.com/floriangroetschla/AgentsNet)

#### Coordination Topology Optimization
- (AAAI 2020) Multi-Agent Game Abstraction via Graph Attention Neural Network [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6211/6067)

- (AAMAS 2021) Deep Implicit Coordination Graphs for Multi-agent Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2006.11438) [[Code]](https://github.com/sisl/DICG)

- (AAMAS 2021) Multi-Agent Graph-Attention Communication and Teaming [[Paper]](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf) 

- (NeurIPS 2021) Learning Distilled Collaboration Graph for Multi-Agent Perception [[Paper]](https://proceedings.neurips.cc/paper/2021/file/f702defbc67edb455949f46babab0c18-Paper.pdf) [[Code]](https://github.com/ai4ce/DiscoNet)

- (TNNLS 2022) Online Multi-Agent Forecasting with Interpretable Collaborative Graph Neural Networks [[Paper]](https://arxiv.org/pdf/2107.00894)

- (AAMAS 2023) Learning Graph-Enhanced Commander-Executor for Multi-Agent Navigation [[Paper]](https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1652.pdf) [[Code]](https://github.com/yang-xy20/mage-x)

- (ICLR 2024) Learning Multi-Agent Communication from Graph Modeling Perspective [[Paper]](https://openreview.net/pdf?id=Qox9rO0kN0) [[Code]](https://github.com/charleshsc/CommFormer)

- (Arxiv 2024) G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks [[Paper]](https://arxiv.org/pdf/2410.11782) [[Code]](https://github.com/yanweiyue/GDesigner)

- (COLM 2024) A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration [[Paper]](https://openreview.net/pdf?id=XII0Wp1XA9) [[Code]](https://github.com/SALT-NLP/DyLAN)

- (ICML 2024) GPTSwarm: Language Agents as Optimizable Graphs [[Paper]](https://openreview.net/pdf?id=uTC9AFXIhg) [[Code]](https://github.com/metauto-ai/gptswarm)

- (ICRA 2025) Reliable and Efficient Multi-Agent Coordination via Graph Neural Network Variational Autoencoders [[Paper]](https://arxiv.org/pdf/2503.02954) [[Code]](https://github.com/mengyuest/gnn-vae-coord)

- (ICLR 2025) Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems [[Paper]](https://openreview.net/pdf?id=LkzuPorQ5L) [[Code]](https://github.com/yanweiyue/AgentPrune)

- (Arxiv 2025) Deep Meta Coordination Graphs for Multi-agent Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2502.04028) [[Code]](https://github.com/Nikunj-Gupta/dmcg-marl)

- (Arxiv 2025) Adaptive Graph Pruning for Multi-Agent Communication [[Paper]](https://arxiv.org/pdf/2506.02951?)

- (Arxiv 2025) Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation [[Paper]](https://www.arxiv.org/pdf/2507.18224) [[Code]](https://github.com/Shiy-Li/ARG-Designer)

---

### Agents for Graph Learning
[[Back to Top](#-awesome-graphs-meet-agents)]
#### Graph Annotation and Synthesis
- (NeurIPS 2020) Graph Policy Network for Transferable Active Learning on Graphs [[Paper]](https://proceedings.neurips.cc/paper/2020/file/73740ea85c4ec25f00f9acbd859f861d-Paper.pdf) [[Code]](https://github.com/ShengdingHu/GraphPolicyNetworkActiveLearning)

- (AAAI 2022) Batch Active Learning with Graph Neural Networks via Multi-Agent Deep Reinforcement Learning [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20897/20656)

- (Arxiv 2024) Exploring the Potential of Large Language Models in Graph Generation [[Paper]](https://arxiv.org/pdf/2403.14358) 

- (Arxiv 2024) LLM-Based Multi-Agent Systems are Scalable Graph Generative Models [[Paper]](https://arxiv.org/pdf/2410.09824) [[Code]](https://github.com/Ji-Cather/GraphAgent)

- (ICLR Workshop 2025) IGDA: Interactive Graph Discovery through Large Language Model Agents [[Paper]](https://openreview.net/pdf?id=cHV3Iw84AC)

- (Arxiv 2025) Plan-over-Graph: Towards Parallelable LLM Agent Schedule [[Paper]](https://arxiv.org/pdf/2502.14563) [[Code]](https://github.com/zsq259/Plan-over-Graph)

- (Arxiv 2025) GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments [[Paper]](https://arxiv.org/pdf/2504.00711)

#### Graph Understanding
- (KDD 2020) Policy-GNN: Aggregation Optimization
for Graph Neural Networks [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403088) 

- (WWW 2021) SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism [[Paper]](https://arxiv.org/pdf/2101.08170) [[Code]](https://github.com/RingBDStack/SUGAR)

- (NeurIPS 2023) MAG-GNN: Reinforcement Learning Boosted Graph Neural Network [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2788b4cdf421e03650868cc4184bfed8-Paper-Conference.pdf) [[Code]](https://github.com/LechengKong/MAG-GNN)

- (ICLR 2023) Agent-based Graph Neural Networks [[Paper]](https://openreview.net/pdf?id=8WTAh0tj2jC) [[Code]](https://github.com/KarolisMart/AgentNet)

- (Arxiv 2023) Graph Agent: Explicit Reasoning Agent for Graphs [[Paper]](https://arxiv.org/pdf/2310.16421) 

- (Arxiv 2023) A Versatile Graph Learning Approach through LLM-based Agent [[Paper]](https://arxiv.org/pdf/2309.04565) 

- (KDD 2024) GraphWiz: An Instruction-Following Language Model for Graph Computational Problems [[Paper]](https://dl.acm.org/doi/abs/10.1145/3637528.3672010) [[Code]](https://github.com/nuochenpku/Graph-Reasoning-LLM)

- (SIGIR 2024) GraphGPT: Graph Instruction Tuning for Large Language Models [[Paper]](https://arxiv.org/pdf/2310.13023) [[Code]](https://github.com/HKUDS/GraphGPT)

- (ICLR 2024) One For All: Towards Training One Graph Model For All Classification Tasks [[Paper]](https://openreview.net/pdf?id=4IT2pgc9v6) [[Code]](https://github.com/LechengKong/OneForAll)

- (ICML 2024) LLaGA: Large Language and Graph Assistant [[Paper]](https://arxiv.org/pdf/2402.08170) [[Code]](https://github.com/VITA-Group/LLaGA)

- (KDD 2024) ZeroG: Investigating Cross-dataset Zero-shot Transferability in Graphs [[Paper]](https://arxiv.org/pdf/2402.11235) [[Code]](https://github.com/NineAbyss/ZeroG)

- (Arxiv 2024) GraphAgent: Agentic Graph Language Assistant [[Paper]](https://arxiv.org/pdf/2412.17029) [[Code]](https://github.com/HKUDS/GraphAgent)

- (Arxiv 2024) Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents [[Paper]](https://arxiv.org/pdf/2410.05130)

- (Arxiv 2024) GraphTeam: Facilitating Large Language Model-based Graph Reasoning via Multi-Agent Collaboration [[Paper]](https://arxiv.org/pdf/2410.18032)

- (Arxiv 2024) GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability [[Paper]](https://arxiv.org/pdf/2403.04483) [[Code]](https://github.com/CGCL-codes/GraphInstruct)

- (AAAI 2025) Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34063/36218) 

---

## 💻 Benchmarks and Open-Source Toolkits

[[Back to Top](#-awesome-graphs-meet-agents)]
### General

- (NeurIPS 2021, RL Agent, Multi-Agent Coordination) Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks [[Paper]](https://openreview.net/pdf?id=cIrPX-Sn5n) [[Code]](https://github.com/uoe-agents/epymarl) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/uoe-agents/epymarl"/></a>

- (JMLR 2024, RL Agent, Multi-Agent Coordination) BenchMARL: Benchmarking Multi-Agent Reinforcement Learning [[Paper]](https://www.jmlr.org/papers/volume25/23-1612/23-1612.pdf) [[Code]](https://github.com/facebookresearch/BenchMARL) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/facebookresearch/BenchMARL"/></a>

- (JMLR 2025, RL Agent, Agent Memory) Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents [[Paper]](https://www.jmlr.org/papers/volume26/24-0043/24-0043.pdf) [[Code]](https://github.com/MarcoMeter/endless-memory-gym) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/MarcoMeter/endless-memory-gym"/></a>

- (EMNLP 2023, LLM Agent, Tool Usage) API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs [[Paper]](https://arxiv.org/pdf/2304.08244) [[Code]](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/AlibabaResearch/DAMO-ConvAI"/></a>

- (NeurIPS 2023, LLM Agent, Task Reasoning, Task Decomposition) PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/7a92bcdede88c7afd108072faf5485c8-Paper-Datasets_and_Benchmarks.pdf) [[Code]](https://github.com/karthikv792/LLMs-Planning) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/karthikv792/LLMs-Planning"/></a>

- (ICLR 2024, LLM Agent, General) AgentBench: Evaluating LLMs as Agents [[Paper]](https://arxiv.org/pdf/2308.03688) [[Code]](https://github.com/THUDM/AgentBench) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/THUDM/AgentBench"/></a>

- (ICLR 2024, LLM Agent, Tool Usage) ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs [[Paper]](https://arxiv.org/pdf/2307.16789) [[Code]](https://github.com/OpenBMB/ToolBench) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/OpenBMB/ToolBench"/></a>

- (NeurIPS 2024, LLM Agent, Tool Usage) GTA: A Benchmark for General Tool Agents [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/8a75ee6d4b2eb0b777f549a32a5a5c28-Paper-Datasets_and_Benchmarks_Track.pdf) [[Code]](https://github.com/open-compass/GTA) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/open-compass/GTA"/></a>


- (ICML 2024, LLM Agent, Task Reasoning, Task Decomposition, Tool Usage) TravelPlanner: A Benchmark for Real-World Planning with Language Agents [[Paper]](https://arxiv.org/pdf/2402.01622) [[Code]](https://github.com/OSU-NLP-Group/TravelPlanner) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/OSU-NLP-Group/TravelPlanner"/></a>

- (ICLR 2025, LLM Agent, Tool Usage, Agent-Environment Interaction) $\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains [[Paper]](https://arxiv.org/pdf/2406.12045) [[Code]](https://github.com/sierra-research/tau-bench) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/sierra-research/tau-bench"/></a>

- (NAACL 2025, LLM Agent, Tool Usage, Agent-Environment Interaction) ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities [[Paper]](https://arxiv.org/pdf/2408.04682) [[Code]](https://github.com/apple/ToolSandbox) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/apple/ToolSandbox"/></a>

- (Arxiv 2025, LLM Agent, Task Reasoning, Task Decomposition, Multi-Agent Coordination) REALM-Bench: A Real-World Planning Benchmark for LLMs and Multi-Agent Systems [[Paper]](https://arxiv.org/abs/2502.18836) [[Code]](https://github.com/genglongling/REALM-Bench) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/genglongling/REALM-Bench"/></a>

### Graph-Related

- (LLM Agent, Tool Usage, Multi-Agent Coordination) LangGraph [[Docs]](https://langchain-ai.github.io/langgraph/) [[Code]](https://github.com/langchain-ai/langgraph) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/langchain-ai/langgraph"/></a>

- (NeurIPS 2024, LLM Agent, Graph Modeling) GLBench: A Comprehensive Benchmark for Graph with Large Language Models [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ab0bd666d034fcaa5566fc7d176daa6-Paper-Datasets_and_Benchmarks_Track.pdf) [[Code]](https://github.com/NineAbyss/GLBench) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/NineAbyss/GLBench"/></a>

- (NeurIPS 2024, LLM Agent, Graph Modeling) Can Large Language Models Analyze Graphs like Professionals? A Benchmark, Datasets and Models [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/ff417c3993894694e88ffc4d3f53d28b-Paper-Datasets_and_Benchmarks_Track.pdf) [[Code]](https://github.com/BUPT-GAMMA/ProGraph) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/BUPT-GAMMA/ProGraph"/></a>

- (ICLR 2025, LLM Agent, Graph Modeling) GraphArena: Evaluating and Exploring Large Language Models on Graph Computation [[Paper]](https://openreview.net/pdf?id=Y1r9yCMzeA) [[Code]](https://github.com/squareRoot3/GraphArena)  <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/squareRoot3/GraphArena"/></a>

- (ICML 2024, LLM Agent, Task Reasoning, Tool Usage, Multi-Agent Coordination) GPTSwarm: Language Agents as Optimizable Graphs [[Paper]](https://openreview.net/pdf?id=uTC9AFXIhg) [[Code]](https://github.com/metauto-ai/gptswarm) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/metauto-ai/gptswarm"/></a>

- (ICLR 2025, LLM Agent, Multi-Agent Coordinatio) Scaling Large Language Model-based Multi-Agent Collaboration [[Paper]](https://arxiv.org/pdf/2406.07155) [[Code]](https://github.com/OpenBMB/ChatDev/tree/macnet) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/OpenBMB/ChatDev"/></a>

- (Arxiv 2025, LLM Agent, Tool Usage) Graph RAG-Tool Fusion [[Paper]](https://arxiv.org/pdf/2502.07223) [[Code]](https://github.com/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS"/></a>

- (Arxiv 2025, LLM Agent, Task Reasoning, Task Decomposition) GNNs as Predictors of Agentic Workflow Performances [[Paper]](https://arxiv.org/pdf/2503.11301) [[Code]](https://github.com/youngsoul0731/Flora-Bench) <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/youngsoul0731/Flora-Bench"/></a>
  
---

## 📃 Citation
```
@article{bei2025graphs,
  title={Graphs Meet AI Agents: Taxonomy, Progress, and Future Opportunities},
  author={Yuanchen Bei and Weizhi Zhang and Siwen Wang and Weizhi Chen and Sheng Zhou and Hao Chen and Yong Li and Jiajun Bu and Shirui Pan and Yizhou Yu and Irwin King and Fakhri Karray and Philip S. Yu},
  journal={arXiv preprint arXiv:2506.18019},
  year={2025}
}
```
