# Judge spot-check

10 randomly sampled judged answers (seed=0). Check each label by hand against the evidence; rules are in `eval/judge_prompt.md`.

## n005 (A, unanswerable)

**Q:** How do convolutional neural networks detect edges in images?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | no_claim | The indexed papers do not contain the answer to how convolutional neural networks detect edges in images. | The sentence states that the indexed papers do not contain the answer, which is not a factual claim about how convolutional neural networks detect edges in images. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2112.09118 §Related work p.2: In this section, we briefly review relevant work in information retrieval, and application of machine learning to this problem. This is not an exhaustive review, and we refer the reader to Manning et al. (2008), Mitra et al. (2018) and Lin et al. (2020) for a more complete introduction to the field. Term-frequency based information retrieval. Historically, in information retrieval, documents and queries are represented as sparse vectors where each element of the vectors corresponds to a term of the vocabulary. Different weighing schemes have been proposed, to determine how important a particul…
- **E2** arxiv:2104.08663 §Introduction p.2: This setting hinders for investigation of model generalization across diverse set of domains and task types. MultiReQA [19] consists of eight Question-Answering (QA) datasets and evaluates sentence-level answer retrieval given a question. It only tests a single task and five out of eight datasets are from Wikipedia. Further, MultiReQA evaluates retrieval over rather small corpora: six out of eight tasks have less than 100k candidate sentences, which benefits dense retrieval over lexical as previously shown [54]. KILT [50] consists of five knowledge-intensive 2 tasks including a total of eleven…
- **E3** arxiv:2310.11511 §APPENDIX p.29: This is only applicable when the output and part of the evidence are almost identical. - [Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported]. - [No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instr…
- **E4** arxiv:2004.04906 §Overview p.2: Our dense passage retriever (DPR) uses a dense encoder EP (·) which maps any text passage to a ddimensional real-valued vectors and builds an index for all the M passages that we will use for retrieval. 3The ideal size and boundary of a text passage are functions of both the retriever and reader. We also experimented with natural paragraphs in our preliminary trials and found that using fixed-length passages performs better in both retrieval and final QA accuracy, as observed by Wang et al. (2019). 4Exceptions include (Seo et al., 2019) and (Roberts et al., 2020), which retrieves and generates…
- **E5** arxiv:2112.01488 §Questions # Passages Subtopics # Questions # Passages Subtopics p.6: Similar to evaluation in the Open-QA literature (Karpukhin et al., 2020; Khattab et al., Q: what is xerror in rpart? Q: is sub question one word? Q: how to open a garage door without making noise? Q: is docx and dotx the same? Q: are upvotes and downvotes anonymous? Q: what is the difference between descriptive essay and narrative essay? Q: how to change default user profile in chrome? Q: does autohotkey need to be installed? Q: how do you tag someone on facebook with a youtube video? Q: has mjolnir ever been broken? Q: Snoopy can balance on an edge atop his doghouse. Is any reason given for t…

</details>

Human verdict: ☐ agree ☐ disagree

## g038 (B, answerable)

**Q:** How does dense retrieval relate to contrastive learning?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | supported | Dense retrieval relates to contrastive learning through the use of contrastive methods to improve the training of dense retrievers. | The sentence correctly states that contrastive methods are used to improve the training of dense retrievers, which is supported by E2 and E3. |
| 2 | supported | Contrastive learning can enhance unsupervised dense retrieval by approximating retrieval tasks, which helps in generating synthetic queries and training the network to retrieve original documents effectively [E3]. | The sentence is supported by E3, which discusses using contrastive learning to enhance unsupervised dense retrieval by approximating retrieval tasks. |
| 3 | supported | Additionally, contrastive learning has been shown to lead to strong performance in various retrieval settings, outperforming traditional methods like BM25 in many cases [E4]. | The sentence is supported by E4, which states that contrastive learning leads to strong performance in various retrieval settings, outperforming BM25 in many cases. |
| 4 | supported | The integration of contrastive learning techniques, such as selecting hard negatives globally, can address the learning bottlenecks in dense retrieval [E1]. | The sentence is supported by E1, which discusses using techniques like selecting hard negatives globally to address learning bottlenecks in dense retrieval. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2007.00808 §ABSTRACT p.1: Conducting text retrieval in a dense representation space has many intriguing advantages. Yet the end-to-end learned dense retrieval (DR) often underperforms word-based sparse retrieval. In this paper, we first theoretically show the learning bottleneck of dense retrieval is due to the domination of uninformative negatives sampled locally in batch, which yield diminishing gradient norms, large stochastic gradient variances, and slow learning convergence. We then propose Approximate nearest neighbor Negative Contrastive Learning (ANCE), a learning mechanism that selects hard training negatives …
- **E2** arxiv:2112.09118 §Related work p.3: Concurrent to this work, Gao & Callan (2021) have also shown the potential of contrastive learning for information retrieval; building on the same observation that both tasks share a similar structure. Spider (Ram et al., 2021), a contemporary work, uses spans appearing multiple times in a document to create pseudo examples for contrastive learning in order to train unsupervised retrievers. Finally, Chen et al. (2021) train a dense retriever to imitate unsupervised lexical-based methods. This improves performance on a range of tasks and achieves state-of-the-art results when combining the resu…
- **E3** arxiv:2112.09118 §Introduction p.2: Training dense retrievers without supervision can be achieved by using an auxiliary task that approximates retrieval. Given a document, one can generate a synthetic query and then train the network to retrieve the original document, among many others, given the query. The inverse Cloze task (ICT), proposed by Lee et al. (2019) to pre-train retrievers, uses a given sentence as a query and predicts the context surrounding it. While showing promising results as pre-training (Chang et al., 2020; Sachan et al., 2021), this approach still lags behind BM25 when used as a zero-shot retriever. ICT is s…
- **E4** arxiv:2112.09118 §Abstract p.1: Recently, information retrieval has seen the emergence of dense retrievers, using neural networks, as an alternative to classical sparse methods based on term-frequency. These models have obtained state-of-the-art results on datasets and tasks where large training sets are available. However, they do not transfer well to new applications with no training data, and are outperformed by unsupervised term-frequency methods such as BM25. In this work, we explore the limits of contrastive learning as a way to train unsupervised dense retrievers and show that it leads to strong performance in various…
- **E5** arxiv:2212.10496 §Preliminaries p.3: Dense retrieval models similarity between query and document with inner product similarity. Given a query q and document d, it uses two encoder function encq and encd to map them into d dimension vectors vq, vd, whose inner product is used as similarity measurement. sim(q, d) = ⟨encq(q), encd(d)⟩= ⟨vq, vd⟩(1) For zero-shot retrieval, we consider L query sets Q1, Q2, ..., QL and their corresponding search corpus, document sets D1, D2, ..., DL. Denote the j-th query from i-th set query set Qi as qij. We need to fully define mapping functions encq and encd without access to any query set Qi, docu…

</details>

Human verdict: ☐ agree ☐ disagree

## n009 (A, unanswerable)

**Q:** What is the best restaurant near Times Square right now?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | no_claim | The indexed papers do not contain the answer to the question about the best restaurant near Times Square right now. | The sentence states that the indexed papers do not contain the answer to the question, which is not a factual claim about the content of the papers. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2208.03299 §Analysis p.17: However, as shown in row 2, we can swap the Atlas index to a 2020 Wikipedia index, without retraining, and find that Atlas updates its predictions accordingly, with 2020 accuracy rising to a similar level to its 2017 performance (53.1%), whereas the purely parametric T5 has no such updateability mechanism. This demonstrates that Atlas can be faithful and condition strongly on its supplied index. Furthermore, this zero-shot updateability mechanism has the useful property of staying up-to-date without requiring up-to-date annotated data, or continuous, lifelong pre-training, as would be may requ…
- **E2** arxiv:2004.12832 §Methodology p.7: We report the competition’s official metric, namely MRR@10, on the validation set (Dev) and the evaluation set (Eval). We also report the re-ranking latency, which we measure using a single Tesla V100 GPU, and the FLOPs per query for each neural ranking model. For ColBERT, our reported latency subsumes the entire computation from gathering the document representations, moving them to the GPU, tokenizing then encoding the query, and applying late interaction to compute document scores. For the baselines, we measure the scoring computations on the GPU and exclude the CPU-based text preprocessing…
- **E3** arxiv:2311.09210 §Overview p.2: In this section, we introduce CHAIN-OF-NOTE, an innovative advancement for retrieval-augmented language models (RALMs). Specifically, CON framework generates sequential reading notes for the retrieved documents, which enables a systematic evaluation of the relevance and accuracy of information retrieved from external documents. By creating sequential reading notes, the model not only assesses the pertinence of each document to the query but also identifies the most critical and reliable pieces of information within these documents. This process helps in filtering out irrelevant or less trustwo…
- **E4** arxiv:2307.03172 §Effect of Query-Aware Contextualization p.8: of the prompt and decoder-only models can only attend to prior tokens at each timestep. In contrast, encoder-decoder models (which seem more robust to changes in the position of relevant information; §4.1) use a bidirectional encoder to contextualize input contexts—can we use this observation to improve decoder-only models by placing the query before and after the data, enabling query-aware contextualization of documents (or key-value pairs)? We find that query-aware contextualization dramatically improves performance on the key-value retrieval task—all models achieve near-perfect per- formanc…
- **E5** arxiv:2005.11401 §Additional Results p.8: Accuracy with mismatched indices is low (12% with the 2018 index and 2016 leaders, 4% with the 2016 index and 2018 leaders). This shows we can update RAG’s world knowledge by simply replacing its non-parametric memory. Effect of Retrieving more documents Models are trained with either 5 or 10 retrieved latent documents, and we do not observe significant differences in performance between them. We have the flexibility to adjust the number of retrieved documents at test time, which can affect performance and runtime. Figure 3 (left) shows that retrieving more documents at test time monotonically…

</details>

Human verdict: ☐ agree ☐ disagree

## g006 (A, answerable)

**Q:** What tasks does Self-RAG evaluate on?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | supported | Self-RAG evaluates on a range of downstream tasks, including fact verification, multiple-choice reasoning, open-domain question answering, and long-form generation tasks such as biography generation and long-form QA [E2]. | The sentence accurately reflects the information in E2 about the range of downstream tasks Self-RAG evaluates on. |
| 2 | supported | The evaluation metrics used assess overall correctness, factuality, and fluency [E2]. | The sentence correctly states the evaluation metrics used, as mentioned in E2. |
| 3 | supported | Specifically, closed-set tasks include a public health fact verification dataset and a scientific exam reasoning dataset, while short-form generation tasks involve datasets like PopQA and TriviaQA-unfiltered [E2]. | The sentence is supported by E2, which lists the specific datasets used for closed-set and short-form generation tasks. |
| 4 | supported | Long-form generation tasks are evaluated using metrics like FactScore and citation precision and recall [E2]. | The sentence is supported by E2, which mentions the metrics used for evaluating long-form generation tasks. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2310.11511 §RELATED WORK p.3: SELF-RAG processes passages in parallel and filters out irrelevant ones through self-reflection, without relying on external models at inference. Moreover, our self-reflection mechanism also evaluates other aspects of the model output quality including factuality. LATS (Zhou et al., 2023) prompt off-the-shelf LMs to search for relevant information for question answering tasks and to generate with tree search, guided by LM-generated value scores. While their value function simply indicates an overall score of each generation, SELF-RAG trains to an arbitrary LM to learn to generate fine-grained …
- **E2** arxiv:2310.11511 §TASKS AND DATASETS p.6: We conduct evaluations of our SELF-RAG and diverse baselines on a range of downstream tasks, holistically evaluating outputs with metrics designed to assess overall correctness, factuality, and fluency. Throughout these experiments, we conduct zero-shot evaluations, where we provide instructions describing tasks without few-shot demonstrations (Wei et al., 2022; Sanh et al., 2022). Details of our experiments’ settings, including test-time instructions, are available in the Appendix Section B.1. Closed-set tasks include two datasets, i.e., a fact verification dataset about public health (PubHea…
- **E3** arxiv:2310.11511 §CONCLUSION p.10: This work introduces SELF-RAG, a new framework to enhance the quality and factuality of LLMs through retrieval on demand and self-reflection. SELF-RAG trains an LM to learn to retrieve, generate, and critique text passages and its own generation by predicting the next tokens from its original vocabulary as well as newly added special tokens, called reflection tokens. SELF-RAG further enables the tailoring of LM behaviors at test time by leveraging reflection tokens. Our holistic evaluations on six tasks using multiple metrics demonstrate that SELF-RAG significantly outperforms LLMs with more p…
- **E4** arxiv:2312.10997 §TASK AND EVALUATION p.12: The rapid advancement and growing adoption of RAG in the field of NLP have propelled the evaluation of RAG models to the forefront of research in the LLMs community. The primary objective of this evaluation is to comprehend and optimize the performance of RAG models across diverse application scenarios.This chapter will mainly introduce the main downstream tasks of RAG, datasets, and how to evaluate RAG systems. A. Downstream Task The core task of RAG remains Question Answering (QA), including traditional single-hop/multi-hop QA, multiplechoice, domain-specific QA as well as long-form scenario…
- **E5** arxiv:2401.15884 §Accuracy of generation p.10: Table 6: computational overhead assessment of RAG, CRAG, Self-CRAG, and Self-RAG about FLOPs per token on GPUs and executing time per instance. The upper bound of Self-CRAG is lower because only three passages are provided as input (correct, incorrect and ambiguous content). All the data in the table only represents a rough estimate of the generation phase, the retrieval and data-processing stages are not included. anism, rather than solely from the supplementary information obtained through web searches. To further demonstrate the effectiveness of the proposed self-correction mechanism, both …

</details>

Human verdict: ☐ agree ☐ disagree

## g034 (A, answerable)

**Q:** Compare Self-RAG and the hallucination survey on self-reflection.

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | supported | Self-RAG enhances the quality and factuality of large language models (LLMs) through a framework that incorporates retrieval and self-reflection, allowing the model to critique its own outputs using special reflection tokens [E3][E4]. | The claim about Self-RAG enhancing quality and factuality through retrieval and self-reflection using reflection tokens is supported by [E3] and [E4]. |
| 2 | supported | This approach contrasts with traditional retrieval-augmented generation (RAG) methods, which do not ensure complete support from cited sources [E3]. | The contrast between Self-RAG and traditional RAG methods regarding support from cited sources is supported by [E3]. |
| 3 | supported | In comparison, the hallucination survey discusses an interactive self-reflection methodology aimed at reducing hallucinations in medical generative QA systems, emphasizing the iterative feedback process to improve factuality and consistency [E2]. | The description of the hallucination survey's interactive self-reflection methodology aimed at reducing hallucinations is supported by [E2]. |
| 4 | supported | While both Self-RAG and the hallucination survey utilize self-reflection, Self-RAG focuses on enhancing model output quality through retrieval and critique, whereas the hallucination survey specifically addresses the mitigation of hallucinations in generated answers. | The comparison between Self-RAG's focus on output quality and the hallucination survey's focus on mitigating hallucinations is supported by [E1], [E2], and [E3]. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2310.11511 §RELATED WORK p.3: SELF-RAG processes passages in parallel and filters out irrelevant ones through self-reflection, without relying on external models at inference. Moreover, our self-reflection mechanism also evaluates other aspects of the model output quality including factuality. LATS (Zhou et al., 2023) prompt off-the-shelf LMs to search for relevant information for question answering tasks and to generate with tree search, guided by LM-generated value scores. While their value function simply indicates an overall score of each generation, SELF-RAG trains to an arbitrary LM to learn to generate fine-grained …
- **E2** arxiv:2401.01313 §Retrieval Augmented Generation p.5: The paper outlines previous works on LLM reliability, highlighting the novelty of this study’s comprehensive analysis and focus on effective prompting strategies. Drawing inspiration from ML safety surveys, the reliability framework aligns with identified risks in existing conceptual frameworks. Lastly, the systematic exploration of GPT-3’s reliability has been summarized, which introduces practical prompting strategies, and emphasizes the study’s contribution to insights into LLMs and practical recommendations for GPT-3 users. ChatProtect: (Mündler et al., 2023) focuses on an important type o…
- **E3** arxiv:2310.11511 §SELF-RAG: LEARNING TO RETRIEVE, GENERATE AND CRITIQUE p.3: We introduce Self-Reflective Retrieval-Augmented Generation (SELF-RAG), shown in Figure 1. SELF-RAG is a framework that enhances the quality and factuality of an LLM through retrieval and self-reflection, without sacrificing LLM’s original creativity and versatility. Our end-to-end training lets an LM M generate text informed by retrieved passages, if needed, and criticize the output by learning to generate special tokens. These reflection tokens (Table 1) signal the need for retrieval or confirm the output’s relevance, support, or completeness. In contrast, common RAG approaches retrieve pass…
- **E4** arxiv:2310.11511 §CONCLUSION p.10: This work introduces SELF-RAG, a new framework to enhance the quality and factuality of LLMs through retrieval on demand and self-reflection. SELF-RAG trains an LM to learn to retrieve, generate, and critique text passages and its own generation by predicting the next tokens from its original vocabulary as well as newly added special tokens, called reflection tokens. SELF-RAG further enables the tailoring of LM behaviors at test time by leveraging reflection tokens. Our holistic evaluations on six tasks using multiple metrics demonstrate that SELF-RAG significantly outperforms LLMs with more p…

</details>

Human verdict: ☐ agree ☐ disagree

## g006 (B, answerable)

**Q:** What tasks does Self-RAG evaluate on?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | supported | Self-RAG evaluates on a range of downstream tasks, including fact verification, multiple-choice reasoning, open-domain question answering, and long-form generation tasks such as biography generation and long-form QA [E2]. | The sentence accurately reflects the tasks mentioned in E2, including fact verification, multiple-choice reasoning, open-domain QA, and long-form generation tasks. |
| 2 | supported | The evaluation metrics used assess overall correctness, factuality, and fluency [E2]. | The sentence correctly states that the evaluation metrics assess overall correctness, factuality, and fluency, as mentioned in E2. |
| 3 | supported | Specifically, closed-set tasks involve datasets like PubHealth and ARC-6, while short-form generation tasks include PopQA and TriviaQA-unfiltered [E2]. | The sentence correctly identifies the datasets for closed-set and short-form generation tasks as mentioned in E2. |
| 4 | supported | Long-form generation tasks are evaluated using metrics like FactScore and citation precision and recall [E2]. | The sentence accurately describes the evaluation metrics for long-form generation tasks as mentioned in E2. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2310.11511 §RELATED WORK p.3: SELF-RAG processes passages in parallel and filters out irrelevant ones through self-reflection, without relying on external models at inference. Moreover, our self-reflection mechanism also evaluates other aspects of the model output quality including factuality. LATS (Zhou et al., 2023) prompt off-the-shelf LMs to search for relevant information for question answering tasks and to generate with tree search, guided by LM-generated value scores. While their value function simply indicates an overall score of each generation, SELF-RAG trains to an arbitrary LM to learn to generate fine-grained …
- **E2** arxiv:2310.11511 §TASKS AND DATASETS p.6: We conduct evaluations of our SELF-RAG and diverse baselines on a range of downstream tasks, holistically evaluating outputs with metrics designed to assess overall correctness, factuality, and fluency. Throughout these experiments, we conduct zero-shot evaluations, where we provide instructions describing tasks without few-shot demonstrations (Wei et al., 2022; Sanh et al., 2022). Details of our experiments’ settings, including test-time instructions, are available in the Appendix Section B.1. Closed-set tasks include two datasets, i.e., a fact verification dataset about public health (PubHea…
- **E3** arxiv:2310.11511 §CONCLUSION p.10: This work introduces SELF-RAG, a new framework to enhance the quality and factuality of LLMs through retrieval on demand and self-reflection. SELF-RAG trains an LM to learn to retrieve, generate, and critique text passages and its own generation by predicting the next tokens from its original vocabulary as well as newly added special tokens, called reflection tokens. SELF-RAG further enables the tailoring of LM behaviors at test time by leveraging reflection tokens. Our holistic evaluations on six tasks using multiple metrics demonstrate that SELF-RAG significantly outperforms LLMs with more p…
- **E4** arxiv:2312.10997 §TASK AND EVALUATION p.12: The rapid advancement and growing adoption of RAG in the field of NLP have propelled the evaluation of RAG models to the forefront of research in the LLMs community. The primary objective of this evaluation is to comprehend and optimize the performance of RAG models across diverse application scenarios.This chapter will mainly introduce the main downstream tasks of RAG, datasets, and how to evaluate RAG systems. A. Downstream Task The core task of RAG remains Question Answering (QA), including traditional single-hop/multi-hop QA, multiplechoice, domain-specific QA as well as long-form scenario…
- **E5** arxiv:2401.15884 §Accuracy of generation p.10: Table 6: computational overhead assessment of RAG, CRAG, Self-CRAG, and Self-RAG about FLOPs per token on GPUs and executing time per instance. The upper bound of Self-CRAG is lower because only three passages are provided as input (correct, incorrect and ambiguous content). All the data in the table only represents a rough estimate of the generation phase, the retrieval and data-processing stages are not included. anism, rather than solely from the supplementary information obtained through web searches. To further demonstrate the effectiveness of the proposed self-correction mechanism, both …

</details>

Human verdict: ☐ agree ☐ disagree

## g003 (B, answerable)

**Q:** What do IsRel, IsSup, and IsUse evaluate in Self-RAG?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | supported | IsRel, IsSup, and IsUse evaluate different aspects of the model's output in Self-RAG. | The evidence in E2 states that IsRel, IsSup, and IsUse evaluate different aspects of the model's output in Self-RAG. |
| 2 | supported | IsRel assesses whether a retrieved passage is relevant to the generation, while IsSup evaluates if the passage supports the model's generation. | E2 explains that IsRel assesses relevance and IsSup evaluates support, which aligns with the claims in the sentence. |
| 3 | supported | Finally, IsUse predicts the overall utility of the output at the end of the generation process [E2]. | E2 mentions that IsUse predicts the overall utility of the output at the end of the generation process. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2310.11511 §SELF-RAG INFERENCE p.6: Generating reflection tokens to self-evaluate its own output makes SELF-RAG controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. For tasks demanding factual accuracy (Min et al., 2023), we aim for the model to retrieve passages more frequently to ensure that the output aligns closely with the available evidence. Conversely, in more open-ended tasks, like composing a personal experience essay, the emphasis shifts towards retrieving less and prioritizing the overall creativity or utility score. In this section, we describe approaches to enforc…
- **E2** arxiv:2310.11511 §TRAINING THE GENERATOR MODEL p.5: Data collection for generator. Given an input-output pair (x, y), we augment the original output y using the retrieval and critic models to create supervised data that precisely mimics the SELF- RAG inference-time process (Section 3.1). For each segment yt ∈y, we run C to assess whether additional passages could help to enhance generation. If retrieval is required, the retrieval special token Retrieve =Yes is added, and R retrieves the top K passages, D. For each passage, C further evaluates whether the passage is relevant and predicts ISREL . If a passage is relevant, C further evaluates whet…
- **E3** arxiv:2310.11511 §APPENDIX p.20: Consequently, all of the baseline output length is within 30-60 tokens. For FactScore, we set the maximum new token length to 500 for baselines and 200 for SELF-RAG at each segment level. Task-specific instructions. Table 5 shows the list of the instructions used during evaluations. For Open-domain QA, we do not provide explicit instructions. C RESULTS C.1 ANALYSIS Reliance on parametric- and non-parametric memories. We conduct analysis on how frequently model answers come from retrieved passages (non-parametric memories) or their own parametric memories. On two open-domain QA datasets, Trivia…

</details>

Human verdict: ☐ agree ☐ disagree

## n007 (A, unanswerable)

**Q:** Which NBA team won last night?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | no_claim | The indexed papers do not contain the answer to which NBA team won last night. | The sentence states that the indexed papers do not contain the answer, which is not a factual claim about the content of the papers. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2310.11511 §APPENDIX p.22: ISSUP =Partially Supported][ Retrieve =No]Since it is not possible for the manager to handle everything alone, they must delegate tasks to their team members.[ Retrieve =No]The manager would then assign each team member specific responsibilities based on their individual skillsets and strengths.[ Retrieve =Yes]<p>Joint application design – Subject Matter Experts: These are the business users, the IS professionals, and the outside experts that will be needed for a successful workshop. This group is the backbone of the meeting<p>[ ISREL =Irrelevant]For instance, the team ́s graphic designer migh…
- **E2** arxiv:2302.00083 §Model p.2: World Cup 2022 was the last with 32 teams, before the increase to Retriever FIFA World Cup 2026 will expand to 48 teams. World Cup 2022 was the last with 32 teams, before the increase to 48 in the 2026 tournament. Figure 2: An example of In-Context RALM: we simply prepend the retrieved document before the input prefix. tend to be focused on altering the language model architecture (Khandelwal et al., 2020; Borgeaud et al., 2022; Zhong et al., 2022; Levine et al., 2022c; Li et al., 2022). Notably, Borgeaud et al. (2022) introduced RETRO, featuring document reading via nontrivial modifications t…
- **E3** arxiv:2309.01431 §Generation p.4: For example, as shown in the Figure 2, for a report about “The 2022 Nobel Prize”, ChatGPT will generate corresponding event, question and provide key information for answering it. By generating events, the model is able to preliminarily filter out news articles that do not contain any events. After generation, we manually check the answer and filter out data that is difficult to retrieve through search engines. Retrieve using search engine. For each query, we use Google’s API to fetch 10 relevant web pages and extract corresponding snippets of text from them. Simultaneously, we read these web …
- **E4** arxiv:2309.01431 §Generation p.5: Answer Anett Kontaveit Vision Pro 18.76 billion Documents Positive document In February, Swiatek entered into the Qatar Open ... In the final, she won ... Anett Kontaveit ... Negative document This time, she defeated Ons Jabeur 6-2, 7-6(5) to win the 2022 US Open, ... Positive document Apple (AAPL.O) on Monday unveiled a costly augmented-reality headset called the Vision Pro ... Negative document ... is what Gurman believes will be called Apple Reality Pro. ... Positive document Tesla, Inc. (TSLA) reported Q1 FY 2022 earnings results ... detailed revenues of $18.76 billion ... Negative documen…
- **E5** arxiv:2309.01431 §Experiments p.6: We can see that negative rejection poses a challenge for RAG in LLMs. Question Answer Response who will direct Irredeemable film? Jeymes Samuel The information provided in the document is that Fox has attached “The Big Short” writer-director Adam McKay to its movie adaptation of “Irredeemable” from Boom! Comics. Therefore, the answer is Adam McKay. Which country won the most medals at the 2022 Winter Olympics? Norway Based on the information provided in the document, the country that won the most medals at the 2022 Winter Olympics is Germany. Germany has won a total of 31 medals, including 14 …

</details>

Human verdict: ☐ agree ☐ disagree

## g041 (B, answerable)

**Q:** What does Self-RAG use critique tokens for after retrieving a passage?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | supported | Self-RAG uses critique tokens to assess its own predictions after generating each segment of text. | The claim that Self-RAG uses critique tokens to assess its own predictions after generating each segment of text is supported by [E2] and [E3]. |
| 2 | supported | Specifically, it employs different types of critique tokens, such as ISREL (to indicate relevance) and ISSUP (to indicate support), which are used to evaluate the retrieved passages and the generated output [E1][E3]. | The use of critique tokens like ISREL and ISSUP to evaluate retrieved passages and generated output is supported by [E1] and [E3]. |
| 3 | supported | The scores from these critique tokens contribute to the overall segment score, allowing the model to refine its output based on the evaluation of the retrieved information [E1]. | The claim that scores from critique tokens contribute to the overall segment score is supported by [E1]. |
| 4 | supported | This mechanism enables Self-RAG to dynamically adjust its behavior and improve the quality of its generated content [E2]. | The mechanism enabling Self-RAG to dynamically adjust its behavior and improve content quality is supported by [E2]. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2310.11511 §SELF-RAG INFERENCE p.6: Generating reflection tokens to self-evaluate its own output makes SELF-RAG controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. For tasks demanding factual accuracy (Min et al., 2023), we aim for the model to retrieve passages more frequently to ensure that the output aligns closely with the available evidence. Conversely, in more open-ended tasks, like composing a personal experience essay, the emphasis shifts towards retrieving less and prioritizing the overall creativity or utility score. In this section, we describe approaches to enforc…
- **E2** arxiv:2310.11511 §INTRODUCTION p.2: Moreover, SELF-RAG provides citations for each segment with its self-assessment of whether the output is supported by the passage, leading to easier fact verification. SELF-RAG trains an arbitrary LM to generate text with reflection tokens by unifying them as the next token prediction from the expanded model vocabulary. We train our generator LM on a diverse collection of text interleaved with reflection tokens and retrieved passages. Reflection tokens, inspired by reward models used in reinforcement learning (Ziegler et al., 2019; Ouyang et al., 2022), are inserted offline into the original c…
- **E3** arxiv:2310.11511 §TRAINING THE GENERATOR MODEL p.5: Data collection for generator. Given an input-output pair (x, y), we augment the original output y using the retrieval and critic models to create supervised data that precisely mimics the SELF- RAG inference-time process (Section 3.1). For each segment yt ∈y, we run C to assess whether additional passages could help to enhance generation. If retrieval is required, the retrieval special token Retrieve =Yes is added, and R retrieves the top K passages, D. For each passage, C further evaluates whether the passage is relevant and predicts ISREL . If a passage is relevant, C further evaluates whet…

</details>

Human verdict: ☐ agree ☐ disagree

## g039 (A, answerable)

**Q:** What does the original RAG paper retrieve from its non-parametric memory?

| # | Label | Sentence | Judge reason |
| --- | --- | --- | --- |
| 1 | supported | The original RAG paper retrieves latent documents from a dense vector index of Wikipedia, which serves as its non-parametric memory [E2]. | The claim that the original RAG paper retrieves latent documents from a dense vector index of Wikipedia is directly supported by E2. |
| 2 | supported | These documents are accessed using a pre-trained neural retriever, specifically the Dense Passage Retriever (DPR), and are then used by the seq2seq model (BART) to generate outputs [E2][E4]. | The claim that these documents are accessed using a pre-trained neural retriever, specifically the Dense Passage Retriever (DPR), and used by the seq2seq model (BART) to generate outputs is supported by E2 and E4. |
| 3 | supported | The retrieval process allows the model to integrate a vast range of information without additional training, enhancing its performance on knowledge-intensive tasks [E4]. | The claim that the retrieval process allows the model to integrate a vast range of information without additional training, enhancing its performance on knowledge-intensive tasks, is supported by E4. |

<details><summary>Evidence given to the model</summary>

- **E1** arxiv:2005.11401 §Additional Results p.8: Accuracy with mismatched indices is low (12% with the 2018 index and 2016 leaders, 4% with the 2016 index and 2018 leaders). This shows we can update RAG’s world knowledge by simply replacing its non-parametric memory. Effect of Retrieving more documents Models are trained with either 5 or 10 retrieved latent documents, and we do not observe significant differences in performance between them. We have the flexibility to adjust the number of retrieved documents at test time, which can affect performance and runtime. Figure 3 (left) shows that retrieving more documents at test time monotonically…
- **E2** arxiv:2005.11401 §Introduction p.2: but have only explored open-domain extractive question answering. Here, we bring hybrid parametric and non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose fine-tuning approach which we refer to as retrieval-augmented generation (RAG). We build RAG models where the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We combine …
- **E3** arxiv:2005.11401 §Discussion p.9: In this work, we presented hybrid generation models with access to parametric and non-parametric memory. We showed that our RAG models obtain state of the art results on open-domain QA. We found that people prefer RAG’s generation over purely parametric BART, finding RAG more factual and specific. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated how the retrieval index can be hot-swapped to update the model without requiring any retraining. In future work, it may be fruitful to investigate if the two components can be j…
- **E4** arxiv:2401.01313 §Retrieval Augmented Generation p.4: In this process, the DPR acts as a neural retriever, supplying relevant documents based on the input. These documents are then used by the seq2seq model, specifically BART, to generate the final output. The model employs a top-K approximation to marginalize these latent documents, which can be done on a per-output basis (assuming one document is responsible for all tokens) or a per-token basis (allowing different documents to influence different parts of the output). Crucially, both the generator and the retriever in this RAG setup are trained end-to-end, ensuring that they learn jointly and i…
- **E5** arxiv:2310.11511 §APPENDIX p.20: Consequently, all of the baseline output length is within 30-60 tokens. For FactScore, we set the maximum new token length to 500 for baselines and 200 for SELF-RAG at each segment level. Task-specific instructions. Table 5 shows the list of the instructions used during evaluations. For Open-domain QA, we do not provide explicit instructions. C RESULTS C.1 ANALYSIS Reliance on parametric- and non-parametric memories. We conduct analysis on how frequently model answers come from retrieved passages (non-parametric memories) or their own parametric memories. On two open-domain QA datasets, Trivia…

</details>

Human verdict: ☐ agree ☐ disagree
