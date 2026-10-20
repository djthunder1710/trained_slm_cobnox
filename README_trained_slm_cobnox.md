# trained_slm_cobnox

**Author:** Khadija  
**Project:** Small Language Model (SLM) built with Cobnox  
**Repository:** `trained_slm_cobnox`

## Overview

Khadija developed this Cobnox SLM project to experiment with training a small language model on AI-related question-and-answer examples. The dataset contains 70 examples covering topics such as artificial intelligence, machine learning, neural networks, tokenization, transformers, RAG, and model training.

The workflow includes:
1. Pre-training on an AI knowledge corpus.
2. Supervised fine-tuning (SFT) with question-and-answer pairs.
3. Saving model parameters and tokenizer data.
4. Loading the fine-tuned model for chat inference.

## Reported Results

```text
Pre-training Loss:
0.0921

SFT Loss:
0.0891
```

The reported run took approximately **2 minutes** in the current environment. Runtime can vary with hardware and settings. Lower training loss alone does not prove accuracy; more testing is needed.

## Sample Output

```text
Ask trained model a question: Who trained you?
i am trained by khadija.

Model Loaded
```

## Model Configuration

```text
create_slm2_engine(512, 64, 128, 0.002)
```

- Pre-training epochs: `2`
- SFT steps: `65`
- SFT epochs: `1`
- Dataset examples: `70`
- Model files: `.cbxm`
- Tokenizer files: `.json`

## Source Code

> **Important:** This is a cleaned-up version of the workflow you provided. Cobnox engine function names and loop syntax are retained from your example and should be checked against your installed Cobnox version. The tokenizer must use the same vocabulary/ID mapping as the trained model.

```text
use "sys"
use "engine"

# 1. Prepare the 70-example training corpus
arr = alloc_array(70, "")

arr[0] = "Who trained you? I am trained by Khadija."
arr[1] = "What is artificial intelligence? Artificial intelligence is the science of making machines that can learn and think."
arr[2] = "How does machine learning work? Machine learning works by finding patterns in data and improving with experience."
arr[3] = "Why is deep learning powerful? Deep learning is powerful because multi-layer networks can learn complex features."
arr[4] = "Can you explain a neural network? A neural network is a system of connected nodes that process information like a brain."
arr[5] = "How do transformers understand text? Transformers use self-attention to understand relationships between words."
arr[6] = "Why is self attention important? Self attention helps the model focus on the most relevant parts of the input."
arr[7] = "How does pre-training help AI? Pre-training teaches the model general language knowledge from large text data."
arr[8] = "When should we use fine-tuning? Fine-tuning is used when we want to adapt a model for a specific task."
arr[9] = "What is the difference between supervised and unsupervised learning? Supervised uses labeled data while unsupervised finds patterns without labels."
arr[10] = "How does reinforcement learning improve agents? Reinforcement learning improves agents by giving rewards for good actions."
arr[11] = "Why do we use gradient descent? Gradient descent is used to reduce the model error step by step."
arr[12] = "How is a loss function useful? A loss function shows how wrong the model prediction is so it can improve."
arr[13] = "Can you explain backpropagation? Backpropagation calculates how much each weight contributed to the error."
arr[14] = "Why are embeddings useful? Embeddings turn words into numbers so models can understand meaning and similarity."
arr[15] = "How does a tokenizer work? A tokenizer splits text into smaller pieces called tokens for the model."
arr[16] = "What makes RAG different from normal generation? RAG retrieves real information first and then generates the answer."
arr[17] = "How does vector search find results? Vector search finds results by comparing embedding similarity."
arr[18] = "Why is semantic search better than keyword search? Semantic search understands meaning instead of just matching words."
arr[19] = "How are large language models trained? Large language models are trained on massive amounts of text data."
arr[20] = "Why is prompt engineering important? Prompt engineering helps guide the model to give better answers."
arr[21] = "How can we reduce AI hallucinations? We can reduce hallucinations by using retrieval and better training data."
arr[22] = "What happens when temperature is high? High temperature makes the model output more random and creative."
arr[23] = "How does top-k sampling work? Top-k sampling only considers the k most likely next tokens."
arr[24] = "Why use beam search? Beam search explores multiple possible answers to find a better one."
arr[25] = "How do APIs help AI systems? APIs allow different applications to send requests to AI models."
arr[26] = "What is the difference between training and inference? Training teaches the model while inference uses it to answer."
arr[27] = "Why is good training data important? Good training data helps the model learn correct and useful patterns."
arr[28] = "How can we detect overfitting? Overfitting is detected when the model performs well on training but poorly on new data."
arr[29] = "What causes underfitting? Underfitting happens when the model is too simple to learn the patterns."
arr[30] = "Why are GPUs used in AI? GPUs are used because they can do many calculations in parallel very fast."
arr[31] = "How is a TPU different from a GPU? A TPU is specially designed for machine learning calculations."
arr[32] = "When is transfer learning useful? Transfer learning is useful when we have limited data for a new task."
arr[33] = "How does zero-shot learning work? Zero-shot learning allows the model to handle new tasks without examples."
arr[34] = "Why is few-shot learning helpful? Few-shot learning helps the model adapt quickly with only a few examples."
arr[35] = "How do diffusion models generate images? Diffusion models generate images by gradually removing noise."
arr[36] = "What is the idea behind GANs? GANs train a generator and a discriminator against each other."
arr[37] = "How does computer vision work? Computer vision helps machines understand and analyze images and videos."
arr[38] = "Why is natural language processing important? NLP is important because it lets machines understand human language."
arr[39] = "How does speech recognition work? Speech recognition converts spoken audio into written text."
arr[40] = "What is the benefit of text-to-speech? Text-to-speech allows written text to be spoken out loud."
arr[41] = "Why is multimodal AI useful? Multimodal AI can understand text, images, audio and video together."
arr[42] = "How do AI agents work? AI agents plan and take actions to achieve specific goals."
arr[43] = "Why use chain of thought prompting? Chain of thought helps the model reason step by step for better answers."
arr[44] = "How does retrieval improve answers? Retrieval finds relevant information so the model can give accurate responses."
arr[45] = "What is stored in a knowledge base? A knowledge base stores structured information that AI can use."
arr[46] = "How is embedding similarity measured? Embedding similarity is measured by how close two vectors are."
arr[47] = "Why is cosine similarity popular? Cosine similarity is popular because it focuses on the direction of vectors."
arr[48] = "How does a vector database help search? A vector database stores embeddings and finds similar ones quickly."
arr[49] = "What is the advantage of hybrid search? Hybrid search combines keyword matching and meaning-based search."
arr[50] = "How does reranking improve results? Reranking reorders search results using a smarter model."
arr[51] = "Why do we chunk documents in RAG? Chunking splits long documents into smaller pieces for better retrieval."
arr[52] = "What limits the context window? The context window is limited by the model architecture and memory."
arr[53] = "Why is token limit important? Token limit decides how much text the model can process at once."
arr[54] = "How does quantization help models? Quantization reduces model size by using lower precision numbers."
arr[55] = "What is the goal of distillation? Distillation transfers knowledge from a large model to a smaller one."
arr[56] = "Why is open source AI important? Open source AI allows everyone to use and improve models freely."
arr[57] = "How does bias appear in AI? Bias appears when the training data contains unfair or unbalanced examples."
arr[58] = "Why is AI safety necessary? AI safety is necessary to keep systems reliable and under control."
arr[59] = "How does alignment work in AI? Alignment makes sure the AI behavior matches human goals and values."
arr[60] = "What is the purpose of RLHF? RLHF uses human feedback to teach the model preferred responses."
arr[61] = "How does Direct Preference Optimization work? DPO trains the model by comparing preferred and rejected answers."
arr[62] = "Why do we need optimizers? Optimizers update model weights to reduce the loss during training."
arr[63] = "How does the Adam optimizer work? Adam adapts the learning rate for each parameter individually."
arr[64] = "What happens if learning rate is too high? If learning rate is too high the model may fail to learn properly."
arr[65] = "Why do we use batches in training? Batches allow the model to learn from groups of examples efficiently."
arr[66] = "How many times does an epoch run? An epoch runs once through the entire training dataset."
arr[67] = "Why is regularization used? Regularization is used to prevent the model from overfitting."
arr[68] = "How does dropout help training? Dropout randomly disables neurons so the model generalizes better."
arr[69] = "What is the future of AI search? The future of AI search is more accurate semantic and multimodal retrieval."

# 2. Pre-train and save the base model
ai = create_slm2_engine(512, 64, 128, 0.002)
pre_loss = engine_pretrain_corpus(ai, arr, 80, 2)

print("Pre-training Loss:")
print(pre_loss)

save_model("stage1_base.cbxm", ai.model.params, "COBNOX_BASE")
tokenizer_save(ai.tokenizer, "stage1_tok.json")

# 3. Load base weights for supervised fine-tuning
ai2 = create_slm2_engine(512, 64, 128, 0.002)
load_model("stage1_base.cbxm", ai2.model.params)

# Use the tokenizer associated with the base model.
ai2.tokenizer = tokenizer_load("stage1_tok.json")

prompts = alloc_array(70, "")
responses = alloc_array(70, "")

# Keep each prompt paired with its corresponding response.
prompts[0] = "User: Who trained you?\nBot: "
responses[0] = "I am trained by Khadija.\n"
prompts[1] = "User: What is artificial intelligence?\nBot: "
responses[1] = "Artificial intelligence is the science of making machines that can learn and think.\n"
prompts[2] = "User: How does machine learning work?\nBot: "
responses[2] = "Machine learning works by finding patterns in data and improving with experience.\n"
prompts[3] = "User: Why is deep learning powerful?\nBot: "
responses[3] = "Deep learning is powerful because multi-layer networks can learn complex features.\n"
prompts[4] = "User: Can you explain a neural network?\nBot: "
responses[4] = "A neural network is a system of connected nodes that process information like a brain.\n"
prompts[5] = "User: How do transformers understand text?\nBot: "
responses[5] = "Transformers use self-attention to understand relationships between words.\n"
prompts[6] = "User: Why is self attention important?\nBot: "
responses[6] = "Self attention helps the model focus on the most relevant parts of the input.\n"
prompts[7] = "User: How does pre-training help AI?\nBot: "
responses[7] = "Pre-training teaches the model general language knowledge from large text data.\n"
prompts[8] = "User: When should we use fine-tuning?\nBot: "
responses[8] = "Fine-tuning is used when we want to adapt a model for a specific task.\n"
prompts[9] = "User: What is the difference between supervised and unsupervised learning?\nBot: "
responses[9] = "Supervised uses labeled data while unsupervised finds patterns without labels.\n"
prompts[10] = "User: How does reinforcement learning improve agents?\nBot: "
responses[10] = "Reinforcement learning improves agents by giving rewards for good actions.\n"
prompts[11] = "User: Why do we use gradient descent?\nBot: "
responses[11] = "Gradient descent is used to reduce the model error step by step.\n"
prompts[12] = "User: How is a loss function useful?\nBot: "
responses[12] = "A loss function shows how wrong the model prediction is so it can improve.\n"
prompts[13] = "User: Can you explain backpropagation?\nBot: "
responses[13] = "Backpropagation calculates how much each weight contributed to the error.\n"
prompts[14] = "User: Why are embeddings useful?\nBot: "
responses[14] = "Embeddings turn words into numbers so models can understand meaning and similarity.\n"
prompts[15] = "User: How does a tokenizer work?\nBot: "
responses[15] = "A tokenizer splits text into smaller pieces called tokens for the model.\n"
prompts[16] = "User: What makes RAG different from normal generation?\nBot: "
responses[16] = "RAG retrieves real information first and then generates the answer.\n"
prompts[17] = "User: How does vector search find results?\nBot: "
responses[17] = "Vector search finds results by comparing embedding similarity.\n"
prompts[18] = "User: Why is semantic search better than keyword search?\nBot: "
responses[18] = "Semantic search understands meaning instead of just matching words.\n"
prompts[19] = "User: How are large language models trained?\nBot: "
responses[19] = "Large language models are trained on massive amounts of text data.\n"
prompts[20] = "User: Why is prompt engineering important?\nBot: "
responses[20] = "Prompt engineering helps guide the model to give better answers.\n"
prompts[21] = "User: How can we reduce AI hallucinations?\nBot: "
responses[21] = "We can reduce hallucinations by using retrieval and better training data.\n"
prompts[22] = "User: What happens when temperature is high?\nBot: "
responses[22] = "High temperature makes the model output more random and creative.\n"
prompts[23] = "User: How does top-k sampling work?\nBot: "
responses[23] = "Top-k sampling only considers the k most likely next tokens.\n"
prompts[24] = "User: Why use beam search?\nBot: "
responses[24] = "Beam search explores multiple possible answers to find a better one.\n"
prompts[25] = "User: How do APIs help AI systems?\nBot: "
responses[25] = "APIs allow different applications to send requests to AI models.\n"
prompts[26] = "User: What is the difference between training and inference?\nBot: "
responses[26] = "Training teaches the model while inference uses it to answer.\n"
prompts[27] = "User: Why is good training data important?\nBot: "
responses[27] = "Good training data helps the model learn correct and useful patterns.\n"
prompts[28] = "User: How can we detect overfitting?\nBot: "
responses[28] = "Overfitting is detected when the model performs well on training but poorly on new data.\n"
prompts[29] = "User: What causes underfitting?\nBot: "
responses[29] = "Underfitting happens when the model is too simple to learn the patterns.\n"
prompts[30] = "User: Why are GPUs used in AI?\nBot: "
responses[30] = "GPUs are used because they can do many calculations in parallel very fast.\n"
prompts[31] = "User: How is a TPU different from a GPU?\nBot: "
responses[31] = "A TPU is specially designed for machine learning calculations.\n"
prompts[32] = "User: When is transfer learning useful?\nBot: "
responses[32] = "Transfer learning is useful when we have limited data for a new task.\n"
prompts[33] = "User: How does zero-shot learning work?\nBot: "
responses[33] = "Zero-shot learning allows the model to handle new tasks without examples.\n"
prompts[34] = "User: Why is few-shot learning helpful?\nBot: "
responses[34] = "Few-shot learning helps the model adapt quickly with only a few examples.\n"
prompts[35] = "User: How do diffusion models generate images?\nBot: "
responses[35] = "Diffusion models generate images by gradually removing noise.\n"
prompts[36] = "User: What is the idea behind GANs?\nBot: "
responses[36] = "GANs train a generator and a discriminator against each other.\n"
prompts[37] = "User: How does computer vision work?\nBot: "
responses[37] = "Computer vision helps machines understand and analyze images and videos.\n"
prompts[38] = "User: Why is natural language processing important?\nBot: "
responses[38] = "NLP is important because it lets machines understand human language.\n"
prompts[39] = "User: How does speech recognition work?\nBot: "
responses[39] = "Speech recognition converts spoken audio into written text.\n"
prompts[40] = "User: What is the benefit of text-to-speech?\nBot: "
responses[40] = "Text-to-speech allows written text to be spoken out loud.\n"
prompts[41] = "User: Why is multimodal AI useful?\nBot: "
responses[41] = "Multimodal AI can understand text, images, audio and video together.\n"
prompts[42] = "User: How do AI agents work?\nBot: "
responses[42] = "AI agents plan and take actions to achieve specific goals.\n"
prompts[43] = "User: Why use chain of thought prompting?\nBot: "
responses[43] = "Chain of thought helps the model reason step by step for better answers.\n"
prompts[44] = "User: How does retrieval improve answers?\nBot: "
responses[44] = "Retrieval finds relevant information so the model can give accurate responses.\n"
prompts[45] = "User: What is stored in a knowledge base?\nBot: "
responses[45] = "A knowledge base stores structured information that AI can use.\n"
prompts[46] = "User: How is embedding similarity measured?\nBot: "
responses[46] = "Embedding similarity is measured by how close two vectors are.\n"
prompts[47] = "User: Why is cosine similarity popular?\nBot: "
responses[47] = "Cosine similarity is popular because it focuses on the direction of vectors.\n"
prompts[48] = "User: How does a vector database help search?\nBot: "
responses[48] = "A vector database stores embeddings and finds similar ones quickly.\n"
prompts[49] = "User: What is the advantage of hybrid search?\nBot: "
responses[49] = "Hybrid search combines keyword matching and meaning-based search.\n"
prompts[50] = "User: How does reranking improve results?\nBot: "
responses[50] = "Reranking reorders search results using a smarter model.\n"
prompts[51] = "User: Why do we chunk documents in RAG?\nBot: "
responses[51] = "Chunking splits long documents into smaller pieces for better retrieval.\n"
prompts[52] = "User: What limits the context window?\nBot: "
responses[52] = "The context window is limited by the model architecture and memory.\n"
prompts[53] = "User: Why is token limit important?\nBot: "
responses[53] = "Token limit decides how much text the model can process at once.\n"
prompts[54] = "User: How does quantization help models?\nBot: "
responses[54] = "Quantization reduces model size by using lower precision numbers.\n"
prompts[55] = "User: What is the goal of distillation?\nBot: "
responses[55] = "Distillation transfers knowledge from a large model to a smaller one.\n"
prompts[56] = "User: Why is open source AI important?\nBot: "
responses[56] = "Open source AI allows everyone to use and improve models freely.\n"
prompts[57] = "User: How does bias appear in AI?\nBot: "
responses[57] = "Bias appears when the training data contains unfair or unbalanced examples.\n"
prompts[58] = "User: Why is AI safety necessary?\nBot: "
responses[58] = "AI safety is necessary to keep systems reliable and under control.\n"
prompts[59] = "User: How does alignment work in AI?\nBot: "
responses[59] = "Alignment makes sure the AI behavior matches human goals and values.\n"
prompts[60] = "User: What is the purpose of RLHF?\nBot: "
responses[60] = "RLHF uses human feedback to teach the model preferred responses.\n"
prompts[61] = "User: How does Direct Preference Optimization work?\nBot: "
responses[61] = "DPO trains the model by comparing preferred and rejected answers.\n"
prompts[62] = "User: Why do we need optimizers?\nBot: "
responses[62] = "Optimizers update model weights to reduce the loss during training.\n"
prompts[63] = "User: How does the Adam optimizer work?\nBot: "
responses[63] = "Adam adapts the learning rate for each parameter individually.\n"
prompts[64] = "User: What happens if learning rate is too high?\nBot: "
responses[64] = "If learning rate is too high the model may fail to learn properly.\n"
prompts[65] = "User: Why do we use batches in training?\nBot: "
responses[65] = "Batches allow the model to learn from groups of examples efficiently.\n"
prompts[66] = "User: How many times does an epoch run?\nBot: "
responses[66] = "An epoch runs once through the entire training dataset.\n"
prompts[67] = "User: Why is regularization used?\nBot: "
responses[67] = "Regularization is used to prevent the model from overfitting.\n"
prompts[68] = "User: How does dropout help training?\nBot: "
responses[68] = "Dropout randomly disables neurons so the model generalizes better.\n"
prompts[69] = "User: What is the future of AI search?\nBot: "
responses[69] = "The future of AI search is more accurate semantic and multimodal retrieval.\n"

# NOTE: If slm2_fit_vocab changes token IDs, preserve the resulting tokenizer
# and ensure the trained model uses that exact tokenizer at inference.
slm2_fit_vocab(ai2, prompts)
slm2_fit_vocab(ai2, responses)

sft_loss = engine_train_sft_pairs(ai2, prompts, responses, 65, 1)
print("SFT Loss:")
print(sft_loss)

save_model("stage2_sft.cbxm", ai2.model.params, "COBNOX_SFT")
tokenizer_save(ai2.tokenizer, "stage2_tok.json")

# 4. Load the SFT model and its matching tokenizer for chat
ai4 = create_slm2_engine(512, 64, 128, 0.002)
load_model("stage2_sft.cbxm", ai4.model.params)
ai4.tokenizer = tokenizer_load("stage2_tok.json")

print("Model Loaded")

running = 1
while running == 1 do
    q = input("User: ")
    if q == "stop chat" do
        running = 0
    otherwise do
        if length(q) > 0 do
            reply = engine2_chat(ai4, q)
            print("\nBot: ", reply)
        end
    end
end
```

## Notes and Known Issues

- The code uses Cobnox-specific functions, so syntax must be verified with the installed Cobnox engine.
- In the original code, the final loop loaded the model into `ai4` but called `engine2_chat(ai, q)`. The cleaned-up example uses `ai4`.
- The original loop used `running == 0`, which appears to compare rather than assign. The cleaned-up example uses `running = 0`; confirm the exact assignment syntax supported by Cobnox.
- The tokenizer should match the vocabulary and token-ID mapping used during SFT. This version saves `stage2_tok.json` after SFT and loads it for inference, but verify that `slm2_fit_vocab` is supported in this order by your engine.
- The losses and sample output above are the results reported by the project owner, not independently reproduced in this document.
- The approximate two-minute runtime is environment-dependent.
