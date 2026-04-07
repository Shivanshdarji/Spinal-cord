/*
 * SpinalCord LLM — C++ Inference Engine
 * =======================================
 * This program uses llama.cpp to:
 *   1. Load a GGUF model (your converted SpinalCord Draft model)
 *   2. Walk through all the transformer layers
 *   3. *** Print confidence/logit statistics at each layer ***
 *      (This is the first "spark" of the Spinal Cord invention!)
 *
 * Build instructions:
 *   cmake -B build -DLLAMA_CUDA=ON
 *   cmake --build build --config Release
 *   ./spinalcord path/to/model.gguf "Your input prompt here"
 *
 * Author: Shivansh Darji | AppDice
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

// llama.cpp headers (must be after cloning and building llama.cpp)
#include "llama.h"
#include "common.h"


// ─────────────────────────────────────────────────────────────────────────────
// UTILITY: Calculate entropy of a probability distribution
// Entropy = measure of "uncertainty" or "confidence"
// Low entropy = model is very confident (spinal cord reflex working)
// High entropy = model is uncertain (brain needs to think)
// ─────────────────────────────────────────────────────────────────────────────
float compute_entropy(const float* logits, int vocab_size) {
    // Find max for numerical stability (log-sum-exp trick)
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Compute softmax probabilities
    float sum = 0.0f;
    std::vector<float> probs(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    // Compute entropy: H = -sum(p * log(p))
    float entropy = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        if (probs[i] > 1e-9f) {
            entropy -= probs[i] * logf(probs[i]);
        }
    }
    return entropy;
}


// ─────────────────────────────────────────────────────────────────────────────
// UTILITY: Get top-k token IDs and their probabilities
// ─────────────────────────────────────────────────────────────────────────────
struct TokenProb {
    llama_token id;
    float prob;
};

std::vector<TokenProb> get_top_k(const float* logits, int vocab_size, int k = 5) {
    std::vector<std::pair<float, int>> scored(vocab_size);
    
    // Find max for softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        scored[i] = {expf(logits[i] - max_logit), i};
        sum += scored[i].first;
    }
    
    // Normalize
    for (auto& [prob, id] : scored) prob /= sum;
    
    // Sort descending
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<TokenProb> result(k);
    for (int i = 0; i < k; ++i) {
        result[i] = {scored[i].second, scored[i].first};
    }
    return result;
}


// ─────────────────────────────────────────────────────────────────────────────
// SPINAL CORD LAYER PROBE
// This is the "spark" — we print confidence at each transformer layer.
// In a real implementation, you'd hook into llama_decode's internal
// layer callbacks. Here we demonstrate with final logits.
// ─────────────────────────────────────────────────────────────────────────────
void spinalcord_layer_probe(
    llama_context* ctx,
    const float* logits,
    int n_vocab,
    int layer_idx,
    int n_layers
) {
    float entropy = compute_entropy(logits, n_vocab);
    
    // Confidence = 1 - normalized entropy
    // max entropy for vocab_size = log(vocab_size)
    float max_entropy = logf((float)n_vocab);
    float confidence  = 1.0f - (entropy / max_entropy);
    
    // Print the Spinal Cord "confidence bar"
    int bar_width = 30;
    int filled    = (int)(confidence * bar_width);
    
    printf("[Layer %2d/%2d] ", layer_idx + 1, n_layers);
    printf("Confidence: %5.1f%% [", confidence * 100.0f);
    for (int i = 0; i < bar_width; ++i) {
        printf(i < filled ? "█" : "░");
    }
    printf("] Entropy: %.3f", entropy);
    
    // 🚨 SPINAL CORD DECISION POINT
    // This is where your invention triggers:
    // If confidence is HIGH early → don't need all layers!
    if (layer_idx <= 3 && confidence > 0.90f) {
        printf(" ← ⚡ REFLEX! (Early exit possible)");
    }
    printf("\n");
}


// ─────────────────────────────────────────────────────────────────────────────
// SPECULATIVE DRAFT GENERATION
// Uses the model in "draft mode" — fast greedy/sample generation
// ─────────────────────────────────────────────────────────────────────────────
struct DraftResult {
    std::vector<llama_token> tokens;
    std::vector<float>       confidences;
    float                    avg_confidence;
    long long                time_ms;
};

DraftResult generate_draft_tokens(
    llama_context* ctx,
    llama_model*   model,
    const std::vector<llama_token>& context_tokens,
    int gamma = 4,
    int n_vocab = 32000
) {
    DraftResult result;
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::vector<llama_token> current = context_tokens;
    
    for (int step = 0; step < gamma; ++step) {
        // Decode current sequence
        llama_batch batch = llama_batch_get_one(
            current.data(), current.size()
        );
        
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[Error] llama_decode failed at draft step %d\n", step);
            break;
        }
        
        // Get logits of the last token
        const float* logits = llama_get_logits_ith(ctx, -1);
        
        // Compute confidence at this step
        float entropy    = compute_entropy(logits, n_vocab);
        float max_entropy = logf((float)n_vocab);
        float confidence  = 1.0f - (entropy / max_entropy);
        
        result.confidences.push_back(confidence);
        
        // Greedy sampling (fastest) - using our custom argmax
        auto top_tokens = get_top_k(logits, n_vocab, 1);
        llama_token next_token = top_tokens[0].id;
        
        result.tokens.push_back(next_token);
        current.push_back(next_token);
        
        const struct llama_vocab * vocab = llama_model_get_vocab(model);
        printf("[Draft Step %d] Token: %-15s | Confidence: %.1f%%\n",
               step + 1,
               llama_vocab_get_text(vocab, next_token),
               confidence * 100.0f);
        
        // EOS check
        if (llama_vocab_is_eog(vocab, next_token)) {
            break;
        }
    }
    
    // Compute average confidence
    float avg = 0.0f;
    for (float c : result.confidences) avg += c;
    result.avg_confidence = result.confidences.empty() ? 0.0f 
                          : avg / result.confidences.size();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start
    ).count();
    
    return result;
}


// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model.gguf> \"<prompt>\"\n", argv[0]);
        printf("Example: %s models/phi-3-mini.gguf \"The spinal cord\"\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt     = argv[2];

    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║       🧠 SpinalCord LLM — Inference Engine          ║\n");
    printf("║            AppDice | Shivansh Darji                 ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // ── Initialize model ────────────────────────────────────────────────────
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 35;   // offload 35 layers to RTX 2050

    printf("[Init] Loading model: %s\n", model_path);
    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "[Error] Failed to load model: %s\n", model_path);
        return 1;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    printf("[Init] Model loaded! Layers: %d, Vocab: %d\n",
           llama_model_n_layer(model),
           llama_vocab_n_tokens(vocab));

    // ── Initialize context ───────────────────────────────────────────────────
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = 2048;
    ctx_params.n_batch  = 512;
    ctx_params.n_threads = 4;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "[Error] Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    int n_vocab  = llama_vocab_n_tokens(vocab);
    int n_layers = llama_model_n_layer(model);

    // ── Tokenize prompt ──────────────────────────────────────────────────────
    std::vector<llama_token> tokens(512);
    int n_tokens = llama_tokenize(
        vocab, prompt, strlen(prompt),
        tokens.data(), tokens.size(),
        /* add_bos */ true, /* special */ false
    );
    tokens.resize(n_tokens);

    printf("\n[Prompt] \"%s\"\n", prompt);
    printf("[Prompt] Token count: %d\n\n", n_tokens);

    // ── Run Initial Forward Pass + Layer Probe ────────────────────────────────
    printf("━━━ 🔬 SPINAL CORD LAYER ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Watching confidence build up through the transformer layers...\n\n");
    
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[Error] Initial llama_decode failed\n");
        return 1;
    }

    // Get final logits (from the last position)
    const float* final_logits = llama_get_logits_ith(ctx, -1);

    // Simulate layer-by-layer analysis using the final logits
    // (In a full implementation, you'd hook into intermediate layer outputs)
    printf("NOTE: Full layer hooks require custom llama.cpp modification.\n");
    printf("      Below shows final-layer analysis. See llama-model.cpp to add hooks.\n\n");
    
    spinalcord_layer_probe(ctx, final_logits, n_vocab, n_layers - 1, n_layers);

    // ── Show Top Predictions ─────────────────────────────────────────────────
    printf("\n━━━ 🎯 TOP TOKEN PREDICTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    auto top_tokens = get_top_k(final_logits, n_vocab, 5);
    for (int i = 0; i < (int)top_tokens.size(); ++i) {
        printf("  Top %d: %-20s | Prob: %.2f%%\n",
               i + 1, llama_vocab_get_text(vocab, top_tokens[i].id), top_tokens[i].prob * 100.0f);
    }

    // ── Speculative Draft Generation ─────────────────────────────────────────
    printf("\n━━━ ⚡ SPINAL CORD SPECULATIVE GENERATION (gamma=4) ━━━━━━━━━━\n");
    printf("The Draft Model generating next 4 tokens speculatively...\n\n");
    
    DraftResult draft = generate_draft_tokens(ctx, model, tokens, 4, n_vocab);
    
    printf("\n[Draft Summary]\n");
    printf("  Tokens generated:    %zu\n", draft.tokens.size());
    printf("  Average confidence:  %.1f%%\n", draft.avg_confidence * 100.0f);
    printf("  Time taken:          %lld ms\n", draft.time_ms);
    
    // Decode and print the drafted sequence
    printf("  Drafted text:        \"");
    for (llama_token tok : draft.tokens) {
        printf("%s", llama_vocab_get_text(vocab, tok));
    }
    printf("\"\n");

    // ── Final Summary ────────────────────────────────────────────────────────
    printf("\n━━━ 📊 SPINALCORD REPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    float entropy     = compute_entropy(final_logits, n_vocab);
    float max_entropy = logf((float)n_vocab);
    float confidence  = 1.0f - (entropy / max_entropy);
    
    printf("  Model:              %s\n", model_path);
    printf("  Prompt:             \"%s\"\n", prompt);
    printf("  Final Confidence:   %.1f%%\n", confidence * 100.0f);
    printf("  Final Entropy:      %.4f nats (max=%.2f)\n", entropy, max_entropy);
    printf("  Speculative Speed:  %zu tokens in %lld ms (%.1f tok/s)\n",
           draft.tokens.size(), draft.time_ms,
           draft.time_ms > 0 ? draft.tokens.size() * 1000.0f / draft.time_ms : 0.0f);
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║  🚀 SpinalCord Engine complete. AppDice.     ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // ── Cleanup ───────────────────────────────────────────────────────────────
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
