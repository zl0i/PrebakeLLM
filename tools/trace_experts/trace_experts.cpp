// trace_experts: records which experts the router picks, per layer, for every
// generated token. The trace lets us simulate expert caches offline (LRU like
// llama.cpp, a cache that learns one token late, a static profiled hot set...)
// and compare their hit rates before writing any GPU code.
//
// Usage:
//   trace_experts <model.gguf> <prompts.json-as-lines.txt> <out.bin> [n_gen=256]
// The prompts file has one prompt per line, with "\n" escaped as the two
// characters backslash-n (written by bench/export_prompts.py).
//
// Output (little-endian int32):
//   header: magic 'EXTR', n_layer, n_expert_used
//   per prompt: prompt_index, n_tokens, then n_tokens * n_layer * n_expert_used expert ids
//
// How it hooks in: llama.cpp lets the caller observe any intermediate tensor
// of the compute graph through an "eval callback". The router output tensor
// is named "ffn_moe_topk-<layer>" (shape [n_expert_used, n_tokens], int32).

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct trace_state {
    int n_layer = 0;
    int n_used = 0;
    bool capture = false;             // only during single-token decode steps
    std::vector<int32_t> step;        // [n_layer][n_used] for the current token
    std::vector<int32_t> buf;         // all tokens of the current prompt
};

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * st = (trace_state *) ud;
    const char * pfx = "ffn_moe_topk-";
    if (std::strncmp(t->name, pfx, std::strlen(pfx)) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return st->capture;  // ask to observe only while decoding single tokens
    }
    const int il = std::atoi(t->name + std::strlen(pfx));
    if (il < 0 || il >= st->n_layer || t->ne[1] != 1) {
        return true;
    }
    std::vector<int32_t> ids(t->ne[0]);
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
    for (int k = 0; k < st->n_used && k < (int) ids.size(); ++k) {
        st->step[il * st->n_used + k] = ids[k];
    }
    return true;
}

static std::string unescape(const std::string & s) {
    std::string r;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == 'n') { r += '\n'; ++i; }
        else if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == '\\') { r += '\\'; ++i; }
        else r += s[i];
    }
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s model.gguf prompts.txt out.bin [n_gen]\n", argv[0]);
        return 1;
    }
    const int n_gen = argc > 4 ? std::atoi(argv[4]) : 256;

    std::vector<std::string> prompts;
    {
        std::ifstream f(argv[2], std::ios::binary);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) prompts.push_back(unescape(line));
        }
    }

    llama_backend_init();
    ggml_backend_load_all();

    // experts stay in RAM (like -cmoe): tracing speed does not matter, and this fits any VRAM
    llama_model_tensor_buft_override ov[2] = {
        { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "failed to load model\n"); return 1; }

    trace_state st;
    st.n_layer = llama_model_n_layer(model);
    st.n_used = 8;  // Qwen3.6-35B-A3B: 8 routed experts per token
    st.step.assign(st.n_layer * st.n_used, -1);

    auto cp = llama_context_default_params();
    cp.n_ctx = 20480;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create context\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::FILE * out = std::fopen(argv[3], "wb");
    const int32_t hdr[3] = { 0x52545845 /* 'EXTR' */, st.n_layer, st.n_used };
    std::fwrite(hdr, sizeof(hdr), 1, out);

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> toks(prompts[pi].size() + 16);
        int n = llama_tokenize(vocab, prompts[pi].c_str(), (int) prompts[pi].size(), toks.data(), (int) toks.size(), true, true);
        toks.resize(n);

        // prompt (prefill): not traced, batches of many tokens
        st.capture = false;
        for (int i = 0; i < n; i += cp.n_batch) {
            int m = std::min<int>(cp.n_batch, n - i);
            if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, m))) { std::fprintf(stderr, "decode failed\n"); return 1; }
        }
        st.buf.clear();
        llama_token cur = 0;
        for (int g = 0; g < n_gen; ++g) {
            // greedy: argmax of the last logits
            const float * lg = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
            cur = best;
            st.capture = true;
            std::fill(st.step.begin(), st.step.end(), -1);
            if (llama_decode(ctx, llama_batch_get_one(&cur, 1))) { std::fprintf(stderr, "decode failed\n"); return 1; }
            st.buf.insert(st.buf.end(), st.step.begin(), st.step.end());
        }
        const int32_t rec[2] = { (int32_t) pi, n_gen };
        std::fwrite(rec, sizeof(rec), 1, out);
        std::fwrite(st.buf.data(), sizeof(int32_t), st.buf.size(), out);
        std::fflush(out);
        std::fprintf(stderr, "prompt %zu: %d prompt tokens, %d generated\n", pi, n, n_gen);
    }
    std::fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
