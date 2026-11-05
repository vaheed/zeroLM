/*
 * ZeroLM Micro - a zero-training, hand-weighted recurrent neural language model
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 *
 * This is an intentionally constrained research prototype, not a general LLM.
 * It uses a manually weighted linear intent encoder and a sparse recurrent
 * decoder whose states emit tokens from a tiny, hand-authored phrase grammar.
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_INTENTS 32
#define MAX_FEATURES 512
#define MAX_STATES 2048
#define MAX_TEXT 4096
#define MAX_LINE 1024
#define MAX_TOKEN 128
#define MAX_STEPS 256
#define MAX_VOCAB 1024

#define DEFAULT_MODEL "weights.zlm"
#define W_ACTIVE 8.0
#define W_INACTIVE (-8.0)


#define MAX_SNAP_WORD 31
#define MAX_SNAP_WORDS 2048
#define MIN_SNAP_LEN 6

typedef struct {
    int intent;
    double weight;
    char phrase[192];
    char key[192]; /* phrase after stem_text; this is what is matched */
} Feature;

typedef struct {
    int next;
    char token[MAX_TOKEN];
} State;

typedef struct {
    int intents;
    int state_count;
    int feature_count;
    Feature features[MAX_FEATURES];
    int snap_count;
    char snap_words[MAX_SNAP_WORDS][MAX_SNAP_WORD + 1];
    State states[MAX_STATES];
    int start[MAX_INTENTS];
    char intent_name[MAX_INTENTS][48];
    int has_intent_name[MAX_INTENTS];
    int has_start[MAX_INTENTS];
    int has_state[MAX_STATES];
    int vocab_count;
    char vocab[MAX_VOCAB][MAX_TOKEN];
    int state_vocab[MAX_STATES];
} Model;

static void stem_text(const char *src, char *dst, size_t cap);
static void add_snap_words(Model *m, const char *key);

static void trim_newline(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = '\0';
}

static void softmax(double *values, int n) {
    if (n <= 0) return;
    double maxv = values[0];
    for (int i = 1; i < n; ++i) if (values[i] > maxv) maxv = values[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        values[i] = exp(values[i] - maxv);
        sum += values[i];
    }
    if (sum <= 0.0) {
        for (int i = 0; i < n; ++i) values[i] = 1.0 / n;
        return;
    }
    for (int i = 0; i < n; ++i) values[i] /= sum;
}

static int argmax(const double *values, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (values[i] > values[best]) best = i;
    return best;
}

/* Normalize English input by folding ASCII case and replacing punctuation
 * with spaces. Non-ASCII bytes are preserved but are not part of the English
 * feature vocabulary shipped with this project. */
static void normalize_text(const char *src, char *dst, size_t cap) {
    size_t j = 0;
    int previous_space = 1;
    for (size_t i = 0; src[i] && j + 2 < cap; ++i) {
        unsigned char c = (unsigned char)src[i];
        if (c < 0x80) {
            if (isalnum(c) || c == '_' || c == '-') {
                dst[j++] = (char)tolower(c);
                previous_space = 0;
            } else if (isspace(c) || ispunct(c)) {
                if (!previous_space && j + 1 < cap) dst[j++] = ' ';
                previous_space = 1;
            } else {
                dst[j++] = (char)c;
                previous_space = 0;
            }
        } else {
            dst[j++] = src[i];
            previous_space = 0;
        }
    }
    while (j && dst[j - 1] == ' ') --j;
    dst[j] = '\0';
}

static int load_model(const char *path, Model *m, char *err, size_t errcap) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, errcap, "cannot open model file: %s", path);
        return 0;
    }
    memset(m, 0, sizeof(*m));
    for (int i = 0; i < MAX_INTENTS; ++i) m->start[i] = -1;

    char line[MAX_LINE];
    int header_seen = 0;
    int line_no = 0;
    while (fgets(line, sizeof(line), fp)) {
        ++line_no;
        trim_newline(line);
        if (line[0] == '\0' || line[0] == '#') continue;

        if (!header_seen) {
            int version = 0;
            if (sscanf(line, "ZEROLM %d", &version) != 1 || version != 1) {
                snprintf(err, errcap, "invalid model header at line %d", line_no);
                fclose(fp);
                return 0;
            }
            header_seen = 1;
            continue;
        }

        if (strncmp(line, "META ", 5) == 0) {
            int intents = 0, states = 0;
            if (sscanf(line + 5, "%d %d", &intents, &states) != 2 ||
                intents < 1 || intents > MAX_INTENTS ||
                states < 1 || states > MAX_STATES) {
                snprintf(err, errcap, "invalid META at line %d", line_no);
                fclose(fp);
                return 0;
            }
            m->intents = intents;
            m->state_count = states;
        } else if (strncmp(line, "INTENT ", 7) == 0) {
            int id = -1;
            char name[48] = {0};
            if (sscanf(line + 7, "%d %47s", &id, name) != 2 || id < 0 || id >= MAX_INTENTS) {
                snprintf(err, errcap, "invalid INTENT at line %d", line_no);
                fclose(fp);
                return 0;
            }
            strcpy(m->intent_name[id], name);
            m->has_intent_name[id] = 1;
        } else if (strncmp(line, "FEATURE ", 8) == 0) {
            int intent = -1;
            double weight = 0.0;
            char phrase[192] = {0};
            if (m->feature_count >= MAX_FEATURES ||
                sscanf(line + 8, "%d %lf %191[^\n]", &intent, &weight, phrase) != 3 ||
                intent < 0 || intent >= MAX_INTENTS) {
                snprintf(err, errcap, "invalid FEATURE at line %d", line_no);
                fclose(fp);
                return 0;
            }
            Feature *f = &m->features[m->feature_count++];
            f->intent = intent;
            f->weight = weight;
            snprintf(f->phrase, sizeof(f->phrase), "%s", phrase);
            stem_text(f->phrase, f->key, sizeof(f->key));
        } else if (strncmp(line, "START ", 6) == 0) {
            int intent = -1, state = -1;
            if (sscanf(line + 6, "%d %d", &intent, &state) != 2 ||
                intent < 0 || intent >= MAX_INTENTS || state < 0 || state >= MAX_STATES) {
                snprintf(err, errcap, "invalid START at line %d", line_no);
                fclose(fp);
                return 0;
            }
            m->start[intent] = state;
            m->has_start[intent] = 1;
        } else if (strncmp(line, "NODE ", 5) == 0) {
            int id = -1, next = -1;
            char token[MAX_TOKEN] = {0};
            if (sscanf(line + 5, "%d %d %127s", &id, &next, token) != 3 ||
                id < 0 || id >= MAX_STATES || next < -1 || next >= MAX_STATES) {
                snprintf(err, errcap, "invalid NODE at line %d", line_no);
                fclose(fp);
                return 0;
            }
            m->states[id].next = next;
            snprintf(m->states[id].token, sizeof(m->states[id].token), "%s", token);
            m->has_state[id] = 1;
        } else {
            snprintf(err, errcap, "unknown record at line %d", line_no);
            fclose(fp);
            return 0;
        }
    }
    fclose(fp);

    if (!header_seen || m->intents < 1 || m->state_count < 1 || m->feature_count < 1) {
        snprintf(err, errcap, "model is incomplete (missing header, META, or features)");
        return 0;
    }
    for (int i = 0; i < m->intents; ++i) {
        if (!m->has_start[i] || !m->has_intent_name[i]) {
            snprintf(err, errcap, "model missing START or INTENT for id %d", i);
            return 0;
        }
    }
    for (int i = 0; i < m->state_count; ++i) {
        if (!m->has_state[i]) {
            snprintf(err, errcap, "model missing NODE %d", i);
            return 0;
        }
    }

    for (int f = 0; f < m->feature_count; ++f) add_snap_words(m, m->features[f].key);

    /* Build the output vocabulary once at load time. */
    for (int state = 0; state < m->state_count; ++state) {
        int vocab_id = -1;
        for (int v = 0; v < m->vocab_count; ++v) {
            if (strcmp(m->vocab[v], m->states[state].token) == 0) {
                vocab_id = v;
                break;
            }
        }
        if (vocab_id < 0) {
            if (m->vocab_count >= MAX_VOCAB) {
                snprintf(err, errcap, "vocabulary exceeds maximum size %d", MAX_VOCAB);
                return 0;
            }
            vocab_id = m->vocab_count++;
            snprintf(m->vocab[vocab_id], MAX_TOKEN, "%s", m->states[state].token);
        }
        m->state_vocab[state] = vocab_id;
    }
    return 1;
}


/* ---- word-form tolerance: light stemming and spell-snapping (mirrors the Rust port) ---- */

static int is_lower_alpha(const char *w, size_t n) {
    if (n == 0) return 0;
    for (size_t i = 0; i < n; ++i)
        if (w[i] < 'a' || w[i] > 'z') return 0;
    return 1;
}

static int ends_with(const char *s, size_t n, const char *suf) {
    size_t k = strlen(suf);
    return n >= k && memcmp(s + n - k, suf, k) == 0;
}

static size_t undouble(char *s, size_t n) {
    if (n >= 2 && s[n - 1] == s[n - 2] && s[n - 1] != 'l' && s[n - 1] != 's' && s[n - 1] != 'z') return n - 1;
    return n;
}

/* Stems one word into out (capacity n + 1) and returns its length. */
static size_t stem_word(const char *w, size_t n, char *out) {
    memcpy(out, w, n);
    if (!is_lower_alpha(w, n)) return n;
    size_t len = n;
    if (len > 4 && ends_with(out, len, "ies")) {
        len -= 3;
        out[len++] = 'y';
    } else if (len > 3 && ends_with(out, len, "es") &&
               (out[len - 3] == 's' || out[len - 3] == 'x' || out[len - 3] == 'z' ||
                ends_with(out, len, "ches") || ends_with(out, len, "shes"))) {
        len -= 2;
    } else if (len > 3 && ends_with(out, len, "s") && !ends_with(out, len, "ss") &&
               !ends_with(out, len, "us") && !ends_with(out, len, "is")) {
        len -= 1;
    }
    if (len > 5 && ends_with(out, len, "ing")) {
        len -= 3;
        len = undouble(out, len);
    } else if (len > 4 && ends_with(out, len, "ed")) {
        len -= 2;
        len = undouble(out, len);
    }
    if (len > 4 && out[len - 1] == 'e') len -= 1;
    return len;
}

/* Stems every space-separated word of src into dst (capacity cap). */
static void stem_text(const char *src, char *dst, size_t cap) {
    size_t o = 0, i = 0, n = strlen(src);
    for (;;) {
        size_t j = i;
        while (j < n && src[j] != ' ') ++j;
        char tmp[MAX_TEXT + 1];
        size_t wl = j - i;
        if (wl > MAX_TEXT) wl = MAX_TEXT;
        size_t sl = stem_word(src + i, wl, tmp);
        if (o + sl + 2 >= cap) break;
        memcpy(dst + o, tmp, sl);
        o += sl;
        if (j >= n) break;
        dst[o++] = ' ';
        i = j + 1;
    }
    dst[o] = '\0';
}

static int within_one_edit(const char *a, size_t la, const char *b, size_t lb) {
    if (la == lb) {
        size_t i = 0;
        while (i < la && a[i] == b[i]) ++i;
        if (i == la) return 1;
        if (memcmp(a + i + 1, b + i + 1, la - i - 1) == 0) return 1;
        return i + 1 < la && a[i] == b[i + 1] && a[i + 1] == b[i] && memcmp(a + i + 2, b + i + 2, la - i - 2) == 0;
    }
    if (la == lb + 1) {
        size_t i = 0;
        while (i < lb && a[i] == b[i]) ++i;
        return memcmp(a + i + 1, b + i, la - i - 1) == 0;
    }
    if (lb == la + 1) return within_one_edit(b, lb, a, la);
    return 0;
}

static void add_snap_words(Model *m, const char *key) {
    size_t n = strlen(key), i = 0;
    for (;;) {
        size_t j = i;
        while (j < n && key[j] != ' ') ++j;
        size_t wl = j - i;
        if (is_lower_alpha(key + i, wl) && wl <= MAX_SNAP_WORD && m->snap_count < MAX_SNAP_WORDS) {
            int seen = 0;
            for (int k = 0; k < m->snap_count && !seen; ++k)
                seen = strlen(m->snap_words[k]) == wl && memcmp(m->snap_words[k], key + i, wl) == 0;
            if (!seen) {
                memcpy(m->snap_words[m->snap_count], key + i, wl);
                m->snap_words[m->snap_count][wl] = '\0';
                ++m->snap_count;
            }
        }
        if (j >= n) break;
        i = j + 1;
    }
}

/* Stem the input, and replace an unknown word of >= MIN_SNAP_LEN letters by the single phrase
 * word one edit away that starts with the same letter. */
static void prepare_text(const Model *m, const char *src, char *dst, size_t cap) {
    size_t o = 0, i = 0, n = strlen(src);
    for (;;) {
        size_t j = i;
        while (j < n && src[j] != ' ') ++j;
        char tmp[MAX_TEXT + 1];
        size_t wl = j - i;
        if (wl > MAX_TEXT) wl = MAX_TEXT;
        size_t sl = stem_word(src + i, wl, tmp);
        const char *emit = tmp;
        size_t el = sl;
        if (sl >= MIN_SNAP_LEN && is_lower_alpha(tmp, sl)) {
            int known = 0, count = 0, hit = -1;
            for (int k = 0; k < m->snap_count; ++k) {
                size_t cl = strlen(m->snap_words[k]);
                if (cl == sl && memcmp(m->snap_words[k], tmp, sl) == 0) known = 1;
                if (m->snap_words[k][0] == tmp[0] && within_one_edit(tmp, sl, m->snap_words[k], cl)) {
                    ++count;
                    hit = k;
                }
            }
            if (!known && count == 1) {
                emit = m->snap_words[hit];
                el = strlen(emit);
            }
        }
        if (o + el + 2 >= cap) break;
        memcpy(dst + o, emit, el);
        o += el;
        if (j >= n) break;
        dst[o++] = ' ';
        i = j + 1;
    }
    dst[o] = '\0';
}

static int is_word_byte(unsigned char c) {
    return (c >= 0x80) || isalnum(c) || c == '_' || c == '-';
}

static int contains_phrase(const char *text, const char *phrase) {
    size_t n = strlen(phrase);
    const char *p = text;
    while ((p = strstr(p, phrase)) != NULL) {
        int left_ok = (p == text) || !is_word_byte((unsigned char)p[-1]);
        int right_ok = !is_word_byte((unsigned char)p[n]);
        if (left_ok && right_ok) return 1;
        ++p;
    }
    return 0;
}

static void classify(const Model *m, const char *input, double *prob, int *selected,
                     double *selected_score) {
    double logits[MAX_INTENTS] = {0};
    char prepared[MAX_TEXT * 2];
    prepare_text(m, input, prepared, sizeof(prepared));
    for (int f = 0; f < m->feature_count; ++f) {
        if (contains_phrase(prepared, m->features[f].key))
            logits[m->features[f].intent] += m->features[f].weight;
    }
    for (int i = 0; i < m->intents; ++i) prob[i] = logits[i];
    softmax(prob, m->intents);
    *selected = argmax(logits, m->intents);
    *selected_score = logits[*selected];
}

/* One-hot-like hidden state is initialized by W_start (implicit -8 everywhere,
 * +8 at the selected intent's start state). */
static void init_hidden(const Model *m, int intent, double *h) {
    double logits[MAX_STATES];
    int start = m->start[intent];
    for (int i = 0; i < m->state_count; ++i)
        logits[i] = (i == start) ? W_ACTIVE : W_INACTIVE;
    softmax(logits, m->state_count);
    memcpy(h, logits, (size_t)m->state_count * sizeof(double));
}

/*
 * Sparse recurrent update, algebraically equal to the dense matrix.
 * Every source row has +8 at its single declared successor and -8 elsewhere:
 *   score[target] = -8 * sum(h) + 16 * incoming_mass[target].
 * The first term is constant across targets and cancels in softmax, so only
 * incoming_mass is needed. This changes O(states^2) work to O(states).
 */
static void recurrent_step(const Model *m, const double *h, double *next_h) {
    for (int target = 0; target < m->state_count; ++target) next_h[target] = 0.0;
    for (int source = 0; source < m->state_count; ++source) {
        int target = m->states[source].next;
        if (target >= 0) next_h[target] += 16.0 * h[source];
    }
    softmax(next_h, m->state_count);
}

/* Sparse output projection. The common -8 offset cancels in softmax. */
static int output_token_id(const Model *m, const double *h, char *out, size_t outcap,
                           double *confidence) {
    double probs[MAX_VOCAB] = {0.0};
    for (int state = 0; state < m->state_count; ++state)
        probs[m->state_vocab[state]] += h[state];
    for (int v = 0; v < m->vocab_count; ++v) probs[v] *= 16.0;
    softmax(probs, m->vocab_count);
    int best = argmax(probs, m->vocab_count);
    size_t copy_len = strlen(m->vocab[best]);
    if (copy_len >= outcap) copy_len = outcap - 1;
    memcpy(out, m->vocab[best], copy_len);
    out[copy_len] = '\0';
    if (confidence) *confidence = probs[best];
    return best;
}

static int generate(const Model *m, int intent, int print_tokens, char *response,
                    size_t response_cap, double *avg_confidence) {
    double h[MAX_STATES], next_h[MAX_STATES];
    init_hidden(m, intent, h);
    response[0] = '\0';
    size_t used = 0;
    double confidence_sum = 0.0;
    int emitted = 0;

    for (int step = 0; step < MAX_STEPS; ++step) {
        char token[MAX_TOKEN];
        double out_conf = 0.0;
        (void)output_token_id(m, h, token, sizeof(token), &out_conf);
        if (strcmp(token, "<eos>") == 0) break;

        size_t token_len = strlen(token);
        size_t needed = token_len + (emitted ? 1 : 0);
        if (used + needed + 1 >= response_cap) break;
        if (emitted) response[used++] = ' ';
        memcpy(response + used, token, token_len);
        used += token_len;
        response[used] = '\0';
        if (print_tokens) printf("%s%s", emitted ? " " : "", token);
        confidence_sum += out_conf;
        ++emitted;

        recurrent_step(m, h, next_h);
        memcpy(h, next_h, (size_t)m->state_count * sizeof(double));
    }
    if (print_tokens) putchar('\n');
    if (avg_confidence) *avg_confidence = emitted ? confidence_sum / emitted : 0.0;
    return emitted;
}

static int answer(const Model *m, const char *raw, int verbose) {
    char normalized[MAX_TEXT];
    normalize_text(raw, normalized, sizeof(normalized));
    double probs[MAX_INTENTS] = {0};
    int intent = 0;
    double score = 0.0;
    classify(m, normalized, probs, &intent, &score);

    /* No matched feature: intentionally abstain instead of fabricating. */
    if (score < 0.75) intent = m->intents - 1;

    char response[MAX_TEXT];
    double confidence = 0.0;
    int count = generate(m, intent, 0, response, sizeof(response), &confidence);
    if (verbose) {
        printf("[intent=%s score=%.2f emitted_tokens=%d]\n",
               m->intent_name[intent], score, count);
    }
    puts(response);
    return intent;
}

static void print_info(const Model *m) {
    long dense_weights = (long)m->intents * m->feature_count +
                         (long)m->state_count * m->intents +
                         (long)m->state_count * m->state_count +
                         (long)m->vocab_count * m->state_count;
    size_t model_struct_bytes = sizeof(*m);
    printf("ZeroLM v0.2.0\n");
    printf("Architecture: manual phrase-to-intent projection + sparse recurrent decoder\n");
    printf("Training: none | pretrained weights: none | runtime: CPU only\n");
    printf("Dimensions: intents=%d features=%d hidden_states=%d vocabulary=%d\n",
           m->intents, m->feature_count, m->state_count, m->vocab_count);
    printf("Dense-equivalent weight slots: %ld (fixed, not trainable)\n", dense_weights);
    printf("Model data structure upper bound: %zu bytes (%.2f KiB)\n",
           model_struct_bytes, model_struct_bytes / 1024.0);
    printf("Decoder: greedy autoregressive token generation, max %d tokens\n", MAX_STEPS);
    printf("Important: this is a constrained phrase model, not a general-purpose LLM.\n");
}
static int run_self_tests(const Model *m) {
    struct Test { const char *input; const char *expected_intent; } tests[] = {
        {"hello", "greeting"},
        {"good morning", "greeting"},
        {"who are you", "identity"},
        {"what can you do", "capabilities"},
        {"thank you", "thanks"},
        {"goodbye", "farewell"},
        {"how do I run ZeroLM", "help"},
        {"explain the architecture", "architecture"},
        {"show the weight file", "weights"},
        {"does it use training", "no_training"},
        {"what are its limitations", "limitations"},
        {"does it have memory", "memory"},
        {"make a plan step by step", "planning"},
        {"write code", "coding"},
        {"solve this math equation", "math"},
        {"explain this term", "explain"},
        {"run the benchmark pipeline", "testing"},
        {"tell me about weight-space programming", "inspiration"},
        {"what is the weather in Paris", "unknown"},
        {"who won the match yesterday", "unknown"}
    };    int total = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;
    puts("Running zero-training smoke tests:");
    for (int i = 0; i < total; ++i) {
        char normalized[MAX_TEXT];
        double probs[MAX_INTENTS] = {0}, score = 0.0;
        int intent = 0;
        normalize_text(tests[i].input, normalized, sizeof(normalized));
        classify(m, normalized, probs, &intent, &score);
        if (score < 0.75) intent = m->intents - 1;
        const char *got = m->intent_name[intent];
        int ok = strcmp(got, tests[i].expected_intent) == 0;
        printf("  %s  input=\"%s\" expected=%s got=%s\n",
               ok ? "PASS" : "FAIL", tests[i].input, tests[i].expected_intent, got);
        if (ok) ++passed;
    }
    printf("Result: %d/%d classifier tests passed\n", passed, total);

    puts("Checking recurrent decoder paths:");
    for (int intent = 0; intent < m->intents; ++intent) {
        char response[MAX_TEXT];
        double confidence = 0.0;
        int tokens = generate(m, intent, 0, response, sizeof(response), &confidence);
        int ok = tokens > 0 && strstr(response, "<eos>") == NULL;
        printf("  %s  intent=%s tokens=%d\n",
               ok ? "PASS" : "FAIL", m->intent_name[intent], tokens);
        if (ok) ++passed;
    }
    printf("Total: %d/%d checks passed\n", passed, total + m->intents);
    return passed == total + m->intents ? 0 : 1;
}

static void usage(const char *argv0) {
    printf("Usage: %s [--model FILE] [--info] [--self-test] [--verbose] [TEXT]\n", argv0);
    puts("Without TEXT, starts an interactive chat. Type /quit to exit.");
}

int main(int argc, char **argv) {
    const char *model_path = DEFAULT_MODEL;
    int info = 0, self_test = 0, verbose = 0;
    int text_start = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--info") == 0) info = 1;
        else if (strcmp(argv[i], "--self-test") == 0) self_test = 1;
        else if (strcmp(argv[i], "--verbose") == 0) verbose = 1;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else { text_start = i; break; }
    }

    Model *model = (Model *)calloc(1, sizeof(Model));
    if (!model) {
        fputs("error: not enough memory to allocate model\n", stderr);
        return 2;
    }
    char err[256];
    if (!load_model(model_path, model, err, sizeof(err))) {
        fprintf(stderr, "model error: %s\n", err);
        free(model);
        return 2;
    }

    if (info) print_info(model);
    if (info && !self_test && text_start < 0) {
        free(model);
        return 0;
    }
    if (self_test) {
        int result = run_self_tests(model);
        free(model);
        return result;
    }
    if (text_start >= 0) {
        char joined[MAX_TEXT] = {0};
        size_t used = 0;
        for (int i = text_start; i < argc; ++i) {
            size_t n = strlen(argv[i]);
            if (used + n + 2 >= sizeof(joined)) break;
            if (used) joined[used++] = ' ';
            memcpy(joined + used, argv[i], n);
            used += n;
            joined[used] = '\0';
        }
        answer(model, joined, verbose);
        free(model);
        return 0;
    }

    puts("ZeroLM v0.2.0 — type /quit to exit.");
    char input[MAX_TEXT];
    for (;;) {
        fputs("you> ", stdout);
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        trim_newline(input);
        if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) break;
        if (input[0] == '\0') continue;
        fputs("bot> ", stdout);
        answer(model, input, verbose);
    }
    free(model);
    return 0;
}
