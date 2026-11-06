//! Reasoning benchmark: compositional reasoning over taught primitives.
//!
//! Per seed a world is generated: a class taxonomy with exclusions, abilities
//! with exceptions, containment, comparative orders, event orders, counts
//! with relative counts, property rules, pronoun-introduced facts, and a few
//! injected contradictions. Statements are taught in random paraphrase forms
//! (some through "it"); questions are asked in other forms. Labels come from
//! the symbolic oracle. Derived categories keep only questions the lookup
//! baseline cannot answer (unseen combinations).
//!
//! Also measured: paraphrase consistency, counterfactual pairs (removing one
//! premise), provenance sufficiency (the reported premises alone reproduce
//! the answer), restart identity, causal ablations of each operator and of
//! the binding, recurrent-state, and retrieval pathways, and a hidden seed
//! whose questions and labels are never written out (only hashes).

use crate::online::Stat;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;
use zerolm_engine::semantic_state::{is_reserved, parse, Utterance, COMPARATIVES};
use zerolm_reasoner::lookup::Lookup;
use zerolm_reasoner::oracle::Oracle;
use zerolm_reasoner::{Observed, Ops, Pathway, Reasoner, ReasonerConfig, Verdict};
use zerolm_tensor::{shuffle, Rng};

pub const SPEC: &str = "model/specs/reasoning.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldSpec {
    pub roots: usize,
    pub children: usize,
    pub grandchildren: usize,
    pub entities: usize,
    pub exclusions: usize,
    pub acts: usize,
    pub regions: usize,
    pub towns_per_region: usize,
    pub sites_per_town: usize,
    pub located_entities: usize,
    pub dims: usize,
    pub order_len: usize,
    pub event_chains: usize,
    pub chain_len: usize,
    pub things: usize,
    pub count_chain: usize,
    pub props: usize,
    pub rule_chain: usize,
    pub contradictions: usize,
    pub pronoun_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub agreement_min: f64,
    pub category_agreement_min: f64,
    pub derived_agreement_min: f64,
    pub lookup_gap_min: f64,
    pub paraphrase_consistency_min: f64,
    pub counterfactual_min: f64,
    pub provenance_min: f64,
    pub ablation_drop_min: f64,
    pub operator_collateral_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub version: u32,
    pub config: ReasonerConfig,
    pub world: WorldSpec,
    pub per_category: usize,
    pub counterfactual_pairs: usize,
    pub provenance_checks: usize,
    pub pilot_seeds: Vec<u64>,
    pub main_seeds: Vec<u64>,
    pub hidden_salt: String,
    pub gates: Gates,
    pub gate_rationale: String,
}

impl Spec {
    pub fn load(root: &Path) -> Result<Spec, String> {
        let t = std::fs::read_to_string(root.join(SPEC)).map_err(|e| format!("{SPEC}: {e}"))?;
        let s: Spec = serde_json::from_str(&t).map_err(|e| format!("{SPEC}: {e}"))?;
        s.config.validate()?;
        if s.main_seeds.len() < 3 || s.main_seeds.iter().any(|x| s.pilot_seeds.contains(x)) {
            return Err("need >= 3 main seeds disjoint from pilot seeds".into());
        }
        if s.world.dims > COMPARATIVES.len() / 2 || s.world.roots < 2 || s.world.exclusions == 0 {
            return Err("bad world spec".into());
        }
        Ok(s)
    }
}

/// Categories and the operators/pathways they depend on.
pub const CATEGORIES: [&str; 20] = [
    "isa_direct",
    "isa_derived",
    "isa_excluded",
    "isa_unknown",
    "ability_inherited",
    "ability_exception",
    "ability_unknown",
    "location_derived",
    "location_unknown",
    "comparison_derived",
    "comparison_unknown",
    "temporal_derived",
    "temporal_unknown",
    "arithmetic_direct",
    "arithmetic_derived",
    "arithmetic_unknown",
    "conditional_derived",
    "conditional_unknown",
    "pronoun",
    "contradiction",
];

/// Categories an ablation is expected to damage (targets) and allowed to touch.
pub fn ablation_targets(name: &str) -> (&'static [&'static str], &'static [&'static str]) {
    match name {
        "class_inclusion" => (&["isa_derived"], &["isa_derived", "isa_excluded"]),
        "transitivity" => (&["location_derived"], &["location_derived"]),
        "attribute_inheritance" => (&["ability_inherited"], &["ability_inherited", "ability_exception"]),
        "comparison" => (&["comparison_derived"], &["comparison_derived"]),
        "arithmetic" => (&["arithmetic_derived"], &["arithmetic_derived"]),
        "temporal_ordering" => (&["temporal_derived"], &["temporal_derived"]),
        "negation" => (&["isa_excluded", "ability_exception", "contradiction"], &["isa_excluded", "ability_exception", "contradiction"]),
        "conditionals" => (&["conditional_derived"], &["conditional_derived"]),
        "recurrent_state" => (&["pronoun"], &CATEGORIES),
        "binding" => (&CATEGORIES_DERIVED, &CATEGORIES),
        "retrieval" => (&CATEGORIES_ANSWERED, &CATEGORIES),
        _ => (&[], &[]),
    }
}

const CATEGORIES_DERIVED: [&str; 7] =
    ["isa_derived", "ability_inherited", "location_derived", "comparison_derived", "temporal_derived", "arithmetic_derived", "conditional_derived"];
const CATEGORIES_ANSWERED: [&str; 10] = [
    "isa_direct",
    "isa_derived",
    "isa_excluded",
    "ability_inherited",
    "location_derived",
    "comparison_derived",
    "temporal_derived",
    "arithmetic_derived",
    "conditional_derived",
    "pronoun",
];

pub const PATHWAYS: [&str; 3] = ["binding", "recurrent_state", "retrieval"];

// ------------------------------------------------------------------ world

#[derive(Debug, Clone)]
pub struct Statement {
    /// Surface form as taught (may use "it").
    pub text: String,
    /// Same statement with the subject named (for rebuilding sub-worlds).
    pub canonical: String,
    pub via_pronoun: bool,
}

#[derive(Debug, Clone)]
pub struct Question {
    pub category: &'static str,
    pub text: String,
    /// A second surface form of the same question.
    pub alt: String,
}

pub struct World {
    pub statements: Vec<Statement>,
    pub questions: Vec<Question>,
}

struct Names {
    rng: Rng,
    used: BTreeSet<String>,
}

impl Names {
    fn fresh(&mut self, syl: usize) -> String {
        const C: &[u8] = b"bdfgklmnprtvz";
        const V: &[u8] = b"aeiou";
        loop {
            let mut s = String::new();
            for _ in 0..syl {
                s.push(C[self.rng.below(C.len())] as char);
                s.push(V[self.rng.below(V.len())] as char);
            }
            if self.rng.below(2) == 0 {
                s.push(C[self.rng.below(C.len())] as char);
            }
            if s.len() >= 4 && !s.ends_with('s') && !is_reserved(&s) && self.used.insert(s.clone()) {
                return s;
            }
        }
    }
}

fn art(w: &str) -> &'static str {
    if w.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    }
}

fn pick<'a>(rng: &mut Rng, forms: &'a [String]) -> &'a String {
    &forms[rng.below(forms.len())]
}

/// Builder of statements with paraphrase forms and optional pronoun subjects.
struct Teach<'a> {
    rng: &'a mut Rng,
    out: Vec<Statement>,
    pronoun_rate: f64,
}

impl Teach<'_> {
    /// `forms(subject)` gives surface forms; `pronoun_ok` allows "it".
    fn say(&mut self, subj: &str, forms: &dyn Fn(&str) -> Vec<String>, pronoun_ok: bool, prev_subject: &mut Option<String>) {
        let canonical = forms(subj)[0].clone();
        let use_it = pronoun_ok && prev_subject.as_deref() == Some(subj) && (self.rng.next_f32() as f64) < self.pronoun_rate;
        let text = if use_it { pick(self.rng, &forms("it")).clone() } else { pick(self.rng, &forms(subj)).clone() };
        self.out.push(Statement { text, canonical, via_pronoun: use_it });
        *prev_subject = Some(subj.to_string());
    }

    fn class(&mut self, forms: Vec<String>) {
        let text = pick(self.rng, &forms).clone();
        self.out.push(Statement { text, canonical: forms[0].clone(), via_pronoun: false });
    }
}

struct Facts {
    parent: BTreeMap<String, String>,
    roots: Vec<String>,
    classes: Vec<String>,
    entities: Vec<String>,
    entity_class: BTreeMap<String, String>,
    excluded: Vec<(String, String)>,
    acts: Vec<String>,
    places_leaf: Vec<String>,
    regions: Vec<String>,
    towns: Vec<String>,
    located: Vec<String>,
    orders: Vec<(&'static str, &'static str, &'static str, Vec<String>)>,
    chains: Vec<Vec<String>>,
    things: Vec<String>,
    counted: Vec<(String, Vec<String>)>,
    props: Vec<String>,
    /// Questions about facts that were stated through "it".
    pronoun_questions: Vec<(String, String)>,
    contradicted: Vec<(String, String)>,
}

pub fn generate(spec: &Spec, seed: u64) -> World {
    let w = &spec.world;
    let mut names = Names { rng: Rng::from_tag(seed, "reasoning-names"), used: BTreeSet::new() };
    let mut rng = Rng::from_tag(seed, "reasoning-world");
    let mut f = Facts {
        parent: BTreeMap::new(),
        roots: Vec::new(),
        classes: Vec::new(),
        entities: Vec::new(),
        entity_class: BTreeMap::new(),
        excluded: Vec::new(),
        acts: Vec::new(),
        places_leaf: Vec::new(),
        regions: Vec::new(),
        towns: Vec::new(),
        located: Vec::new(),
        orders: Vec::new(),
        chains: Vec::new(),
        things: Vec::new(),
        counted: Vec::new(),
        props: Vec::new(),
        pronoun_questions: Vec::new(),
        contradicted: Vec::new(),
    };
    let mut teach = Teach { rng: &mut rng, out: Vec::new(), pronoun_rate: w.pronoun_rate };
    let mut class_stmts = Vec::new();
    // Taxonomy.
    let mut leaves = Vec::new();
    for _ in 0..w.roots {
        let r = names.fresh(2);
        f.roots.push(r.clone());
        f.classes.push(r.clone());
        for _ in 0..w.children {
            let c = names.fresh(2);
            f.parent.insert(c.clone(), r.clone());
            f.classes.push(c.clone());
            class_stmts.push((c.clone(), r.clone()));
            for _ in 0..w.grandchildren {
                let g = names.fresh(2);
                f.parent.insert(g.clone(), c.clone());
                f.classes.push(g.clone());
                class_stmts.push((g.clone(), c.clone()));
                leaves.push(g);
            }
        }
    }
    for (c, p) in &class_stmts {
        teach.class(vec![
            format!("every {c} is {} {p}", art(p)),
            format!("each {c} is {} {p}", art(p)),
            format!("all {c}s are {p}s"),
            format!("{c}s are {p}s"),
        ]);
    }
    let mut roots = f.roots.clone();
    for i in 0..w.exclusions.min(w.roots - 1) {
        let (a, b) = (roots[i].clone(), roots[i + 1].clone());
        teach.class(vec![format!("no {a} is {} {b}", art(&b)), format!("no {a}s are {b}s")]);
        f.excluded.push((a, b));
    }
    shuffle(&mut roots, seed);
    // Abilities: each act on one mid-level class, with one exception below it.
    for _ in 0..w.acts {
        let v = names.fresh(2);
        f.acts.push(v.clone());
        let mids: Vec<&String> = f.classes.iter().filter(|c| f.parent.contains_key(*c) && !leaves.contains(c)).collect();
        let c = mids[teach.rng.below(mids.len())].clone();
        teach.class(vec![format!("{c}s can {v}"), format!("every {c} can {v}"), format!("all {c}s can {v}"), format!("{} {c} can {v}", art(&c))]);
        let kids: Vec<&String> = leaves.iter().filter(|l| f.parent.get(*l) == Some(&c)).collect();
        if let Some(k) = kids.first() {
            teach.class(vec![format!("every {k} cannot {v}"), format!("no {k} can {v}"), format!("{} {k} cannot {v}", art(k))]);
        }
    }
    // Places.
    for _ in 0..w.regions {
        let r = names.fresh(2);
        f.regions.push(r.clone());
        for _ in 0..w.towns_per_region {
            let t = names.fresh(2);
            f.towns.push(t.clone());
            teach.class(vec![format!("{t} is in {r}"), format!("{t} is inside {r}"), format!("{t} is located in {r}")]);
            for _ in 0..w.sites_per_town {
                let s = names.fresh(2);
                teach.class(vec![format!("{s} is in {t}"), format!("{s} is inside {t}"), format!("{s} is located in {t}")]);
                f.places_leaf.push(s);
            }
        }
    }
    for _ in 0..w.props {
        f.props.push(names.fresh(2));
    }
    for _ in 0..w.things {
        f.things.push(names.fresh(2));
    }
    // Entities, each taught as a block so later facts may use "it".
    type Item = (String, Box<dyn Fn(&str) -> Vec<String>>, (String, String));
    let mut blocks: Vec<Vec<Item>> = Vec::new();
    for i in 0..w.entities {
        let e = names.fresh(2);
        f.entities.push(e.clone());
        let c = leaves[teach.rng.below(leaves.len())].clone();
        f.entity_class.insert(e.clone(), c.clone());
        let mut block: Vec<Item> = Vec::new();
        let c1 = c.clone();
        let q = (format!("is {e} {} {c}?", art(&c)), format!("{e} is {} {c}, right?", art(&c)));
        block.push((e.clone(), Box::new(move |s: &str| vec![format!("{s} is {} {c1}", art(&c1))]), q));
        if i < w.located_entities {
            let site = f.places_leaf[teach.rng.below(f.places_leaf.len())].clone();
            f.located.push(e.clone());
            let q = (format!("is {e} in {site}?"), format!("is {e} inside {site}?"));
            block.push((e.clone(), Box::new(move |s: &str| vec![format!("{s} is in {site}"), format!("{s} is inside {site}"), format!("{s} lives in {site}")]), q));
        }
        if i % 3 == 0 {
            let p = f.props[teach.rng.below(w.rule_chain.min(f.props.len()))].clone();
            let q = (format!("is {e} {p}?"), format!("{e} is {p}, right?"));
            block.push((e.clone(), Box::new(move |s: &str| vec![format!("{s} is {p}")]), q));
        }
        blocks.push(block);
    }
    // Orders (comparatives) and event chains.
    let mut dims: Vec<usize> = (0..COMPARATIVES.len() / 2).collect();
    shuffle(&mut dims, seed ^ 7);
    for &d in dims.iter().take(w.dims) {
        let (gt, lt) = (COMPARATIVES[2 * d].0, COMPARATIVES[2 * d + 1].0);
        let mut order = f.entities.clone();
        shuffle(&mut order, teach.rng.next_u64());
        order.truncate(w.order_len);
        for p in order.windows(2) {
            let (a, b) = (&p[0], &p[1]);
            teach.class(vec![format!("{a} is {gt} than {b}"), format!("{b} is {lt} than {a}")]);
        }
        f.orders.push((COMPARATIVES[2 * d].1, gt, lt, order));
    }
    for _ in 0..w.event_chains {
        let chain: Vec<String> = (0..w.chain_len).map(|_| names.fresh(2)).collect();
        for p in chain.windows(2) {
            let (a, b) = (&p[0], &p[1]);
            teach.class(vec![
                format!("{a} happened before {b}"),
                format!("{a} came before {b}"),
                format!("{a} preceded {b}"),
                format!("{b} happened after {a}"),
                format!("{b} followed {a}"),
            ]);
        }
        f.chains.push(chain);
    }
    // Counts: a base count and a chain of relative counts per thing.
    for t in f.things.clone() {
        let mut members = f.entities.clone();
        shuffle(&mut members, teach.rng.next_u64());
        members.truncate(w.count_chain);
        let mut count = 10 + teach.rng.below(30) as i64;
        let base = members[0].clone();
        let n0 = count;
        teach.class(vec![format!("{base} has {n0} {t}s"), format!("{base} owns {n0} {t}s")]);
        for p in members.windows(2) {
            let (prev, cur) = (&p[0], &p[1]);
            let delta = 1 + teach.rng.below(6) as i64;
            let more = teach.rng.below(2) == 0 || count - delta < 0;
            if more {
                teach.class(vec![format!("{cur} has {delta} more {t}s than {prev}")]);
                count += delta;
            } else {
                teach.class(vec![format!("{cur} has {delta} fewer {t}s than {prev}"), format!("{cur} has {delta} less {t}s than {prev}")]);
                count -= delta;
            }
        }
        f.counted.push((t.clone(), members));
    }
    // Rules: a chain of generic rules over the first `rule_chain` properties.
    for p in f.props[..w.rule_chain.min(f.props.len())].windows(2) {
        let (a, b) = (&p[0], &p[1]);
        teach.class(vec![
            format!("if something is {a} then it is {b}"),
            format!("if a thing is {a}, it is {b}"),
            format!("if anything is {a} then it is {b}"),
        ]);
    }
    // Interleave entity blocks (with pronouns) between class statements.
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    shuffle(&mut order, seed ^ 0x5eed);
    let mut stmts = std::mem::take(&mut teach.out);
    let mut prev: Option<String> = None;
    for i in order {
        for (subj, forms, q) in &blocks[i] {
            teach.say(subj, forms.as_ref(), true, &mut prev);
            if teach.out.last().map(|s| s.via_pronoun).unwrap_or(false) {
                f.pronoun_questions.push(q.clone());
            }
        }
        prev = None;
        // A class statement breaks the pronoun chain.
        if let Some(s) = stmts.pop() {
            teach.out.push(s);
        }
    }
    teach.out.extend(stmts);
    // Injected contradictions (asserted, later denied).
    for e in f.entities.iter().rev().take(w.contradictions) {
        let c = &f.entity_class[e];
        teach.out.push(Statement { text: format!("{e} is not {} {c}", art(c)), canonical: format!("{e} is not {} {c}", art(c)), via_pronoun: false });
        f.contradicted.push((e.clone(), c.clone()));
    }
    let statements = teach.out;
    let questions = questions(spec, seed, &f, &statements);
    World { statements, questions }
}

fn ancestors(f: &Facts, c: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = c.to_string();
    while let Some(p) = f.parent.get(&cur) {
        out.push(p.clone());
        cur = p.clone();
    }
    out
}

/// Candidate questions per category (two surface forms each).
fn candidates(f: &Facts) -> Vec<(&'static str, String, String)> {
    let mut out = Vec::new();
    let isa = |e: &str, c: &str| (format!("is {e} {} {c}?", art(c)), format!("is it true that {e} is {} {c}?", art(c)));
    for e in &f.entities {
        let c = &f.entity_class[e];
        let (a, b) = isa(e, c);
        out.push(("isa_direct", a, b));
        for anc in ancestors(f, c) {
            let (a, b) = isa(e, &anc);
            out.push(("isa_derived", a, b));
        }
        for cls in &f.classes {
            let (a, b) = isa(e, cls);
            out.push(("isa_excluded", a.clone(), b.clone()));
            out.push(("isa_unknown", a, b));
        }
        for v in &f.acts {
            for cat in ["ability_inherited", "ability_exception", "ability_unknown"] {
                out.push((cat, format!("can {e} {v}?"), format!("is {e} able to {v}?")));
            }
        }
        for p in f.towns.iter().chain(&f.regions) {
            for cat in ["location_derived", "location_unknown"] {
                out.push((cat, format!("is {e} in {p}?"), format!("is {e} inside {p}?")));
            }
        }
        for p in &f.props {
            for cat in ["conditional_derived", "conditional_unknown"] {
                out.push((cat, format!("is {e} {p}?"), format!("{e} is {p}, right?")));
            }
        }
        for t in &f.things {
            for cat in ["arithmetic_direct", "arithmetic_derived", "arithmetic_unknown"] {
                out.push((cat, format!("how many {t}s does {e} have?"), format!("how many {t}s has {e} got?")));
            }
        }
    }
    for (_, gt, lt, order) in &f.orders {
        for a in order {
            for b in &f.entities {
                if a == b {
                    continue;
                }
                for cat in ["comparison_derived", "comparison_unknown"] {
                    out.push((cat, format!("is {a} {gt} than {b}?"), format!("is {b} {lt} than {a}?")));
                }
                out.push(("comparison_derived", format!("who is {gt}, {a} or {b}?"), format!("which is {gt}, {a} or {b}?")));
            }
        }
    }
    let events: Vec<&String> = f.chains.iter().flatten().collect();
    for a in &events {
        for b in &events {
            if a != b {
                for cat in ["temporal_derived", "temporal_unknown"] {
                    out.push((cat, format!("did {a} happen before {b}?"), format!("did {b} happen after {a}?")));
                }
            }
        }
    }
    for (a, b) in &f.pronoun_questions {
        out.push(("pronoun", a.clone(), b.clone()));
    }
    for (e, c) in &f.contradicted {
        let (a, b) = isa(e, c);
        out.push(("contradiction", a, b));
    }
    out
}

fn questions(spec: &Spec, seed: u64, f: &Facts, statements: &[Statement]) -> Vec<Question> {
    let mut o = Oracle::new(spec.config.max_depth);
    for s in statements {
        o.observe(&s.text);
    }
    let lookup = Lookup { oracle: &o };
    let mut cands = candidates(f);
    shuffle(&mut cands, Rng::from_tag(seed, "reasoning-questions").next_u64());
    let mut per: BTreeMap<&str, usize> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (cat, text, alt) in cands {
        if per.get(cat).copied().unwrap_or(0) >= spec.per_category || seen.contains(&(cat, text.clone())) {
            continue;
        }
        let Some(Utterance::Query(q)) = parse(&text) else { panic!("generator produced an unparseable question: {text}") };
        let l = o.answer(&q);
        let lk = lookup.answer(&q, None);
        let v = &l.verdict;
        let keep = match cat {
            "isa_direct" | "pronoun" => *v == Verdict::Yes && !l.derived,
            "isa_excluded" => *v == Verdict::No && l.derived,
            "ability_exception" => *v == Verdict::No,
            // Positive inferences only; derived "no" answers belong to the exception/exclusion categories.
            "isa_derived" | "ability_inherited" | "location_derived" | "conditional_derived" => *v == Verdict::Yes && l.derived && lk != *v,
            "isa_unknown" | "ability_unknown" | "location_unknown" | "comparison_unknown" | "temporal_unknown" | "arithmetic_unknown"
            | "conditional_unknown" => *v == Verdict::Unknown,
            "arithmetic_direct" => matches!(v, Verdict::Number(_)) && !l.derived,
            "contradiction" => *v == Verdict::Contradiction,
            // Derived categories: an inference the lookup baseline cannot make.
            _ => !matches!(v, Verdict::Unknown | Verdict::Contradiction) && l.derived && lk != *v,
        };
        if keep {
            seen.insert((cat, text.clone()));
            *per.entry(cat).or_insert(0) += 1;
            out.push(Question { category: cat, text, alt });
        }
    }
    out
}

// ------------------------------------------------------------------ evaluation

#[derive(Debug, Clone, Default, Serialize)]
pub struct CategoryScore {
    pub category: &'static str,
    pub questions: usize,
    pub agreement: f64,
    pub lookup_agreement: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AblationResult {
    pub name: String,
    pub kind: &'static str,
    pub target_categories: Vec<&'static str>,
    /// Mean agreement on the target categories, intact and ablated.
    pub target_intact: f64,
    pub target_ablated: f64,
    /// Largest agreement drop on a category the ablation is not expected to touch.
    pub collateral_max_drop: f64,
    pub per_category: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeedResult {
    pub seed: u64,
    pub hidden: bool,
    /// Hidden seeds publish only hashes: no seed, no questions, no labels.
    pub seed_sha256: String,
    pub statements: usize,
    pub statements_via_pronoun: usize,
    pub questions: usize,
    pub questions_sha256: String,
    pub categories: Vec<CategoryScore>,
    pub agreement: f64,
    pub derived_agreement: f64,
    pub lookup_agreement: f64,
    pub lookup_derived_agreement: f64,
    pub derived_flag_agreement: f64,
    pub paraphrase_consistency: f64,
    pub counterfactual_required: usize,
    pub counterfactual_required_followed: f64,
    pub counterfactual_invariant: usize,
    pub counterfactual_invariant_kept: f64,
    pub provenance_checked: usize,
    pub provenance_sufficient: f64,
    pub restart_identical: bool,
    pub ablations: Vec<AblationResult>,
    pub question_p50_us: f64,
    pub question_p95_us: f64,
    pub slots_used: usize,
    pub symbols_used: usize,
    pub checks: Vec<Check>,
}

fn build(cfg: ReasonerConfig, texts: &[&str], ops: Ops, pathway: Option<Pathway>) -> Result<(Reasoner, Vec<Option<usize>>), String> {
    let mut r = Reasoner::new(cfg).map_err(|e| e.to_string())?;
    r.ops = ops;
    if let Some(p) = pathway {
        r.ablate(p);
    }
    let mut slot_of = Vec::with_capacity(texts.len());
    for t in texts {
        let o = r.observe(t, "reasoning:synthetic-world").map_err(|e| e.to_string())?;
        slot_of.push(match o {
            Observed::Stored { slot } | Observed::Duplicate { slot } => Some(slot),
            _ => None,
        });
    }
    Ok((r, slot_of))
}

fn oracle_of(cfg: &ReasonerConfig, texts: &[&str]) -> Oracle {
    let mut o = Oracle::new(cfg.max_depth);
    for t in texts {
        o.observe(t);
    }
    o
}

fn label(o: &Oracle, text: &str) -> Verdict {
    match parse(text) {
        Some(Utterance::Query(q)) => o.answer(&q).verdict,
        _ => Verdict::Unknown,
    }
}

fn ask(r: &Reasoner, text: &str) -> Verdict {
    r.ask(text).map(|a| a.verdict).unwrap_or(Verdict::Unknown)
}

fn frac(n: usize, d: usize) -> f64 {
    if d == 0 {
        1.0
    } else {
        n as f64 / d as f64
    }
}

fn sha(s: &str) -> String {
    zerolm_learning::streams::sha256_hex(s.as_bytes())
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn per_category(world: &World, verdicts: &[Verdict], labels: &[Verdict]) -> BTreeMap<&'static str, f64> {
    let mut m: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for ((q, v), l) in world.questions.iter().zip(verdicts).zip(labels) {
        let e = m.entry(q.category).or_insert((0, 0));
        e.0 += (v == l) as usize;
        e.1 += 1;
    }
    m.into_iter().map(|(k, (a, n))| (k, frac(a, n))).collect()
}

pub fn run_seed(spec: &Spec, seed: u64, hidden: bool, work: &Path) -> Result<SeedResult, String> {
    let cfg = spec.config;
    let world = generate(spec, seed);
    let texts: Vec<&str> = world.statements.iter().map(|s| s.text.as_str()).collect();
    let canon: Vec<&str> = world.statements.iter().map(|s| s.canonical.as_str()).collect();
    let mut checks = Vec::new();
    let (mut r, slot_of) = build(cfg, &texts, Ops::ALL, None)?;
    let o = oracle_of(&cfg, &texts);
    let lookup = Lookup { oracle: &o };

    // Main evaluation.
    let mut times = Vec::new();
    let mut verdicts = Vec::new();
    let mut answers = Vec::new();
    let mut labels = Vec::new();
    let mut lookups = Vec::new();
    let (mut flag_ok, mut para_ok) = (0, 0);
    for q in &world.questions {
        let Some(Utterance::Query(pq)) = parse(&q.text) else { return Err(format!("unparseable question {}", q.text)) };
        let l = o.answer(&pq);
        let t0 = Instant::now();
        let a = r.ask(&q.text).ok_or("question did not parse")?;
        times.push(t0.elapsed().as_secs_f64() * 1e6);
        flag_ok += (a.derived == l.derived) as usize;
        para_ok += (ask(&r, &q.alt) == a.verdict) as usize;
        lookups.push(lookup.answer(&pq, None));
        verdicts.push(a.verdict.clone());
        labels.push(l.verdict);
        answers.push(a);
    }
    let nq = world.questions.len();
    let agree = |v: &[Verdict], filter: &dyn Fn(&Question) -> bool| {
        let idx: Vec<usize> = (0..nq).filter(|&i| filter(&world.questions[i])).collect();
        frac(idx.iter().filter(|&&i| v[i] == labels[i]).count(), idx.len())
    };
    let is_derived = |q: &Question| CATEGORIES_DERIVED.contains(&q.category);
    let cat_scores = per_category(&world, &verdicts, &labels);
    let lk_scores = per_category(&world, &lookups, &labels);
    let categories: Vec<CategoryScore> = CATEGORIES
        .iter()
        .map(|c| CategoryScore {
            category: c,
            questions: world.questions.iter().filter(|q| q.category == *c).count(),
            agreement: cat_scores.get(c).copied().unwrap_or(0.0),
            lookup_agreement: lk_scores.get(c).copied().unwrap_or(0.0),
        })
        .collect();
    let missing: Vec<&str> = categories.iter().filter(|c| c.questions == 0).map(|c| c.category).collect();
    checks.push(Check { name: "all_categories_populated", passed: missing.is_empty(), detail: format!("empty: {missing:?}") });
    // Questions through the mutable entry point must not write memory.
    let before = r.n_slots();
    for q in &world.questions {
        r.observe(&q.text, "reasoning:questions").map_err(|e| e.to_string())?;
    }
    checks.push(Check {
        name: "answers_never_written_back",
        passed: r.n_slots() == before,
        detail: format!("{} slots before and {} after {nq} questions", before, r.n_slots()),
    });

    // Counterfactual pairs: drop one premise of the oracle's proof.
    let mut crng = Rng::from_tag(seed, "reasoning-counterfactual");
    let mut cf_idx: Vec<usize> = (0..nq).filter(|&i| is_derived(&world.questions[i]) || world.questions[i].category == "isa_excluded").collect();
    shuffle(&mut cf_idx, crng.next_u64());
    let (mut req, mut req_ok, mut inv, mut inv_ok) = (0, 0, 0, 0);
    let co = oracle_of(&cfg, &canon);
    for &i in cf_idx.iter().take(spec.counterfactual_pairs) {
        let q = &world.questions[i];
        let Some(Utterance::Query(pq)) = parse(&q.text) else { continue };
        let proof = co.answer(&pq).premises;
        // Required change: remove a premise of the proof.
        if let Some(&drop) = proof.get(crng.below(proof.len().max(1))) {
            let sub: Vec<&str> = canon.iter().enumerate().filter(|(j, _)| *j != drop).map(|(_, t)| *t).collect();
            let (rr, _) = build(cfg, &sub, Ops::ALL, None)?;
            let ol = label(&oracle_of(&cfg, &sub), &q.text);
            if ol != labels[i] {
                req += 1;
                req_ok += (ask(&rr, &q.text) == ol && verdicts[i] == labels[i]) as usize;
            }
        }
        // Invariance: remove a statement outside the proof.
        let outside: Vec<usize> = (0..canon.len()).filter(|j| !proof.contains(j)).collect();
        let drop = outside[crng.below(outside.len())];
        let sub: Vec<&str> = canon.iter().enumerate().filter(|(j, _)| *j != drop).map(|(_, t)| *t).collect();
        let ol = label(&oracle_of(&cfg, &sub), &q.text);
        if ol == labels[i] {
            let (rr, _) = build(cfg, &sub, Ops::ALL, None)?;
            inv += 1;
            inv_ok += (ask(&rr, &q.text) == ol) as usize;
        }
    }

    // Provenance sufficiency: the reported premises alone reproduce the answer.
    let slot_to_stmt: BTreeMap<usize, usize> = slot_of.iter().enumerate().filter_map(|(j, s)| s.map(|s| (s, j))).collect();
    let mut pv: Vec<usize> = (0..nq).filter(|&i| !matches!(answers[i].verdict, Verdict::Unknown) && !answers[i].premises.is_empty()).collect();
    shuffle(&mut pv, crng.next_u64());
    let (mut pchk, mut pok) = (0, 0);
    for &i in pv.iter().take(spec.provenance_checks) {
        let sub: Vec<&str> = answers[i].premises.iter().filter_map(|s| slot_to_stmt.get(s)).map(|&j| canon[j]).collect();
        let (rr, _) = build(cfg, &sub, Ops::ALL, None)?;
        pchk += 1;
        pok += (ask(&rr, &world.questions[i].text) == answers[i].verdict) as usize;
    }

    // Restart identity.
    let dir = work.join(format!("seed-{seed}"));
    let _ = std::fs::remove_dir_all(&dir);
    r.save(&dir).map_err(|e| e.to_string())?;
    let loaded = Reasoner::load(&dir).map_err(|e| e.to_string())?;
    let restart_identical = world.questions.iter().zip(&answers).all(|(q, a)| loaded.ask(&q.text).as_ref() == Some(a));
    let _ = std::fs::remove_dir_all(&dir);

    // Ablations.
    let mut ablations = Vec::new();
    let intact = cat_scores.clone();
    for name in Ops::NAMES.iter().copied().chain(PATHWAYS) {
        let (kind, verdicts_ab) = if PATHWAYS.contains(&name) {
            let p = match name {
                "binding" => Pathway::Binding,
                "recurrent_state" => Pathway::RecurrentState,
                _ => Pathway::Retrieval,
            };
            let (rr, _) = build(cfg, &texts, Ops::ALL, Some(p))?;
            ("pathway", world.questions.iter().map(|q| ask(&rr, &q.text)).collect::<Vec<_>>())
        } else {
            r.ops = Ops::without(name);
            let v = world.questions.iter().map(|q| ask(&r, &q.text)).collect::<Vec<_>>();
            r.ops = Ops::ALL;
            ("operator", v)
        };
        let ab = per_category(&world, &verdicts_ab, &labels);
        let (targets, allowed) = ablation_targets(name);
        let mean = |m: &BTreeMap<&str, f64>| targets.iter().map(|c| m.get(c).copied().unwrap_or(0.0)).sum::<f64>() / targets.len().max(1) as f64;
        let collateral = CATEGORIES
            .iter()
            .filter(|c| !allowed.contains(c))
            .map(|c| intact.get(c).copied().unwrap_or(0.0) - ab.get(c).copied().unwrap_or(0.0))
            .fold(0.0f64, f64::max);
        ablations.push(AblationResult {
            name: name.to_string(),
            kind,
            target_categories: targets.to_vec(),
            target_intact: mean(&intact),
            target_ablated: mean(&ab),
            collateral_max_drop: collateral,
            per_category: ab.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        });
    }

    let mut qtext = String::new();
    for (q, l) in world.questions.iter().zip(&labels) {
        qtext += &format!("{}\t{}\t{}\n", q.category, q.text, l.text());
    }
    Ok(SeedResult {
        seed: if hidden { 0 } else { seed },
        hidden,
        seed_sha256: sha(&seed.to_string()),
        statements: world.statements.len(),
        statements_via_pronoun: world.statements.iter().filter(|s| s.via_pronoun).count(),
        questions: nq,
        questions_sha256: sha(&qtext),
        categories,
        agreement: agree(&verdicts, &|_| true),
        derived_agreement: agree(&verdicts, &is_derived),
        lookup_agreement: agree(&lookups, &|_| true),
        lookup_derived_agreement: agree(&lookups, &is_derived),
        derived_flag_agreement: frac(flag_ok, nq),
        paraphrase_consistency: frac(para_ok, nq),
        counterfactual_required: req,
        counterfactual_required_followed: frac(req_ok, req),
        counterfactual_invariant: inv,
        counterfactual_invariant_kept: frac(inv_ok, inv),
        provenance_checked: pchk,
        provenance_sufficient: frac(pok, pchk),
        restart_identical,
        ablations,
        question_p50_us: pct(&mut times.clone(), 0.5),
        question_p95_us: pct(&mut times, 0.95),
        slots_used: r.n_slots(),
        symbols_used: r.n_symbols(),
        checks,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub agreement: Stat,
    pub derived_agreement: Stat,
    pub lookup_derived_agreement: Stat,
    pub paraphrase_consistency: Stat,
    pub counterfactual_required_followed: Stat,
    pub provenance_sufficient: Stat,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Report kind (always "reasoning").
    pub report: &'static str,
    pub mode: &'static str,
    pub passed: bool,
    pub failures: Vec<String>,
    pub spec: Spec,
    pub spec_sha256: String,
    pub hidden_seed_source: String,
    pub seeds: Vec<SeedResult>,
    pub summary: Summary,
    pub scope_note: &'static str,
}

pub const SCOPE_NOTE: &str = "Reasoning here is over a bounded, programmed statement language with generated pseudo-word content. Structures are stored as bound phasor vectors and answered by vector probes, unbinding, cleanup, and binding-based arithmetic; the operators' control flow is programmed, not learned, and agrees with a symbolic reference by design. This shows compositional answers from stored primitives with causal pathway evidence, not open-domain language understanding or learned reasoning.";

/// The hidden seed: from ZEROLM_HIDDEN_SEED if set, else derived from the
/// current commit (not chosen by the developer, changes every commit).
pub fn hidden_seed(root: &Path, salt: &str) -> Option<(u64, String)> {
    if let Ok(s) = std::env::var("ZEROLM_HIDDEN_SEED") {
        if !s.trim().is_empty() {
            return Some((zerolm_tensor::fnv1a(format!("{salt}:{s}").as_bytes()), "env:ZEROLM_HIDDEN_SEED".into()));
        }
    }
    let out = std::process::Command::new("git").arg("-C").arg(root).args(["rev-parse", "HEAD"]).output().ok()?;
    let head = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && head.len() == 40).then(|| (zerolm_tensor::fnv1a(format!("{salt}:{head}").as_bytes()), "git:HEAD".to_string()))
}

pub fn run(root: &Path, spec: &Spec, seeds: &[u64], gate: bool, work: &Path) -> Result<Report, String> {
    let spec_sha256 = sha(&std::fs::read_to_string(root.join(SPEC)).map_err(|e| e.to_string())?);
    let mut jobs: Vec<(u64, bool)> = seeds.iter().map(|&s| (s, false)).collect();
    let mut hidden_seed_source = "not_run (pilot)".to_string();
    if gate {
        let (h, src) = hidden_seed(root, &spec.hidden_salt).ok_or("no hidden seed: set ZEROLM_HIDDEN_SEED or run inside a git checkout")?;
        jobs.push((h, true));
        hidden_seed_source = src;
    }
    let results: Vec<Result<SeedResult, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = jobs.iter().map(|&(seed, hidden)| s.spawn(move || run_seed(spec, seed, hidden, work))).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("seed thread panicked".into()))).collect()
    });
    let seeds_out: Vec<SeedResult> = results.into_iter().collect::<Result<_, _>>()?;
    let g = &spec.gates;
    let mut failures = Vec::new();
    for r in &seeds_out {
        let id = if r.hidden { "hidden".to_string() } else { format!("seed {}", r.seed) };
        let mut fail = |ok: bool, what: String| {
            if !ok {
                failures.push(format!("{id}: {what}"));
            }
        };
        for c in &r.checks {
            fail(c.passed, format!("{} ({})", c.name, c.detail));
        }
        fail(r.agreement >= g.agreement_min, format!("agreement {:.4} < {}", r.agreement, g.agreement_min));
        fail(r.derived_agreement >= g.derived_agreement_min, format!("derived agreement {:.4} < {}", r.derived_agreement, g.derived_agreement_min));
        for c in &r.categories {
            fail(c.agreement >= g.category_agreement_min, format!("{} agreement {:.4} < {}", c.category, c.agreement, g.category_agreement_min));
        }
        fail(
            r.derived_agreement - r.lookup_derived_agreement >= g.lookup_gap_min,
            format!("lookup gap {:.4} < {}", r.derived_agreement - r.lookup_derived_agreement, g.lookup_gap_min),
        );
        fail(r.paraphrase_consistency >= g.paraphrase_consistency_min, format!("paraphrase consistency {:.4}", r.paraphrase_consistency));
        fail(r.counterfactual_required >= 5 && r.counterfactual_required_followed >= g.counterfactual_min, format!("counterfactual {}/{:.4}", r.counterfactual_required, r.counterfactual_required_followed));
        fail(r.counterfactual_invariant_kept >= g.counterfactual_min, format!("counterfactual invariance {:.4}", r.counterfactual_invariant_kept));
        fail(r.provenance_sufficient >= g.provenance_min, format!("provenance {:.4}", r.provenance_sufficient));
        fail(r.restart_identical, "answers changed after restart".into());
        for a in &r.ablations {
            fail(a.target_intact - a.target_ablated >= g.ablation_drop_min, format!("ablation {} drops targets only {:.4}", a.name, a.target_intact - a.target_ablated));
            if a.kind == "operator" {
                fail(a.collateral_max_drop <= g.operator_collateral_max, format!("ablation {} collateral {:.4}", a.name, a.collateral_max_drop));
            }
        }
    }
    let visible: Vec<&SeedResult> = seeds_out.iter().filter(|r| !r.hidden).collect();
    let stat = |f: &dyn Fn(&SeedResult) -> f64| Stat::of(&visible.iter().map(|r| f(r)).collect::<Vec<_>>());
    let summary = Summary {
        agreement: stat(&|r| r.agreement),
        derived_agreement: stat(&|r| r.derived_agreement),
        lookup_derived_agreement: stat(&|r| r.lookup_derived_agreement),
        paraphrase_consistency: stat(&|r| r.paraphrase_consistency),
        counterfactual_required_followed: stat(&|r| r.counterfactual_required_followed),
        provenance_sufficient: stat(&|r| r.provenance_sufficient),
    };
    Ok(Report {
        report: "reasoning",
        mode: if gate { "main" } else { "pilot" },
        passed: gate && failures.is_empty(),
        failures,
        spec: spec.clone(),
        spec_sha256,
        hidden_seed_source,
        seeds: seeds_out,
        summary,
        scope_note: SCOPE_NOTE,
    })
}

pub fn summary_md(r: &Report) -> String {
    let mut s = format!("## Compositional reasoning ({} mode)\n\n", r.mode);
    s += &format!("**{}** | hidden seed source: {}\n\n", if r.passed { "PASSED" } else if r.mode == "pilot" { "PILOT (not gated)" } else { "FAILED" }, r.hidden_seed_source);
    s += "| seed | questions | agreement | derived | lookup (derived) | paraphrase | counterfactual | provenance | restart |\n|---|---|---|---|---|---|---|---|---|\n";
    for x in &r.seeds {
        let id = if x.hidden { format!("hidden ({}…)", &x.seed_sha256[..12]) } else { x.seed.to_string() };
        s += &format!(
            "| {id} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {}/{:.3} | {}/{:.3} | {} |\n",
            x.questions,
            x.agreement,
            x.derived_agreement,
            x.lookup_derived_agreement,
            x.paraphrase_consistency,
            x.counterfactual_required,
            x.counterfactual_required_followed,
            x.provenance_checked,
            x.provenance_sufficient,
            x.restart_identical
        );
    }
    if let Some(x) = r.seeds.iter().find(|x| !x.hidden) {
        s += &format!("\n### Seed {} by category (reasoner vs oracle, lookup vs oracle)\n\n| category | n | reasoner | lookup |\n|---|---|---|---|\n", x.seed);
        for c in &x.categories {
            s += &format!("| {} | {} | {:.3} | {:.3} |\n", c.category, c.questions, c.agreement, c.lookup_agreement);
        }
        s += "\n| ablation | kind | targets | intact | ablated | max collateral drop |\n|---|---|---|---|---|---|\n";
        for a in &x.ablations {
            s += &format!("| {} | {} | {} | {:.3} | {:.3} | {:.3} |\n", a.name, a.kind, a.target_categories.len(), a.target_intact, a.target_ablated, a.collateral_max_drop);
        }
        s += &format!("\nStatements {} ({} via \"it\"), slots {}, symbols {}, question p50/p95 {:.0}/{:.0} µs.\n", x.statements, x.statements_via_pronoun, x.slots_used, x.symbols_used, x.question_p50_us, x.question_p95_us);
    }
    if !r.failures.is_empty() {
        s += "\n### Failures\n\n";
        for f in &r.failures {
            s += &format!("- {f}\n");
        }
    }
    s += &format!("\n_{}_\n", r.scope_note);
    s
}
