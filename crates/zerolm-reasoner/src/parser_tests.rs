use super::*;
use crate::ReasonerConfig;
use zerolm_engine::semantic_state::{parse, is_reserved, NUMBER_WORDS};

fn reasoner() -> Reasoner {
    Reasoner::new(ReasonerConfig { dim: 64, symbols: 32, slots: 32, number_range: 20, ..Default::default() }).unwrap()
}

#[test]
fn token_attributes_match_reference_for_reserved_plural_and_numeric_edges() {
    let r = reasoner();
    let mut words: Vec<String> = ["dax", "it", "s", "as", "has", "hass", "zeros", "ones", "tallers", "age", "height", "0", "-0", "--0", "-1", "1000000", "1000001", "999999", "1s", "1-2", "9223372036854775807", "9223372036854775808"].iter().map(|w| w.to_string()).collect();
    words.extend(NUMBER_WORDS.iter().map(|w| w.to_string()));
    words.extend([format!("{}1", "0".repeat(300)), format!("-{}0", "0".repeat(300)), format!("1{}", "0".repeat(300))]);
    let attributes = r.parser_features(&words);
    for (word, row) in words.iter().zip(attributes.data.chunks_exact(FEATURES)) {
        let singular = word.strip_suffix('s').filter(|w| w.len() >= 2).unwrap_or(word);
        let number = word.parse::<i64>().ok().filter(|n| (0..=1_000_000).contains(n)).or_else(|| NUMBER_WORDS.iter().position(|w| *w == word).map(|n| n as i64));
        assert_eq!(row[1], (!is_reserved(word)) as u8 as f32, "{word}: name");
        assert_eq!(row[3], (word == "it") as u8 as f32, "{word}: pronoun");
        assert_eq!(row[4], (!is_reserved(singular)) as u8 as f32, "{word}: singular name");
        assert_eq!(row[5], word.ends_with('s') as u8 as f32, "{word}: suffix");
        assert_eq!(row[6], number.is_some() as u8 as f32, "{word}: numeral validity");
        assert_eq!(row[7], number.unwrap_or(0) as f32, "{word}: numeral value");
        assert_eq!(row[14], singular.len() as f32, "{word}: singular span");
    }
}

#[test]
fn question_marker_matches_unicode_trim_end_semantics() {
    let r = reasoner();
    for point in [9,10,11,12,13,32,0x85,0xa0,0x1680,0x2000,0x2001,0x2002,0x2003,0x2004,0x2005,0x2006,0x2007,0x2008,0x2009,0x200a,0x2028,0x2029,0x202f,0x205f,0x3000,0x200b,0xfeff,0x2026,0] {
        let c = char::from_u32(point).unwrap();
        for text in [format!("dax is wet?{c}"), format!("dax is wet{c}? \t"), format!("{c}{c}"), format!("?{c}{c}")] {
            assert_eq!(r.raw_question(&text).data[0], text.trim_end().ends_with('?') as u8 as f32, "{text:?}");
            assert_eq!(r.parse(&text), parse(&text), "{text:?}");
        }
    }
    assert_eq!(r.raw_question("").data, [0.0]);
}

#[test]
fn all_seeded_templates_and_guard_mutations_match_the_reference() {
    let r = reasoner();
    let seed: serde_json::Value = serde_json::from_str(include_str!("../../../model/bootstrap/reasoning-grammar.json")).unwrap();
    let mut tested = 0;
    for template in seed["parser"]["templates"].as_array().unwrap() {
        let mut words: Vec<String> = template["tokens"].as_array().unwrap().iter().map(|cell| {
            if let Some(s) = cell.as_str() { return s.to_string(); }
            let requirements = cell["require"].as_array().unwrap();
            if requirements.iter().any(|r| r == "number_valid") { "seven".into() }
            else if requirements.iter().any(|r| r == "singular_name" || r == "plural_subject") { "nubs".into() }
            else { "dax".into() }
        }).collect();
        for pair in template["same"].as_array().unwrap() {
            words[pair[1].as_u64().unwrap() as usize] = words[pair[0].as_u64().unwrap() as usize].clone();
        }
        let suffix = if template["question"] == 1 { "?" } else { "" };
        let check = |words: &[String]| {
            let text = format!("{}{suffix}", words.join(" "));
            assert_eq!(r.parse(&text), parse(&text), "{text:?}");
        };
        check(&words); tested += 1;
        for position in 0..words.len() {
            let old = words[position].clone();
            for replacement in ["is", "it", "has", "zeros", "zup"] {
                words[position] = replacement.into(); check(&words); tested += 1;
            }
            words[position] = old;
        }
    }
    assert!(tested > 2_000);
    for text in [
        "dax has 1000000 nubs", "dax has 1000001 nubs", "dax has -0 nubs", "dax has -1 nubs", "dax has twenty fewer nubs than zup",
        "if something is wet then it is cold", "if dax is wet zup is cold", "if dax is wet dax is cold", "if dax is wet it is cold",
        "dax is not wet?", "dax has five nubs?", "one two three four five six seven eight nine ten",
    ] { assert_eq!(r.parse(text), parse(text), "{text:?}"); }
    let a = format!("{}x", "a".repeat(65)); let b = format!("{}y", "a".repeat(65));
    for other in [&a, &b] {
        let text = format!("if {a} is wet then {other} is cold");
        assert_eq!(r.parse(&text), parse(&text), "scope equality must include the whole name");
    }
}

#[test]
fn parser_decisions_depend_on_the_named_circuit_blocks() {
    let mut r = reasoner();
    assert!(r.parse("dax has 7 nubs").is_some());
    r.grammar.parser.byte_lookup.ablate_block("match").unwrap();
    assert_eq!(r.parse("dax has 7 nubs"), None);
    let mut r = reasoner();
    assert_eq!(r.parse("dax has 1000001 nubs"), None);
    r.grammar.parser.number.ablate_block("decimal").unwrap();
    assert!(r.parse("dax has 1000001 nubs").is_some(), "the bounds must be carried by the decimal block");
    let mut r = reasoner();
    assert!(r.parse("if dax is wet then dax is cold").is_some());
    r.grammar.parser.compare.ablate_block("match").unwrap();
    assert_eq!(r.parse("if dax is wet then dax is cold"), None);
    let mut r = reasoner();
    let before = r.parse("dax is wet?\u{2003}");
    r.grammar.parser.whitespace.ablate_block("match").unwrap();
    assert_ne!(r.parse("dax is wet?\u{2003}"), before);
    for program in ["features", "matcher"] {
        let mut r = reasoner();
        assert!(r.parse("dax is wet").is_some());
        match program {
            "features" => r.grammar.parser.features.ablate_block("select").unwrap(),
            "matcher" => r.grammar.parser.matcher.ablate_block("match").unwrap(),
            _ => unreachable!(),
        }
        assert_eq!(r.parse("dax is wet"), None);
    }
}
