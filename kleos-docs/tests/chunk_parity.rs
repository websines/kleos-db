//! The plain-text chunkers against Chonkie 1.7 (tests/fixtures/chonkie.json,
//! made by scripts/docs/chonkie_fixture.py from the generated sample pages).
use kleos_docs::chunk::*;

#[test]
fn token_sentence_recursive_match_chonkie() {
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!("fixtures/chonkie.json")).unwrap();
    let (mut n, mut same) = (0, 0);
    let mut fails = vec![];
    for c in &cases {
        let text = c["text"].as_str().unwrap();
        let size = c["size"].as_u64().unwrap() as usize;
        let tok: Box<dyn Tokenizer> = if c["tokenizer"] == "word" { Box::new(WordTokenizer) } else { Box::new(CharTokenizer) };
        for (name, got) in [
            ("token", token_chunks(text, tok.as_ref(), size, size / 10)),
            ("sentence", sentence_chunks(text, tok.as_ref(), size, 0, 1, 12)),
            ("recursive", recursive_chunks(text, tok.as_ref(), size, 24, &default_levels())),
        ] {
            let want: Vec<(String, usize, usize, usize)> = c[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| (x[0].as_str().unwrap().to_string(), x[1].as_u64().unwrap() as usize, x[2].as_u64().unwrap() as usize, x[3].as_u64().unwrap() as usize))
                .collect();
            let got: Vec<(String, usize, usize, usize)> = got.into_iter().map(|x| (x.text, x.start, x.end, x.tokens)).collect();
            n += 1;
            if got == want {
                same += 1;
            } else {
                fails.push(format!("{name} {} {size}: {} vs {} chunks", c["tokenizer"], got.len(), want.len()));
            }
        }
    }
    assert!(fails.is_empty(), "{same}/{n} identical; differ: {fails:#?}");
}
