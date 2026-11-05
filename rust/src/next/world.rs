//! Deterministic grid world and executor. The executor receives only structured `Op`s produced by
//! the network; it has no access to the natural-language input.
//!
//! Coordinates: origin top-left, x grows to the right, y grows downward, so "up" is y-1.

use super::decoder::Op;
use crate::jsonl::{self, Json};

#[derive(Debug, Clone, PartialEq)]
pub struct Obj {
    pub id: i64,
    pub kind: String,
    pub color: String,
    pub x: i64,
    pub y: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct World {
    pub width: i64,
    pub height: i64,
    pub objects: Vec<Obj>,
}

impl World {
    pub fn from_json(j: &Json) -> Option<World> {
        let width = j.get("width")?.as_f64()? as i64;
        let height = j.get("height")?.as_f64()? as i64;
        let mut objects = Vec::new();
        for o in j.get("objects")?.as_array()? {
            objects.push(Obj {
                id: o.get("id")?.as_f64()? as i64,
                kind: o.get("type")?.as_str()?.to_string(),
                color: o.get("color")?.as_str()?.to_string(),
                x: o.get("x")?.as_f64()? as i64,
                y: o.get("y")?.as_f64()? as i64,
            });
        }
        Some(World { width, height, objects })
    }

    pub fn parse_str(s: &str) -> Option<World> {
        World::from_json(&jsonl::parse(s)?)
    }

    pub fn to_json(&self) -> String {
        let mut objs = self.objects.clone();
        objs.sort_by_key(|o| o.id);
        let items: Vec<String> = objs
            .iter()
            .map(|o| format!("{{\"id\":{},\"type\":\"{}\",\"color\":\"{}\",\"x\":{},\"y\":{}}}", o.id, o.kind, o.color, o.x, o.y))
            .collect();
        format!("{{\"width\":{},\"height\":{},\"objects\":[{}]}}", self.width, self.height, items.join(","))
    }

    fn matching(&self, kind: &str, color: Option<&str>) -> Vec<usize> {
        self.objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.kind == kind && match color {
                Some(c) => o.color == c,
                None => true,
            })
            .map(|(i, _)| i)
            .collect()
    }
}

pub fn execute(world: &World, ops: &[Op]) -> (World, Vec<String>) {
    let mut w = world.clone();
    let mut status = Vec::new();
    for op in ops {
        status.push(apply(&mut w, op));
    }
    (w, status)
}

fn apply(w: &mut World, op: &Op) -> String {
    if op.get("negated") == Some("true") {
        return "skipped_negated".into();
    }
    if let Some(ct) = op.get("condition.type") {
        let exists = !w.matching(ct, op.get("condition.color")).is_empty();
        let unless = op.get("condition.negated") == Some("true");
        if exists == unless {
            return "skipped_condition".into();
        }
    }
    let (Some(action), Some(kind)) = (op.get("action"), op.get("object.type")) else {
        return "malformed_op".into();
    };
    let found = w.matching(kind, op.get("object.color"));
    let idx = match found.as_slice() {
        [] => return "no_such_object".into(),
        [i] => *i,
        _ => return "ambiguous_object".into(),
    };
    match action {
        "move" => {
            let n: i64 = op.get("count").and_then(|c| c.parse().ok()).unwrap_or(1);
            let (dx, dy) = match op.get("direction") {
                Some("left") => (-1, 0),
                Some("right") => (1, 0),
                Some("up") => (0, -1),
                Some("down") => (0, 1),
                _ => return "malformed_op".into(),
            };
            let (nx, ny) = (w.objects[idx].x + dx * n, w.objects[idx].y + dy * n);
            if nx < 0 || ny < 0 || nx >= w.width || ny >= w.height {
                return "out_of_bounds".into();
            }
            w.objects[idx].x = nx;
            w.objects[idx].y = ny;
            "ok".into()
        }
        "paint" => match op.get("to_color") {
            Some(c) => {
                w.objects[idx].color = c.to_string();
                "ok".into()
            }
            None => "malformed_op".into(),
        },
        "remove" => {
            w.objects.remove(idx);
            "ok".into()
        }
        _ => "malformed_op".into(),
    }
}
