// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::*;

#[rstest::rstest]
// Just past the inline boundary, typical, and large values.
#[case(13, 7, 42)]
#[case(20, 7, 42)]
#[case(255, 7, 42)]
#[case(4096, 7, 42)]
// Zero and `u32::MAX` buffer index/offset, so field assembly cannot overflow unnoticed.
#[case(13, 0, 0)]
#[case(13, u32::MAX, u32::MAX)]
fn new_ref_matches_make_view(#[case] len: u32, #[case] buffer_index: u32, #[case] offset: u32) {
    let value: Vec<u8> = (0..len)
        .map(|i| u8::try_from(i % 251).vortex_expect("i % 251 fits in u8"))
        .collect();
    let prefix = [value[0], value[1], value[2], value[3]];
    let made = BinaryView::make_view(&value, buffer_index, offset);
    let built = BinaryView::new_ref(len, prefix, buffer_index, offset);
    assert_eq!(made.as_u128(), built.as_u128(), "mismatch at len {len}");
    assert!(!built.is_inlined());
    let r = built.as_view();
    assert_eq!(r.size, len);
    assert_eq!(r.prefix, prefix);
    assert_eq!(r.buffer_index, buffer_index);
    assert_eq!(r.offset, offset);
}

/// Byte strings over `{\x00, a, \xff}` up to `max_len`; `\x00` collides with zero padding.
fn values(max_len: usize) -> Vec<Vec<u8>> {
    let mut values = vec![Vec::new()];
    let mut frontier = vec![Vec::new()];
    for _ in 0..max_len {
        frontier = frontier
            .iter()
            .flat_map(|value| {
                b"\x00a\xff".iter().map(|&byte| {
                    let mut next = value.clone();
                    next.push(byte);
                    next
                })
            })
            .collect();
        values.extend(frontier.iter().cloned());
    }
    values
}

/// [`values`] up to three bytes, plus copies padded with `\x00` or `\xff` to end inside the
/// prefix, inside the inlined tail, or past the inline limit.
fn padded_values() -> Vec<Vec<u8>> {
    let short = values(3);
    let mut padded = short.clone();
    for value in &short {
        for pad in [b'\x00', b'\xff'] {
            for extra in [2, 9, 13] {
                let mut value = value.clone();
                value.resize(value.len() + extra, pad);
                padded.push(value);
            }
        }
    }
    padded
}

/// A view holding `value`, inlined when it fits and otherwise referencing buffer 0 at offset 0.
fn view_of(value: &[u8]) -> BinaryView {
    BinaryView::make_view(value, 0, 0)
}

/// A lower order prefix proves a lower value; an equal one decides nothing.
#[test]
fn order_prefix_refines_value_order() {
    let values = values(6);
    let views = values.iter().map(|value| view_of(value)).collect::<Vec<_>>();
    for (a, a_view) in values.iter().zip(&views) {
        for (b, b_view) in values.iter().zip(&views) {
            let (a_prefix, b_prefix) = (a_view.order_prefix(), b_view.order_prefix());
            if a_prefix < b_prefix {
                assert!(a < b, "prefix ordered {a:?} below {b:?}");
            }
            if a == b {
                assert_eq!(a_prefix, b_prefix, "{a:?} disagreed with itself");
            }
        }
    }
}

/// The value comparisons agree with byte order whether each side is inlined or buffer-backed.
#[test]
fn value_comparisons_match_byte_order() {
    let values = padded_values();
    for a in &values {
        let (a_view, a_buffers) = (view_of(a), [a.as_slice()]);
        for b in &values {
            let (b_view, b_buffers) = (view_of(b), [b.as_slice()]);
            assert_eq!(
                a_view.value_cmp(&a_buffers, &b_view, &b_buffers),
                a.cmp(b),
                "{a:?} against {b:?}"
            );
            assert_eq!(
                a_view.value_eq(&a_buffers, &b_view, &b_buffers),
                a == b,
                "{a:?} against {b:?}"
            );
        }
    }
}

/// The head rules equality out but never in.
#[test]
fn head_rules_out_inequality() {
    let values = values(6);
    let views = values.iter().map(|value| view_of(value)).collect::<Vec<_>>();
    for (a, a_view) in values.iter().zip(&views) {
        for (b, b_view) in values.iter().zip(&views) {
            if a == b {
                assert_eq!(a_view.head(), b_view.head());
            } else if a_view.head() == b_view.head() {
                assert_eq!(a.len(), b.len(), "head collided across lengths");
                assert_eq!(a[..4.min(a.len())], b[..4.min(b.len())]);
            }
        }
    }
}
